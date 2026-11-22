"""Single-agent Pool8 environment over the real game engine.

    env = PoolEnv(opponent="greedy")          # or "lite", "ev", "random",
    obs = env.reset(seed)                     #    "nn:<weights.npz>", callable
    while obs is not None:
        obs, reward, done, info = env.step(action)

No gym dependency. Rules, rack, turn order and execution noise are
`engine.Match` itself (the server's state machine): actions are validated
by `Match.parse` and executed by `Match.step`, so the env cannot drift from
the server. With the same seed it reproduces a server game played with a
configured `seed` (noise stream `seed * 7919 + 1`, breaker
`Random(seed ^ 0x5EED).randrange(2)`, the agent in slot 0) shot for shot;
`check_replay` proves it against recorded replays.

Observations are protocol `turn` dicts (what a websocket player receives);
`rl.encode.encode_obs` makes the vector. An action is a wire action dict, or
a learned (idx, cont) pair that `rl.encode.decode` turns into one. An
invalid wire action plays the server's fallback shot (the baseline), as the
server eventually does, and is reported in info["invalid"].

Reward: +1 win, -1 loss, 0 draw (shot limit), only on the final step.
Not modelled: the shot clock and the match wall clock (time_limit_s).

Opponents are callables `turn -> wire action`; search opponents use the
player's decision seed (shot_no * 7 + slot) and run in-process
(POOL_WORKERS=1): parallelism belongs at the game level.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import sys
from dataclasses import replace

import numpy as np

os.environ.setdefault("POOL_WORKERS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import baseline  # noqa: E402
import engine  # noqa: E402
import policy as P  # noqa: E402
from rl import encode as E  # noqa: E402

RULES = {"call-shot": "wpa", "bar": "standard"}
NOISE_KEYS = ("sigma_phi", "sigma_v", "sigma_ab", "speed_gain", "english_gain")


def noise_config(noise=None) -> P.Config:
    """Table execution noise as the server builds it (dict of NOISE_KEYS,
    a Config, or None = the default table)."""
    if isinstance(noise, P.Config):
        return noise
    return replace(P.DEFAULT, name="table",
                   **{k: float(v) for k, v in (noise or {}).items()
                      if k in NOISE_KEYS})


def breaker_slot(seed: int) -> int:
    """server.load_config's random breaker for a configured seed."""
    return random.Random(seed ^ 0x5EED).randrange(2)


# ---------------------------------------------------------------- policies

def call_to_wire(call: dict) -> dict:
    """policy call dict -> wire action (player.to_wire without the note)."""
    phi, v, a, b = call["stroke"]
    act = {"stroke": {"phi": phi % 360.0, "speed": v, "side": a,
                      "vertical": b},
           "call": ({"ball": call["ball"], "pocket": call["pocket"]}
                    if call["kind"] == "pot" else None)}
    if call.get("cue_place"):
        act["cue"] = {"x": call["cue_place"][0], "y": call["cue_place"][1]}
    act["meta"] = {"kind": call["kind"], "ev": call.get("ev"),
                   "p_make": call.get("p_make")}
    return act


def greedy_policy(turn: dict) -> dict:
    """The bundled `greedy` player / the server's baseline shot."""
    return baseline.action(E.to_state(turn), turn["phase"],
                           E.GROUPS[turn["my_group"]])


def search_policy(preset: str = "lite", noise=None, rules: str = "call-shot"):
    """The bundled EV search (`lite` or full `ev`) with the table's noise,
    seeded like player.py (shot_no * 7 + slot)."""
    cfg = replace(P.PRESETS[preset], **{k: getattr(noise_config(noise), k)
                                        for k in NOISE_KEYS})

    def act(turn: dict) -> dict:
        P.RULES = RULES[rules]
        state = E.to_state(turn)
        ctx = {"me": E.GROUPS[turn["my_group"]],
               "opp": E.GROUPS[turn["opp_group"]]}
        seed = int(turn["shot_no"]) * 7 + int(turn.get("slot", 0))
        phase = turn["phase"]
        if phase == "break":
            call = P.choose_break(state, cfg, seed)
        elif phase.startswith("ball_in_hand"):
            call = P.choose_bih(state, ctx, cfg, seed,
                                kitchen=phase == "ball_in_hand_kitchen")
        else:
            call = P.choose(state, ctx, cfg, seed)
        return call_to_wire(call)
    act.__name__ = f"{preset}_policy"
    return act


def random_policy(seed: int = 0):
    """Uniform legal (idx, cont) through the learned-action decoder."""
    def act(turn: dict) -> dict:
        rng = np.random.default_rng([seed, int(turn["shot_no"])])
        legal = np.flatnonzero(E.action_mask(turn))
        return E.decode(turn, int(rng.choice(legal)),
                        rng.uniform(-1, 1, E.N_CONT))[0]
    return act


def make_policy(spec, noise=None, rules: str = "call-shot"):
    """'greedy' | 'lite' | 'ev' | 'random[:seed]' | 'nn:<weights.npz>'
    | 'nn' (random weights) | a callable(turn) -> wire."""
    if callable(spec):
        return spec
    name, _, arg = str(spec).partition(":")
    if name == "greedy":
        return greedy_policy
    if name in ("lite", "ev", "draft"):
        if os.environ.get("POOL_WORKERS") != "1":
            os.environ["POOL_WORKERS"] = "1"  # no nested pools in workers
        return search_policy(name, noise, rules)
    if name == "random":
        return random_policy(int(arg or 0))
    if name == "nn":
        from rl.nn_player import MLPPolicy
        pol = MLPPolicy(arg or None)
        return lambda turn: pol.act(turn)[0]
    raise ValueError(f"unknown policy {spec!r}")


# --------------------------------------------------------------------- env

class ReplayDiverged(Exception):
    """A scripted replay opponent was asked for a shot the replay lacks."""


class RawCall(dict):
    """A policy call dict ({"kind", "stroke": (phi, v, a, b), ...}) that
    PoolEnv executes WITHOUT wire validation. Only for replay checks, where
    recorded executed strokes may lie outside the wire limits."""


class PoolEnv:
    """One learning agent vs one opponent policy on engine.Match.

    The agent is always protocol slot 0; agent_seat picks who breaks:
    "A" (agent breaks), "B" (opponent breaks) or "random" (the server's
    seeded breaker draw)."""

    def __init__(self, opponent="greedy", agent_seat: str = "random",
                 noise=None, rules: str = "call-shot", max_shots: int = 80,
                 raw_phi: bool = False, abs_place: bool = False):
        self.noise = noise_config(noise)
        self.rules = rules
        self.max_shots = max_shots
        self.agent_seat = agent_seat
        self.raw_phi, self.abs_place = raw_phi, abs_place
        self.opponent = make_policy(opponent, self.noise, rules)
        self.m: engine.Match | None = None
        self.history: list[dict] = []

    # ----------------------------------------------------------- helpers
    def turn(self, seat: str | None = None) -> dict:
        seat = seat or self.m.turn
        return {"type": "turn", **self.m.observation(seat),
                "slot": self.slot_of[seat], "deadline_s": 40.0,
                "time_bank_s": [420.0, 420.0]}

    @property
    def done(self) -> bool:
        return self.m is not None and self.m.done

    def _fallback(self) -> dict:
        m = self.m
        try:
            return m.parse(baseline.action(m.state, m.phase, m.groups[m.turn]))
        except Exception:  # noqa: BLE001 — server.fallback_call
            return m.parse(baseline.safe_action(m.state, m.phase))

    def _play(self, wire, source: str = "player") -> dict:
        """Validate and execute one action for the seat to move."""
        m = self.m
        seat = m.turn
        turn = self.turn(seat)
        if isinstance(wire, (tuple, list)):
            wire = E.decode(turn, wire[0], wire[1], self.raw_phi,
                            self.abs_place)[0]
        invalid = None
        if isinstance(wire, RawCall):
            call, wire = dict(wire), call_to_wire(wire)
        else:
            try:
                call = m.parse(wire)
            except engine.InvalidAction as e:
                invalid, source = str(e), "invalid"
                call = self._fallback()
        P.RULES = RULES[self.rules]
        rec = m.step(call, planned=False)
        rec.pop("shot", None)
        meta = (wire.get("meta") or {}) if isinstance(wire, dict) else {}
        entry = {"shot_no": rec["shot_no"], "seat": seat,
                 "slot": self.slot_of[seat], "turn": turn,
                 "action": {k: wire[k] for k in ("call", "stroke", "cue")
                            if isinstance(wire, dict) and k in wire},
                 "source": source, "invalid": invalid,
                 "meta": {k: meta.get(k) for k in ("ev", "p_make", "kind")},
                 "kind": call["kind"],
                 "executed": dict(zip(("phi", "speed", "side", "vertical"),
                                      rec["executed"])),
                 "result": rec["result"], "reason": rec["reason"],
                 "legal": rec["legal"], "made": rec["made"],
                 "pocketed": rec["pocketed"],
                 "first_contact": rec["first_contact"],
                 "balls_after": engine.wire_balls(rec["post_state"]),
                 "groups_after": {s: E.GROUP_NAME[g]
                                  for s, g in rec["groups_after"].items()}}
        self.history.append(entry)
        return entry

    def _run_opponent(self) -> list[dict]:
        out = []
        while not self.m.done and self.m.turn != self.seat:
            t = self.turn()
            try:
                wire = self.opponent(t)
                src = "player"
            except ReplayDiverged:
                raise
            except Exception as e:  # noqa: BLE001 — opponent bug: baseline
                wire, src = greedy_policy(t), f"opponent error: {e!r}"
            out.append(self._play(wire, src))
        return out

    def outcome(self) -> float:
        if not self.m.done or self.m.winner is None:
            return 0.0
        return 1.0 if self.m.winner == self.seat else -1.0

    # -------------------------------------------------------------- API
    def reset(self, seed: int) -> dict | None:
        """Rack game `seed`; returns the agent's first turn (None if the
        opponent broke and ran out before the agent ever shot)."""
        P.RULES = RULES[self.rules]
        self.seed = int(seed)
        if self.agent_seat == "random":
            breaker = breaker_slot(self.seed)        # agent = slot 0
        else:
            breaker = 0 if self.agent_seat == "A" else 1
        self.slot_of = {"A": breaker, "B": 1 - breaker}
        self.seat = "A" if breaker == 0 else "B"
        self.m = engine.Match(self.seed, noise=self.noise,
                              max_shots=self.max_shots)
        self.history = []
        self.opp_first = self._run_opponent()
        return None if self.m.done else self.turn()

    def step(self, action):
        """Play the agent's action, then the opponent until it is the
        agent's turn again. Returns (turn | None, reward, done, info)."""
        if self.m is None or self.m.done:
            raise RuntimeError("reset() first")
        if self.m.turn != self.seat:
            raise RuntimeError("not the agent's turn")
        mine = self._play(action)
        opp = self._run_opponent()
        done = self.m.done
        info = {"result": mine["result"], "reason": mine["reason"],
                "made": mine["made"], "invalid": mine["invalid"],
                "kind": mine["kind"], "executed": mine["executed"],
                "opp_shots": len(opp), "shots": [mine] + opp,
                "shot_no": self.m.shot_no,
                "winner_slot": (None if self.m.winner is None
                                else self.slot_of[self.m.winner]),
                "end_reason": self.m.end_reason}
        reward = self.outcome() if done else 0.0
        return (None if done else self.turn()), reward, done, info


# ---------------------------------------------------------- replay parity

def load_replay(path: str) -> dict:
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def replay_wire(shot: dict) -> dict:
    """The intended wire action recorded for one replay shot."""
    act = {"call": shot["call"], "stroke": dict(shot["intended"])}
    if shot["phase"] != "shoot":
        c = shot["balls_before"]["cue"]
        act["cue"] = {"x": c["x"], "y": c["y"]}
    return act


def _max_dev(a: dict, b: dict) -> float:
    dev = 0.0
    for k in a:
        if a[k]["pocketed"] != b[k]["pocketed"]:
            return math.inf
        if not a[k]["pocketed"]:
            dev = max(dev, abs(a[k]["x"] - b[k]["x"]),
                      abs(a[k]["y"] - b[k]["y"]))
    return dev


def check_replay(path_or_replay, mode: str = "intended") -> dict:
    """Re-play a recorded game through PoolEnv and compare every shot.

    mode "intended": the env runs with the replay's seed and table noise
    and is fed the recorded *intended* actions (agent = slot 0, opponent
    = slot 1's recorded actions), so the env's own noise stream must
    reproduce each *executed* stroke and every outcome exactly.
    mode "executed": a zero-noise env fed the recorded *executed* strokes
    straight into Match.step (they may exceed the wire limits, so they skip
    validation): physics + rules alone must reproduce the outcomes.

    Requires a replay played with a configured `seed` (local / dist games).
    Returns {"shots", "mismatches": [...], "max_dev_m", "winner_ok"}.
    """
    r = (path_or_replay if isinstance(path_or_replay, dict)
         else load_replay(path_or_replay))
    shots = r["shots"]
    by_no = {s["shot_no"]: s for s in shots}
    noise = r["settings"]["noise"]
    if mode == "executed":
        noise = {k: 0.0 for k in ("sigma_phi", "sigma_v", "sigma_ab")}

    def shot_for(turn):
        s = by_no.get(turn["shot_no"])
        if s is None or s["slot"] != turn["slot"]:
            raise ReplayDiverged(f"shot {turn['shot_no']}: the replay has "
                           f"{'no such shot' if s is None else 'the other seat'}")
        return s

    def scripted(turn):
        return replay_wire(shot_for(turn))

    env = PoolEnv(opponent=scripted, noise=noise,
                  rules=r["settings"]["rules"],
                  max_shots=r["settings"]["max_shots"],
                  agent_seat="A" if r["breaker_slot"] == 0 else "B")
    if mode == "executed":
        def scripted(turn):  # noqa: F811 — executed stroke, unvalidated
            s = shot_for(turn)
            call = {"kind": s["kind"],
                    "stroke": tuple(s["executed"][k] for k in
                                    ("phi", "speed", "side", "vertical"))}
            if s["call"]:
                call.update(ball=s["call"]["ball"], pocket=s["call"]["pocket"],
                            first=s["call"]["ball"])
            if s["phase"] != "shoot":
                c = s["balls_before"]["cue"]
                call["cue_place"] = (c["x"], c["y"])
            return RawCall(call)
        env.opponent = scripted

    out = {"shots": 0, "mismatches": [], "max_dev_m": 0.0}
    try:
        obs = env.reset(r["seed"])
        while obs is not None:
            obs, _, _, _ = env.step(scripted(obs))
    except ReplayDiverged as e:  # e.g. a replay whose noise seed was secret
        out["mismatches"].append(f"diverged: {e}")
    out["shots"] = len(env.history)
    if len(env.history) != len(shots):
        out["mismatches"].append(f"{len(env.history)} shots vs replay "
                                 f"{len(shots)}")
    for h, s in zip(env.history, shots):
        n = s["shot_no"]
        if h["slot"] != s["slot"]:
            out["mismatches"].append(f"shot {n}: slot {h['slot']} vs {s['slot']}")
        if mode == "intended" and h["executed"] != s["executed"]:
            out["mismatches"].append(f"shot {n}: executed {h['executed']} "
                                     f"vs {s['executed']}")
        for k in ("result", "reason", "pocketed", "first_contact", "made",
                  "legal"):
            if h[k] != s[k]:
                out["mismatches"].append(f"shot {n}: {k} {h[k]!r} vs {s[k]!r}")
        dev = _max_dev(h["balls_after"], s["balls_after"])
        out["max_dev_m"] = max(out["max_dev_m"], dev)
        if dev > 1e-6:
            out["mismatches"].append(f"shot {n}: table off by {dev:.3g} m")
        if mode == "intended":
            ga = {str(env.slot_of[seat]): g for seat, g in h["groups_after"].items()}
            if ga != s["groups_after"]:
                out["mismatches"].append(f"shot {n}: groups {ga} vs "
                                         f"{s['groups_after']}")
    want = r["results"]["winner_slot"]
    got = None if env.m.winner is None else env.slot_of[env.m.winner]
    out["winner_ok"] = got == want
    if got != want:
        out["mismatches"].append(f"winner slot {got} vs {want}")
    return out


if __name__ == "__main__":
    import glob
    import time
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(
        ROOT, "dist", "games", "*", "replay.json.gz")))
    for p in paths:
        t = time.time()
        for mode in ("intended", "executed"):
            res = check_replay(p, mode)
            print(f"{os.path.relpath(p, ROOT)} [{mode}]: {res['shots']} shots, "
                  f"max dev {res['max_dev_m']:.2e} m, "
                  f"{len(res['mismatches'])} mismatches"
                  + (f" {res['mismatches'][:3]}" if res["mismatches"] else "")
                  + f" ({time.time() - t:.1f}s)", flush=True)
