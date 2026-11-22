"""A numpy MLP Pool8 policy and a websocket player that runs it.

  COWORLD_PLAYER_WS_URL=ws://... python rl/nn_player.py [--weights=W.npz]
      [--sample] [--seed=N] [--raw-phi] [--abs-place]

Network: encode_obs (rl/encode.py) -> standardize (obs_mean / obs_std)
-> tanh MLP trunk -> three heads:
  logits [105]  discrete (ball, pocket | no call); illegal ones masked
  mu [6]        continuous action, squashed with tanh (log_std for sampling)
  value [1]     shooter's win expectation in [-1, 1] (tanh), for stage (b)
Weights are an .npz with W0, b0, W1, b1, ..., Wpi, bpi, Wmu, bmu, log_std,
Wv, bv, obs_mean, obs_std (`MLPPolicy.save`); without one the net is
random-initialised from --seed. Every action goes through rl.encode.decode
and the server's own validator before it is sent; any failure (exception
or invalid action) plays the scripted baseline instead, like player.py.
The protocol loop mirrors player.py and reuses its helpers by import.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import baseline  # noqa: E402
import policy as P  # noqa: E402
from rl import encode as E  # noqa: E402

try:  # reuse the bundled player's helpers (never edit player.py)
    import player as _player
    to_state, GROUPS, RULES = _player.to_state, _player.GROUPS, _player.RULES
except Exception:  # noqa: BLE001 — player.py mid-edit: same helpers here
    _player = None
    to_state = E.to_state
    GROUPS = E.GROUPS
    RULES = {"call-shot": "wpa", "bar": "standard"}


def log(msg: str) -> None:
    print(f"[nn-player] {msg}", flush=True)


class MLPPolicy:
    """tanh MLP over encode_obs with policy (discrete + continuous) and
    value heads. Pure numpy; `forward` takes a batch."""

    def __init__(self, weights=None, hidden=(256, 256), seed: int = 0,
                 head_scale: float = 1.0, raw_phi: bool = False,
                 abs_place: bool = False):
        self.raw_phi, self.abs_place = raw_phi, abs_place
        if isinstance(weights, str):
            with np.load(weights) as z:
                weights = {k: z[k] for k in z.files}
        if weights is not None:
            self.p = {k: np.asarray(v) for k, v in weights.items()}
            self.raw_phi = bool(self.p.get("raw_phi", raw_phi))
            self.abs_place = bool(self.p.get("abs_place", abs_place))
        else:
            self.p = self.init(hidden, seed, head_scale)
        self.n_layers = sum(1 for k in self.p if k.startswith("W")
                            and k[1:].isdigit())
        if self.p["W0"].shape[0] != E.OBS_DIM:
            raise ValueError(f"weights expect obs dim {self.p['W0'].shape[0]}, "
                             f"encoder gives {E.OBS_DIM}")

    @staticmethod
    def init(hidden=(256, 256), seed: int = 0, head_scale: float = 1.0) -> dict:
        rng = np.random.default_rng(seed)
        p, d = {}, E.OBS_DIM
        for i, h in enumerate(hidden):
            p[f"W{i}"] = (rng.standard_normal((d, h)) / np.sqrt(d)).astype(np.float32)
            p[f"b{i}"] = np.zeros(h, np.float32)
            d = h
        for name, n, s in (("pi", E.N_DISCRETE, head_scale),
                           ("mu", E.N_CONT, head_scale), ("v", 1, 0.1)):
            p[f"W{name}"] = (s * rng.standard_normal((d, n)) / np.sqrt(d)
                             ).astype(np.float32)
            p[f"b{name}"] = np.zeros(n, np.float32)
        p["log_std"] = np.full(E.N_CONT, -1.0, np.float32)
        p["obs_mean"] = np.zeros(E.OBS_DIM, np.float32)
        p["obs_std"] = np.ones(E.OBS_DIM, np.float32)
        return p

    def save(self, path: str) -> None:
        np.savez(path, **self.p, raw_phi=self.raw_phi, abs_place=self.abs_place)

    def forward(self, x: np.ndarray):
        """x [B, OBS_DIM] -> (logits [B,105], mu [B,6], value [B])."""
        h = (np.asarray(x, np.float32) - self.p["obs_mean"]) / self.p["obs_std"]
        for i in range(self.n_layers):
            h = np.tanh(h @ self.p[f"W{i}"] + self.p[f"b{i}"])
        return (h @ self.p["Wpi"] + self.p["bpi"],
                h @ self.p["Wmu"] + self.p["bmu"],
                np.tanh(h @ self.p["Wv"] + self.p["bv"])[:, 0])

    def act(self, turn: dict, sample: bool = False, rng=None):
        """(wire action, aux) for one protocol turn message."""
        logits, mu, value = self.forward(E.encode_obs(turn)[None])
        mask = E.action_mask(turn)
        lg = np.where(mask, logits[0], -np.inf)
        if sample:
            rng = rng or np.random.default_rng()
            pr = np.exp(lg - lg.max())
            pr /= pr.sum()
            idx = int(rng.choice(E.N_DISCRETE, p=pr))
            z = mu[0] + np.exp(self.p["log_std"]) * rng.standard_normal(E.N_CONT)
        else:
            idx, z = int(np.argmax(lg)), mu[0]
        cont = np.tanh(z)
        wire, info = E.decode(turn, idx, cont, self.raw_phi, self.abs_place)
        ball, k = info["ball"], info["k"]
        label = ("break" if turn["phase"] == "break" else
                 f"{ball}→{E.POCKET_IDS[k]}" if k != E.NO_CALL else
                 f"no call, at {ball}")
        wire["note"] = f"nn · {label} · value {value[0]:+.2f}"
        wire["meta"] = {"kind": ("break" if turn["phase"] == "break" else
                                 "pot" if wire["call"] else "safe"),
                        "label": label, "ev": round(float(value[0] + 1) / 2, 3),
                        "search": "nn"}
        return wire, {"idx": info["idx"], "cont": cont, "value": float(value[0]),
                      **info}


def fallback_action(msg: dict) -> dict:
    """The scripted baseline for this turn (never raises)."""
    st = to_state(msg["balls"])
    try:
        act = baseline.action(st, msg["phase"], GROUPS[msg["my_group"]])
    except Exception:  # noqa: BLE001
        act = baseline.safe_action(st, msg["phase"])
    act["note"] = "baseline (nn failed)"
    return act


def choose(pol: MLPPolicy, msg: dict, sample: bool, rng) -> tuple[dict, str]:
    """(action, source): the net's action if it validates, else baseline."""
    try:
        act, _ = pol.act(msg, sample, rng)
        err = E.validate(msg, act)
        if err is None:
            return act, "nn"
        log(f"shot {msg['shot_no']}: net action invalid ({err}); baseline")
    except Exception as e:  # noqa: BLE001 — never lose a shot to a bug
        log(f"shot {msg['shot_no']}: net failed ({e!r}); baseline")
    return fallback_action(msg), "baseline"


async def play(pol: MLPPolicy, sample: bool, seed: int) -> dict:
    import websockets

    url = os.environ["COWORLD_PLAYER_WS_URL"]
    rng = np.random.default_rng(seed)
    stats = {"nn": 0, "baseline": 0, "errors": 0}
    fallback: dict = {}
    log(f"connecting (sample={sample})")
    async with websockets.connect(url, ping_timeout=None,
                                  max_size=2 ** 24) as ws:
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "hello":
                P.RULES = RULES.get(msg["settings"]["rules"], "wpa")
                log(f"slot {msg['slot']}, rules {msg['settings']['rules']}")
            elif kind == "turn":
                t0 = time.monotonic()
                act, src = choose(pol, msg, sample, rng)
                stats[src] += 1
                act.update(type="shot", shot_no=msg["shot_no"])
                fallback = {**fallback_action(msg), "type": "shot",
                            "shot_no": msg["shot_no"]}
                await ws.send(json.dumps(act))
                log(f"shot {msg['shot_no']} ({msg['phase']}) "
                    f"{time.monotonic() - t0:.2f}s [{src}]: {act.get('note')}")
            elif kind == "error":
                stats["errors"] += 1
                log(f"server rejected action: {msg.get('message')}")
                if msg.get("retries_left", 0) > 0 and \
                        msg.get("shot_no") == fallback.get("shot_no"):
                    await ws.send(json.dumps(fallback))
            elif kind == "final":
                log(f"final: {msg['results'].get('end_reason')} scores "
                    f"{msg['results'].get('scores')}; shots by source {stats}")
                return stats
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw-phi", action="store_true")
    ap.add_argument("--abs-place", action="store_true")
    a = ap.parse_args()
    pol = MLPPolicy(a.weights, seed=a.seed, raw_phi=a.raw_phi,
                    abs_place=a.abs_place)
    log(f"weights {a.weights or f'random (seed {a.seed})'}, "
        f"obs {E.OBS_DIM}, layers {pol.n_layers}")
    asyncio.run(play(pol, a.sample, a.seed))


if __name__ == "__main__":
    main()
