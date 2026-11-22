"""One 8-ball match as a step-by-step state machine (the game server's core).

Same rules, rack and execution as `game.play_game` — `Match.step` is that
loop's body, so a seeded match driven by the same decisions reproduces a
`play_game` record shot for shot (tests/test_parity.py checks this). The
difference is who decides: the server asks the seat to move, validates the
wire action, and applies ONE shared execution-noise model to every seat.
"""

from __future__ import annotations

import math
import random

import pooltool as pt

import game
import policy as P

GROUP_NAME = {P.SOLIDS: "solids", P.STRIPES: "stripes", None: None}
SPEED_MAX = {"break": 12.0, "shoot": 8.0}
SPIN_MAX = 0.6
TIP_MAX = 0.7
TRAJ_DT = 1 / 30  # replay sampling (s)


def table_info() -> dict:
    """Public table geometry (metres), sent to players and viewers."""
    pockets = {pk: {"x": float(p.center[0]), "y": float(p.center[1]),
                    "r": float(p.radius)} for pk, p in P._T.pockets.items()}
    return {"width": P.TX, "length": P.TY, "ball_radius": P.BALL_R,
            "head_string_y": P.HEAD_STRING, "pockets": pockets,
            "pocket_aim_points": {pk: list(xy) for pk, xy in P.POCKETS.items()}}


def _r(x: float) -> float:
    return round(float(x), 5)


def trajectories(shot: pt.System | None) -> dict:
    """{ball: {"t0": start, "xy": [[x, y], ...] every TRAJ_DT, "down": t}}
    for balls that move (others are static and omitted)."""
    if shot is None:
        return {}
    sys_ = pt.continuize(shot, dt=TRAJ_DT, inplace=False)
    out = {}
    for bid, ball in sys_.balls.items():
        rvw, s, t = ball.history_cts.vectorize()
        xy = rvw[:, 0, :2]
        moving = [i for i in range(1, len(t))
                  if abs(xy[i, 0] - xy[i - 1, 0]) + abs(xy[i, 1] - xy[i - 1, 1])
                  > 1e-5]
        if not moving:
            continue
        i0, i1 = max(0, moving[0] - 1), moving[-1]
        down = next((float(t[i]) for i in range(len(s)) if s[i] == 4), None)
        if down is not None:
            i1 = min(i1, next(i for i in range(len(s)) if s[i] == 4))
        out[bid] = {"t0": _r(t[i0]),
                    "xy": [[round(float(x), 4), round(float(y), 4)]
                           for x, y in xy[i0:i1 + 1]],
                    "down": None if down is None else _r(down)}
    return out


def wire_balls(state: dict) -> dict:
    # full precision: pool is chaotic, so a table rounded even to 0.01 mm
    # re-simulates to a different game (tools/audit_replays.py checks this)
    return {b: {"x": float(bl["x"]), "y": float(bl["y"]),
                "pocketed": bool(bl.get("pocketed"))}
            for b, bl in state["balls"].items()}


def warm_up() -> None:
    """Compile pooltool's numba kernels (ball, cushion and pocket events):
    a break and a soft shot, each with a replay sampling."""
    st = game.rack_state(1)
    for stroke in ((90.0, 10.0, 0.0, -0.2), (60.0, 1.5, 0.2, 0.3)):
        trajectories(P.simulate(P.with_cue(st, (P.TX / 2, 0.4)), stroke))


class InvalidAction(ValueError):
    pass


class Match:
    """Seats are "A" (breaks) and "B". `noise` is the execution model every
    stroke gets, whoever plays it."""

    def __init__(self, seed: int, noise: P.Config = P.DEFAULT,
                 max_shots: int = 80, noise_seed: int | None = None):
        self.seed = seed
        self.noise = noise
        self.max_shots = max_shots
        # default noise stream = game.play_game's (parity); the server passes
        # a secret noise_seed so the public rack can't predict it
        self.rng = random.Random(seed * 7919 + 1 if noise_seed is None
                                 else noise_seed)
        self.state = game.rack_state(seed)
        self.groups = {"A": None, "B": None}
        self.turn = "A"
        self.bih = None
        self.shot_no = 0
        self.winner = None
        self.end_reason = None
        self.log: list[dict] = []

    # ------------------------------------------------------------ queries
    @property
    def done(self) -> bool:
        return self.winner is not None or self.shot_no > self.max_shots

    @property
    def phase(self) -> str:
        if self.shot_no == 0:
            return "break"
        if self.bih == "kitchen":
            return "ball_in_hand_kitchen"
        if self.bih:
            return "ball_in_hand"
        return "shoot"

    def other(self, seat: str) -> str:
        return "B" if seat == "A" else "A"

    def ctx(self, seat: str) -> dict:
        return {"me": self.groups[seat], "opp": self.groups[self.other(seat)]}

    def observation(self, seat: str) -> dict:
        """What the seat to move sees (the whole table is public)."""
        me, opp = self.groups[seat], self.groups[self.other(seat)]
        return {"shot_no": self.shot_no, "phase": self.phase,
                "balls": wire_balls(self.state),
                "my_group": GROUP_NAME[me], "opp_group": GROUP_NAME[opp],
                "legal_targets": sorted(P.targets(self.state, me), key=int),
                "balls_left": {"me": P.n_left(self.state, me) - 1,
                               "opp": P.n_left(self.state, opp) - 1}}

    # ---------------------------------------------------------- validation
    def parse(self, msg: dict) -> dict:
        """Wire action -> policy call dict, or InvalidAction (with a reason
        the player is told)."""
        if not isinstance(msg, dict):
            raise InvalidAction("action must be a JSON object")
        st = msg.get("stroke")
        if not isinstance(st, dict):
            raise InvalidAction("missing stroke object")
        try:
            phi = float(st["phi"])
            v = float(st["speed"])
            a = float(st.get("side", 0.0))
            b = float(st.get("vertical", 0.0))
        except (KeyError, TypeError, ValueError):
            raise InvalidAction("stroke needs numeric phi, speed "
                                "(side, vertical optional)") from None
        if not all(math.isfinite(x) for x in (phi, v, a, b)):
            raise InvalidAction("stroke values must be finite")
        vmax = SPEED_MAX["break" if self.phase == "break" else "shoot"]
        if not 0.3 <= v <= vmax:
            raise InvalidAction(f"speed must be in [0.3, {vmax}] m/s")
        if abs(a) > SPIN_MAX or abs(b) > SPIN_MAX:
            raise InvalidAction(f"side/vertical must be in [-{SPIN_MAX}, "
                                f"{SPIN_MAX}]")
        if math.hypot(a, b) > TIP_MAX:  # past this the tip would miscue
            raise InvalidAction(f"tip offset hypot(side, vertical) must be "
                                f"≤ {TIP_MAX}")
        call: dict = {"stroke": (phi % 360.0, v, a, b)}

        phase = self.phase
        cue = msg.get("cue")
        if phase == "shoot":
            if cue is not None:
                raise InvalidAction("cue placement only with ball in hand "
                                    "or on the break")
        else:
            try:
                xy = (float(cue["x"]), float(cue["y"]))
            except (KeyError, TypeError, ValueError):
                raise InvalidAction(f"phase {phase} needs cue {{x, y}}") from None
            kitchen = phase in ("break", "ball_in_hand_kitchen")
            if not P.placement_ok(self.state, xy, kitchen):
                raise InvalidAction(
                    "cue placement off the table, overlapping a ball"
                    + (", or not behind the head string" if kitchen else ""))
            call["cue_place"] = xy

        if phase == "break":
            call["kind"] = "break"
            return call
        c = msg.get("call")
        if c is None:
            call["kind"] = "safe"
            return call
        try:
            ball, pocket = str(c["ball"]), str(c["pocket"])
        except (KeyError, TypeError):
            raise InvalidAction("call must be null or {ball, pocket}") from None
        if pocket not in P.POCKETS:
            raise InvalidAction(f"pocket must be one of {sorted(P.POCKETS)}")
        legal = P.targets(self.state, self.groups[self.turn])
        if ball not in legal:
            raise InvalidAction(f"ball {ball} is not yours to call "
                                f"(legal: {sorted(legal, key=int)})")
        call.update(kind="pot", ball=ball, pocket=pocket, first=ball)
        return call

    # ---------------------------------------------------------------- step
    def step(self, call: dict, planned: bool = True) -> dict:
        """Execute one validated call for the seat to move (game.play_game's
        loop body) and return the shot record."""
        turn, other = self.turn, self.other(self.turn)
        ctx = self.ctx(turn)
        state = self.state
        if call.get("cue_place"):
            state = P.with_cue(state, call["cue_place"])
        executed = P.noisy(call["stroke"], self.rng, self.noise)
        shot = P.simulate(state, executed)
        res = P.summarize(shot, state)
        j = P.adjudicate(state, ctx, call, res)
        made = (call["kind"] == "pot" and res is not None
                and res["pm"].get(call["ball"]) == call["pocket"])
        rec = {"shot_no": self.shot_no, "player": turn, "phase": self.phase,
               "groups_before": dict(self.groups), "bih": self.bih,
               "state": state, "call": call, "stroke": list(call["stroke"]),
               "executed": list(executed),
               "pocketed": {} if res is None else res["pm"],
               "first_contact": None if res is None else res["fc"],
               "result": j["result"], "legal": j["legal"],
               "reason": j["reason"], "made": made, "shot": shot}
        self.groups[turn], self.groups[other] = j["me"], j["opp"]
        self.state = j["state"]
        self.bih = None
        r = j["result"]
        if r == "win":
            self.winner, self.end_reason = turn, "8 legally pocketed"
        elif r == "loss":
            self.winner, self.end_reason = other, j["reason"]
        elif r == "foul":
            self.bih = "kitchen" if self.shot_no == 0 else "anywhere"
            self.turn = other
        elif r == "turnover":
            self.turn = other
        rec["post_state"] = self.state
        rec["groups_after"] = dict(self.groups)
        if planned and call["kind"] in ("pot", "safe", "break"):
            rec["planned"] = trajectories(P.simulate(state, call["stroke"]))
        self.log.append(rec)
        self.shot_no += 1
        if self.shot_no > self.max_shots and self.winner is None:
            self.end_reason = f"shot limit ({self.max_shots + 1} shots)"
        return rec

    def end_on_time(self, bank: dict) -> None:
        """Match clock expired. `bank` is each seat's unused thinking time:
        the seat that spent more of its clock loses, so running the clock
        down never pays; within a second of each other it is a draw. (Balls
        left would reward a leader for stalling with slow safeties.)"""
        a, b = bank["A"], bank["B"]
        self.winner = None if abs(a - b) < 1.0 else ("A" if a > b else "B")
        self.end_reason = (f"time limit: {a:.0f}s–{b:.0f}s of clock left"
                           + ("" if self.winner else ", draw"))
        self.shot_no = self.max_shots + 1  # done

    # ------------------------------------------------------------- outputs
    def wire_record(self, rec: dict, slot_of: dict) -> dict:
        """Public JSON for one shot: broadcast live and stored in the replay."""
        call = rec["call"]
        phi, v, a, b = rec["stroke"]
        ephi, ev, ea, eb = rec["executed"]
        gb = rec["groups_before"]
        return {
            "shot_no": rec["shot_no"], "slot": slot_of[rec["player"]],
            "phase": rec["phase"],
            "groups_before": {str(slot_of[s]): GROUP_NAME[g]
                              for s, g in gb.items()},
            "groups_after": {str(slot_of[s]): GROUP_NAME[g]
                             for s, g in rec["groups_after"].items()},
            "balls_before": wire_balls(rec["state"]),
            "call": ({"ball": call["ball"], "pocket": call["pocket"]}
                     if call["kind"] == "pot" else None),
            "kind": call["kind"],
            "intended": {"phi": float(phi), "speed": float(v),
                         "side": float(a), "vertical": float(b)},
            "executed": {"phi": float(ephi), "speed": float(ev),
                         "side": float(ea), "vertical": float(eb)},
            "result": rec["result"], "legal": rec["legal"],
            "reason": rec["reason"], "made": rec["made"],
            "pocketed": rec["pocketed"], "first_contact": rec["first_contact"],
            "balls_after": wire_balls(rec["post_state"]),
            "traj": trajectories(rec["shot"]),
            "planned": rec.get("planned", {}),
        }
