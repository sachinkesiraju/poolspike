"""Observation and action encodings for a learned Pool8 policy.

Observation
-----------
`encode_obs(turn)` maps a protocol `turn` message (PROTOCOL.md; the env and
`make_turn` build the same dict from engine state) to a fixed float32
vector of length `OBS_DIM`. Everything is relative to the player to move:

  balls   16 x 9   cue, 1..15 in id order: x, y (scaled to [-1, 1]),
                   on_table, role one-hot (cue / mine / theirs / 8 / open),
                   legal-target flag. Pocketed balls are zeros except role.
                   With the cue in hand (break, ball in hand) the cue's
                   x, y and on_table are 0: its old spot is meaningless.
  global  9        phase one-hot (4), my balls left / 7, opp balls left / 7,
                   open-table flag, on-the-8 flag, shot_no / 80
  pairs   15x6x4   per (object ball, pocket): pocket lane open, geometric
                   make odds from the cue (policy.shot_prob; 0 with the
                   cue in hand), cos(cut angle) from the cue (0 in hand),
                   ball-to-pocket distance / table length

`obs_layout()` names the slices.

Action
------
A learned action is (discrete index, continuous vector in [-1, 1]^6):

  discrete  idx = (ball - 1) * 7 + k, ball 1..15, k in 0..5 = pocket
            POCKET_IDS[k] (a called pot), k = 6 = no call (a legal-hit
            attempt whose intended first ball is `ball`). On the break the
            index is ignored (the reference ball is the apex).
  cont[0]   aim: offset from the reference line, AIM_MAX * u**3 degrees
            (fine near 0, up to +-30 deg for banks / kicks / safeties).
            Reference: ghost-ball line for a call, ball centre for no call,
            apex ball centre on the break. raw_phi=True: phi = 180 * (u + 1).
  cont[1]   speed, log-uniform over [0.3, 8] m/s ([0.3, 12] on the break)
  cont[2:4] side, vertical = 0.6 * u, then pulled radially inside
            hypot <= 0.7 (the PROTOCOL tip limit)
  cont[4:6] cue placement (break / ball in hand only):
            break, or abs_place=True: absolute (x, y) over the legal region
            (kitchen or whole table);
            ball in hand with a call: behind the ghost ball, distance
            d in PLACE_D (u) at angle PLACE_TH * v off the shot line;
            ball in hand, no call: polar around the target ball.
            An overlapping spot moves to the nearest legal one.

`decode` turns that into a wire action that always passes the server's
validation (`validate` runs engine.Match.parse itself), and `encode_action`
inverts a wire action back to (idx, cont, exact) where `exact` says the
round trip reproduces it (False when it lies outside the ranges above).
`mirror_turn` / `mirror_wire` reflect the table left-right (x -> TX - x,
phi -> 180 - phi, side -> -side, l <-> r pockets) for data augmentation.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import engine  # noqa: E402
import policy as P  # noqa: E402

TX, TY, R = P.TX, P.TY, P.BALL_R
HEAD = P.HEAD_STRING
BALLS = tuple(str(i) for i in range(1, 16))          # object balls, id order
BALL_IDS = ("cue",) + BALLS
POCKET_IDS = ("lb", "lc", "lt", "rb", "rc", "rt")     # mirror: k <-> (k+3)%6
PHASES = ("break", "shoot", "ball_in_hand", "ball_in_hand_kitchen")
ROLES = ("cue", "mine", "theirs", "eight", "open")
GROUPS = {"solids": P.SOLIDS, "stripes": P.STRIPES, None: None}
GROUP_NAME = {P.SOLIDS: "solids", P.STRIPES: "stripes", None: None}

N_POCKETS = 6
NO_CALL = 6
N_CALL = 7
N_DISCRETE = len(BALLS) * N_CALL                      # 105
CONT = ("aim", "speed", "side", "vertical", "place_u", "place_v")
N_CONT = len(CONT)

SPEED_MIN = 0.3
SPEED_MAX = dict(engine.SPEED_MAX)                    # break 12, shoot 8
SPIN_MAX = engine.SPIN_MAX                            # 0.6
TIP_MAX = engine.TIP_MAX                              # 0.7
AIM_MAX = 30.0                                        # deg at |u| = 1
PLACE_D = (0.01, 0.80)                                # m behind the ghost
PLACE_TH = 80.0                                       # deg off the line
EDGE = R + 0.003 + 1e-6                               # policy.placement_ok

PER_BALL = 9
N_GLOBAL = 9
PAIR_FEATS = 4
OBS_DIM = len(BALL_IDS) * PER_BALL + N_GLOBAL + len(BALLS) * N_POCKETS * PAIR_FEATS


def obs_layout() -> dict:
    nb = len(BALL_IDS) * PER_BALL
    return {"balls": slice(0, nb), "global": slice(nb, nb + N_GLOBAL),
            "pairs": slice(nb + N_GLOBAL, OBS_DIM)}


# ------------------------------------------------------------------ turns

def to_state(turn_or_balls: dict) -> dict:
    balls = turn_or_balls.get("balls", turn_or_balls)
    return {"balls": {b: {"x": float(v["x"]), "y": float(v["y"]),
                          "pocketed": bool(v["pocketed"])}
                      for b, v in balls.items()}}


def make_turn(state: dict, phase: str, me, opp, shot_no: int,
              slot: int = 0) -> dict:
    """The protocol `turn` message the server would send (engine
    Match.observation plus type/slot; the clock fields are nominal)."""
    return {"type": "turn", "slot": slot, "shot_no": shot_no, "phase": phase,
            "balls": engine.wire_balls(state),
            "my_group": GROUP_NAME[me], "opp_group": GROUP_NAME[opp],
            "legal_targets": sorted(P.targets(state, me), key=int),
            "balls_left": {"me": P.n_left(state, me) - 1,
                           "opp": P.n_left(state, opp) - 1},
            "deadline_s": 40.0, "time_bank_s": [420.0, 420.0]}


def _shim_match(turn: dict) -> engine.Match:
    """An engine.Match positioned at this turn, without racking (for the
    server's own validation code)."""
    m = engine.Match.__new__(engine.Match)
    m.state = to_state(turn)
    m.groups = {"A": GROUPS[turn["my_group"]], "B": GROUPS[turn["opp_group"]]}
    m.turn = "A"
    phase = turn["phase"]
    m.shot_no = 0 if phase == "break" else max(1, int(turn["shot_no"]))
    m.bih = {"ball_in_hand_kitchen": "kitchen",
             "ball_in_hand": "anywhere"}.get(phase)
    m.winner, m.max_shots = None, 10 ** 9
    return m


def validate(turn: dict, wire: dict) -> str | None:
    """None if the server would accept `wire` for this turn, else its
    error message (engine.Match.parse is the validator)."""
    try:
        _shim_match(turn).parse(wire)
        return None
    except engine.InvalidAction as e:
        return str(e)


# ------------------------------------------------------------ observation

def _role(b: str, me) -> int:
    if b == "cue":
        return 0
    if b == "8":
        return 3
    if me is None:
        return 4
    return 1 if b in me else 2


def encode_obs(turn: dict, pairs: bool = True) -> np.ndarray:
    """Fixed-size float32 observation (see the module docstring).
    pairs=False leaves the pair block zero (saves ~1 ms)."""
    out = np.zeros(OBS_DIM, np.float32)
    phase = turn["phase"]
    me = GROUPS[turn.get("my_group")]
    legal = set(turn.get("legal_targets") or ())
    balls = turn["balls"]
    in_hand = phase != "shoot"
    for i, b in enumerate(BALL_IDS):
        v = balls.get(b)
        o = i * PER_BALL
        out[o + 3 + _role(b, me)] = 1.0
        if v is None or v["pocketed"] or (b == "cue" and in_hand):
            continue
        out[o] = 2.0 * v["x"] / TX - 1.0
        out[o + 1] = 2.0 * v["y"] / TY - 1.0
        out[o + 2] = 1.0
        out[o + 8] = float(b in legal)
    g = len(BALL_IDS) * PER_BALL
    out[g + PHASES.index(phase)] = 1.0
    bl = turn.get("balls_left") or {}
    out[g + 4] = bl.get("me", 7) / 7.0
    out[g + 5] = bl.get("opp", 7) / 7.0
    out[g + 6] = float(me is None)
    out[g + 7] = float(legal == {"8"})
    out[g + 8] = min(1.0, turn.get("shot_no", 0) / 80.0)
    if pairs:
        st = to_state(turn)
        c = P.cue_xy(st)
        o = g + N_GLOBAL
        for i, b in enumerate(BALLS):
            v = balls.get(b)
            if v is None or v["pocketed"]:
                continue
            B = (v["x"], v["y"])
            for k, pk in enumerate(POCKET_IDS):
                j = o + (i * N_POCKETS + k) * PAIR_FEATS
                Pk = P.POCKETS[pk]
                d_op = math.dist(B, Pk) or 1e-6
                out[j] = float(P.pocket_lane_open(st, b, pk))
                out[j + 3] = d_op / TY
                if in_hand:
                    continue
                out[j + 1] = P.shot_prob(st, c, b, pk)
                ux, uy = (Pk[0] - B[0]) / d_op, (Pk[1] - B[1]) / d_op
                G = (B[0] - ux * 2 * R, B[1] - uy * 2 * R)
                vx, vy = G[0] - c[0], G[1] - c[1]
                out[j + 2] = (vx * ux + vy * uy) / (math.hypot(vx, vy) or 1e-6)
    return out


# ---------------------------------------------------------------- actions

def action_mask(turn: dict) -> np.ndarray:
    """bool[N_DISCRETE]: indices `decode` accepts without coercion. The
    break allows only (apex, no call): its index is not used."""
    m = np.zeros(N_DISCRETE, bool)
    if turn["phase"] == "break":
        m[index(apex_ball(to_state(turn)), NO_CALL)] = True
        return m
    for b in turn["legal_targets"]:
        i = BALLS.index(b) * N_CALL
        m[i:i + N_CALL] = True
    return m


def index(ball: str, k: int) -> int:
    return BALLS.index(ball) * N_CALL + k


def split(idx: int) -> tuple[str, int]:
    bi, k = divmod(int(idx), N_CALL)
    return BALLS[bi], k


def apex_ball(state: dict) -> str:
    return min(P.live(state), key=lambda b: state["balls"][b]["y"])


def _wrap180(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def _lerp(lo: float, hi: float, u: float) -> float:
    return lo + (hi - lo) * (u + 1.0) / 2.0


def _unlerp(lo: float, hi: float, x: float) -> float:
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def _region(kitchen: bool):
    return (EDGE, TX - EDGE, EDGE, (HEAD - 1e-4) if kitchen else TY - EDGE)


def legalize(state: dict, C, kitchen: bool) -> tuple[tuple, bool]:
    """Nearest legal cue spot to C (clamped into the region first), and
    whether it moved. Rings of 4 mm; the table always has room."""
    x0, x1, y0, y1 = _region(kitchen)
    cx, cy = min(max(C[0], x0), x1), min(max(C[1], y0), y1)
    moved = (cx, cy) != (C[0], C[1])
    if P.placement_ok(state, (cx, cy), kitchen):
        return (cx, cy), moved
    for r in np.arange(0.004, 1.5, 0.004):
        n = min(360, max(12, int(2 * math.pi * r / 0.004)))
        for j in range(n):
            a = 2 * math.pi * j / n
            p = (min(max(cx + r * math.cos(a), x0), x1),
                 min(max(cy + r * math.sin(a), y0), y1))
            if P.placement_ok(state, p, kitchen):
                return p, True
    import baseline  # unreachable in practice
    return baseline.free_spot(state, kitchen), True


def _ghost(state: dict, ball: str, pocket: str):
    b = state["balls"][ball]
    B = (b["x"], b["y"])
    Pk = P.POCKETS[pocket]
    d = math.dist(B, Pk) or 1e-6
    u = ((Pk[0] - B[0]) / d, (Pk[1] - B[1]) / d)
    return B, u, (B[0] - u[0] * 2 * R, B[1] - u[1] * 2 * R)


def _place(state, phase, ball, k, pu, pv, abs_place):
    kitchen = phase in ("break", "ball_in_hand_kitchen")
    if phase == "break" or abs_place:
        x0, x1, y0, y1 = _region(kitchen)
        C = (_lerp(x0, x1, pu), _lerp(y0, y1, pv))
    elif k == NO_CALL:
        b = state["balls"][ball]
        th = math.radians(180.0 * pv)
        d = 2 * R + _lerp(*PLACE_D, pu)
        C = (b["x"] + d * math.cos(th), b["y"] + d * math.sin(th))
    else:
        _, u, G = _ghost(state, ball, POCKET_IDS[k])
        th = math.atan2(-u[1], -u[0]) + math.radians(PLACE_TH * pv)
        d = _lerp(*PLACE_D, pu)
        C = (G[0] + d * math.cos(th), G[1] + d * math.sin(th))
    return legalize(state, C, kitchen)


def _unplace(state, phase, ball, k, C, abs_place):
    """(pu, pv, exact) for a recorded cue spot."""
    kitchen = phase in ("break", "ball_in_hand_kitchen")
    if phase == "break" or abs_place:
        x0, x1, y0, y1 = _region(kitchen)
        pu, pv = _unlerp(x0, x1, C[0]), _unlerp(y0, y1, C[1])
    elif k == NO_CALL:
        b = state["balls"][ball]
        w = (C[0] - b["x"], C[1] - b["y"])
        pu = _unlerp(*PLACE_D, math.hypot(*w) - 2 * R)
        pv = math.degrees(math.atan2(w[1], w[0])) / 180.0
    else:
        _, u, G = _ghost(state, ball, POCKET_IDS[k])
        w = (C[0] - G[0], C[1] - G[1])
        pu = _unlerp(*PLACE_D, math.hypot(*w))
        pv = _wrap180(math.degrees(math.atan2(w[1], w[0]))
                      - math.degrees(math.atan2(-u[1], -u[0]))) / PLACE_TH
    exact = abs(pu) <= 1.0 + 1e-9 and abs(pv) <= 1.0 + 1e-9
    return float(np.clip(pu, -1, 1)), float(np.clip(pv, -1, 1)), exact


def _phi_ref(state, phase, ball, k, C) -> float:
    if phase == "break" or k == NO_CALL:
        b = state["balls"][ball]
        return P.phi_towards(C, (b["x"], b["y"]))
    _, _, G = _ghost(state, ball, POCKET_IDS[k])
    return P.phi_towards(C, G)


def decode(turn: dict, idx: int, cont, raw_phi: bool = False,
           abs_place: bool = False) -> tuple[dict, dict]:
    """(wire action, info) for a learned action. Always protocol-valid:
    idx outside the mask is coerced to a legal ball (info["coerced"]),
    cont is clipped to [-1, 1] (NaN -> 0)."""
    u = np.clip(np.nan_to_num(np.asarray(cont, float).reshape(-1)[:N_CONT],
                              nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    state = to_state(turn)
    phase = turn["phase"]
    info = {"coerced": False, "moved_cue": False}
    if phase == "break":
        ball, k = apex_ball(state), NO_CALL
    else:
        ball, k = split(int(idx) % N_DISCRETE)
        legal = turn["legal_targets"]
        if ball not in legal:
            info["coerced"] = True
            want = state["balls"][ball]
            ball = min(legal, key=lambda b: math.dist(
                (want["x"], want["y"]),
                (state["balls"][b]["x"], state["balls"][b]["y"])))
    if phase == "shoot":
        C = P.cue_xy(state)
    else:
        C, info["moved_cue"] = _place(state, phase, ball, k, u[4], u[5],
                                      abs_place)
    if raw_phi:
        phi = 180.0 * (u[0] + 1.0)
    else:
        phi = _phi_ref(state, phase, ball, k, C) + AIM_MAX * u[0] ** 3
    vmax = SPEED_MAX["break" if phase == "break" else "shoot"]
    v = math.exp(_lerp(math.log(SPEED_MIN), math.log(vmax), u[1]))
    v = min(max(v, SPEED_MIN), vmax)
    a, b = SPIN_MAX * u[2], SPIN_MAX * u[3]
    h = math.hypot(a, b)
    if h > TIP_MAX:
        s = TIP_MAX * (1 - 1e-9) / h
        a, b = a * s, b * s
    wire = {"call": (None if phase == "break" or k == NO_CALL else
                     {"ball": ball, "pocket": POCKET_IDS[k]}),
            "stroke": {"phi": float(phi % 360.0), "speed": float(v),
                       "side": float(a), "vertical": float(b)}}
    if phase != "shoot":
        wire["cue"] = {"x": float(C[0]), "y": float(C[1])}
    info.update(ball=ball, k=k, idx=index(ball, k))
    return wire, info


def first_ball_on_line(state: dict, C, phi: float, balls) -> str | None:
    """First of `balls` whose 2R disc the cue's centre line crosses."""
    ux, uy = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    best = (math.inf, None)
    for b in balls:
        v = state["balls"][b]
        if v["pocketed"]:
            continue
        wx, wy = v["x"] - C[0], v["y"] - C[1]
        t = wx * ux + wy * uy
        if t <= 0:
            continue
        if (wx * uy - wy * ux) ** 2 <= (2 * R) ** 2 and t < best[0]:
            best = (t, b)
    return best[1]


def encode_action(turn: dict, wire: dict, raw_phi: bool = False,
                  abs_place: bool = False, target: str | None = None
                  ) -> tuple[int, np.ndarray, bool]:
    """Inverse of `decode`: (idx, cont, exact). For a no-call stroke the
    intended first ball is `target` if given, else the first legal ball on
    the cue line, else the legal ball closest in angle."""
    state = to_state(turn)
    phase = turn["phase"]
    st = wire["stroke"]
    phi, v = float(st["phi"]), float(st["speed"])
    a, b = float(st.get("side", 0.0)), float(st.get("vertical", 0.0))
    C = ((wire["cue"]["x"], wire["cue"]["y"]) if phase != "shoot"
         else P.cue_xy(state))
    call = wire.get("call")
    if phase == "break":
        ball, k = apex_ball(state), NO_CALL
    elif call:
        ball, k = str(call["ball"]), POCKET_IDS.index(call["pocket"])
    else:
        legal = turn["legal_targets"]
        ball = target if target in legal else first_ball_on_line(
            state, C, phi, legal)
        if ball is None:
            ball = min(legal, key=lambda x: abs(_wrap180(P.phi_towards(
                C, (state["balls"][x]["x"], state["balls"][x]["y"])) - phi)))
        k = NO_CALL
    exact = True
    cont = np.zeros(N_CONT, np.float64)
    if raw_phi:
        cont[0] = (phi % 360.0) / 180.0 - 1.0
    else:
        off = _wrap180(phi - _phi_ref(state, phase, ball, k, C))
        exact &= abs(off) <= AIM_MAX
        off = max(-AIM_MAX, min(AIM_MAX, off))
        cont[0] = math.copysign(abs(off / AIM_MAX) ** (1 / 3), off)
    vmax = SPEED_MAX["break" if phase == "break" else "shoot"]
    cont[1] = _unlerp(math.log(SPEED_MIN), math.log(vmax), math.log(v))
    cont[2], cont[3] = a / SPIN_MAX, b / SPIN_MAX
    if phase != "shoot":
        pu, pv, ok = _unplace(state, phase, ball, k, C, abs_place)
        cont[4], cont[5] = pu, pv
        exact &= ok
    exact &= bool(np.all(np.abs(cont) <= 1.0 + 1e-6))
    return index(ball, k), np.clip(cont, -1.0, 1.0), bool(exact)


def wire_close(w1: dict, w2: dict, tol: float = 1e-6) -> bool:
    """Same call, cue spot and stroke (phi compared on the circle)."""
    if (w1.get("call") or None) != (w2.get("call") or None):
        return False
    c1, c2 = w1.get("cue"), w2.get("cue")
    if (c1 is None) != (c2 is None):
        return False
    if c1 is not None and max(abs(c1["x"] - c2["x"]),
                              abs(c1["y"] - c2["y"])) > tol:
        return False
    s1, s2 = w1["stroke"], w2["stroke"]
    return (abs(_wrap180(s1["phi"] - s2["phi"])) <= tol
            and all(abs(float(s1.get(x, 0)) - float(s2.get(x, 0))) <= tol
                    for x in ("speed", "side", "vertical")))


# ---------------------------------------------------------------- mirror

_MIRROR_PK = {"lb": "rb", "lc": "rc", "lt": "rt",
              "rb": "lb", "rc": "lc", "rt": "lt"}


def mirror_turn(turn: dict) -> dict:
    out = dict(turn)
    out["balls"] = {b: {**v, "x": TX - v["x"]} for b, v in turn["balls"].items()}
    return out


def mirror_wire(wire: dict) -> dict:
    out = {k: v for k, v in wire.items() if k not in ("note", "meta")}
    st = wire["stroke"]
    out["stroke"] = {"phi": (180.0 - float(st["phi"])) % 360.0,
                     "speed": float(st["speed"]),
                     "side": -float(st.get("side", 0.0)),
                     "vertical": float(st.get("vertical", 0.0))}
    if wire.get("call"):
        out["call"] = {"ball": wire["call"]["ball"],
                       "pocket": _MIRROR_PK[wire["call"]["pocket"]]}
    if wire.get("cue"):
        out["cue"] = {"x": TX - wire["cue"]["x"], "y": wire["cue"]["y"]}
    return out
