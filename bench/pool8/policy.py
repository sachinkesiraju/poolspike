"""Noise-aware expected-value policy for full 8-ball games.

Skill model: every executed stroke gets Gaussian noise on aim (phi), speed
(V) and tip offset (a, b) — `Config`. The game applies exactly this noise when
it executes a call, so a make-probability here is the real odds in the game,
not a separate guess that execution never tests.

Decision = argmax expected value over everything the search finds:
  * pots — direct, bank, kick, combo lanes, each with a speed × draw/follow ×
    english sweep so the stroke is chosen for the position it delivers;
  * safeties — a contact-fraction × speed × spin sweep on every legal ball,
    plus a full 360° aim sweep (kick escapes) whenever direct contacts are
    scarce;
  * ball in hand — cue placement searched, not a fixed kitchen spot;
  * the break — placement/speed/spin chosen by sampled outcomes.

EV of a candidate = mean over noisy executions of the value of what happens,
under ONE rules function (`adjudicate`) shared with the game itself:
  legal own pot → V(me to move); miss → 1 − V(opponent to move);
  foul → 1 − V(opponent with ball in hand); 8-ball outcomes terminal (1 / 0).

V(state) (`value`) is a leave evaluator: the mover's best next-shot
probability (geometric, calibrated against sim pot basins — see
`shot_prob`), how open each side's remaining balls are, and a two-player
race model (`_race`) over balls left. The policy itself never trusts the
geometry for its own shot: every candidate is sim-verified and its make
probability comes from the measured aim basin under `Config` noise.
"""

from __future__ import annotations

import functools
import math
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from dataclasses import replace  # noqa: F401 — re-exported for callers

import pooltool as pt

import fastsim  # identical-result speed patches for pooltool
from pool_env import (build_system, first_contact, ghost_ball_aim,
                      phi_towards, pocket_map, rail_after_contact)

BALL_R = pt.BallParams().R
_T = pt.Table.default()
TX, TY = float(_T.w), float(_T.l)
POCKETS = {
    "lb": (0.0, 0.0), "lt": (0.0, TY), "rb": (TX, 0.0), "rt": (TX, TY),
    "lc": (0.0, TY / 2), "rc": (TX, TY / 2),
}
# inward pocket normals (pocket -> table), for approach-angle feasibility
_NORMAL = {
    "lb": (0.7071, 0.7071), "lt": (0.7071, -0.7071),
    "rb": (-0.7071, 0.7071), "rt": (-0.7071, -0.7071),
    "lc": (1.0, 0.0), "rc": (-1.0, 0.0),
}
SOLIDS = tuple(str(i) for i in range(1, 8))
STRIPES = tuple(str(i) for i in range(9, 16))
HEAD_STRING = TY / 4  # kitchen: y < HEAD_STRING
# "wpa" (default, call your shot): only the called ball in the called pocket
# keeps you at the table; an uncalled ball that drops (on a safety or slop)
# stays down but the turn ends. "standard" (bar rules): any legally pocketed ball
# of your group keeps you shooting. Under "standard" the search learns to
# play slop "safeties" that drop a ball, so it stays opt-in.
RULES = os.environ.get("POOL_RULES", "wpa")


# Effective pocket half-width (m) seen by the object ball, fit to sim aim
# basins on 122 random single-ball shots (demo/calib: median 33mm corners
# across approach angles, 34mm sides straight-on tapering to 0 by ~65deg).
W_CORNER, W_SIDE = 0.033, 0.034



@dataclass(frozen=True)
class Config:
    """One player = one strategy. Every knob a tournament / autoresearch loop
    should be able to vary lives here; nothing else in the module is tuned.

    Execution noise (1 sigma) is the player's skill: phi in degrees, v
    relative, a/b in tip-offset units (fraction of R). The game executes this
    player's strokes with exactly this noise.
    """
    name: str = "ev"
    sigma_phi: float = 0.10
    sigma_v: float = 0.025
    sigma_ab: float = 0.025
    # race model: continuation make-rate for a fully blocked vs fully open
    # group, and the opponent's make-rate when facing a played safety
    q_lo: float = 0.50
    q_hi: float = 0.86
    q_safe: float = 0.30
    # search breadth / strategy toggles
    # call-your-shot without safeties (house rules): every shot is a called
    # pot — a pure defensive hit only when nothing at all can be called.
    # A pot's EV still prices the leave if it misses, so hard shots are
    # chosen with the miss in mind. True = also weigh deliberate safeties.
    safeties: bool = False
    position: bool = True       # sweep speed/spin for shape (False = stun)
    deep_below: float = 0.8     # banks/kicks/combos when best pot is below
    shortlist: int = 12         # strokes that get a measured aim basin
    finalists: int = 6          # pots that get sampled EV
    n_pot_mc: int = 8
    n_safe_mc: int = 6
    # model-risk discount per collision the stroke noise doesn't cover: real
    # tables add ball/cushion variation at every contact, so a kiss through a
    # cluster that the sim makes 50/50 is not a real 100% shot
    risk_kiss: float = 0.35     # called ball touches another ball en route
    risk_rail: float = 0.20     # per cushion the called ball banks off
    risk_kick: float = 0.20     # per cushion the cue hits before contact
    risk_combo: float = 0.20    # carrier ball in a combo
    # aim error grows with pace and side spin (a real stroke's does):
    # sigma_phi × (1 + speed_gain·(V − 2.5)⁺ + english_gain·|a|)
    speed_gain: float = 0.20
    english_gain: float = 1.5
    # ball in hand is worth at least this next-shot chance: the cue can be
    # placed for a bank, a breakout or a perfect angle even when no ball
    # has a straight path (without it, fouling looked nearly free)
    bih_floor: float = 0.55
    # compute budget (defaults = full search). max_lanes > 0 keeps only the
    # geometrically most promising lanes per search stage; exec_grid "lite"
    # sweeps fewer speed/spin strokes; break_samples < 8 also thins the
    # break options. Used by the hosted (fraction-of-a-core) preset.
    max_lanes: int = 0
    exec_grid: str = "full"
    break_samples: int = 16
    # refine every candidate whose first-pass EV is within this band of the
    # leader, not just the leader: a runner-up valued on 8 noisy samples can
    # be a near-sure pot that only looks worse by sampling luck
    refine_close: float = 0.08
    # EVs within this of each other are a tie: take the more makeable shot
    tie_eps: float = 0.01
    # learned next-shot odds for value() (leave_net.py); "" = geometric p1
    leave_net: str = ""


DEFAULT = Config()
# fast preset for iterating on experiments: same strategy, thinner search
DRAFT = Config(name="draft", shortlist=6, finalists=3, n_pot_mc=6,
               n_safe_mc=4, deep_below=0.5)
# hosted preset: same strategy and value model on a ~10x smaller search, for
# a player container with a fraction of one CPU and a per-shot clock
LITE = Config(name="lite", shortlist=4, finalists=2, n_pot_mc=4,
              n_safe_mc=3, deep_below=0.5, max_lanes=8, exec_grid="lite",
              break_samples=3)
# pattern player (pattern.py): the full search, then a second ply on its
# best pots; this preset is the first ply
PATTERN = Config(name="pattern")
PRO = Config(name="pro")
SAFE = Config(name="safe", safeties=True)  # ev, weighing safeties
PLUS = Config(name="plus", safeties=True)  # safe + pro.py break book
LEAVE = Config(name="leave", leave_net=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "leave_net.npz"))
PRESETS = {"ev": DEFAULT, "draft": DRAFT, "lite": LITE, "pattern": PATTERN,
           "pro": PRO, "safe": SAFE,
           "plus": PLUS,
           "leave": LEAVE}

# stroke grids
EXEC_SPEEDS = (1.2, 1.8, 2.5, 3.3, 4.3)
# position sweep: draw/follow at every pace, plus left/right english ×
# draw/stun/follow at the three working paces. Each stroke gets its own
# solved aim; english costs accuracy (Config.english_gain), so it is used
# only when the leave pays for it.
EXEC_GRID = ([(v, 0.0, b) for v in EXEC_SPEEDS
              for b in (-0.5, -0.25, 0.0, 0.25, 0.5)]
             + [(v, a, b) for v in (1.8, 2.5, 3.3) for a in (-0.3, 0.3)
                for b in (-0.35, 0.0, 0.35)])
EXEC_DEEP = [(v, 0.0, b) for v in (2.2, 3.0, 3.8) for b in (-0.3, 0.0, 0.3)]
EXEC_LITE = ([(v, 0.0, b) for v in (1.4, 2.4, 3.6) for b in (-0.4, 0.0, 0.4)]
             + [(2.4, a, b) for a in (-0.3, 0.3) for b in (-0.35, 0.35)])
SAFE_SPEEDS = (0.9, 1.6, 2.6)
SAFE_FRACS = (-0.85, -0.5, -0.2, 0.0, 0.2, 0.5, 0.85)
SAFE_SPINS = (-0.4, 0.3)
SWEEP_SPEEDS = (1.2, 2.4)
SWEEP_STEP = 1.0  # deg, full-circle aim sweep for kick safeties


# --------------------------------------------------------------- state utils

def live(state: dict) -> set[str]:
    return {b for b, bl in state["balls"].items()
            if b != "cue" and not bl.get("pocketed")}


def group_of(ball: str) -> tuple[str, ...]:
    return SOLIDS if int(ball) < 8 else STRIPES


def targets(state: dict, group) -> set[str]:
    """Balls the mover may contact first (open table: anything but the 8)."""
    lv = live(state)
    if group is None:
        # open table with only the 8 left (every other ball went down
        # uncalled): the 8 is the shot for both sides
        return (lv - {"8"}) or (lv & {"8"})
    mine = lv & set(group)
    return mine if mine else (lv & {"8"})


def n_left(state: dict, group) -> int:
    """Shots still needed to win: own balls + the 8."""
    lv = live(state)
    if group is None:
        return min(len(lv & set(SOLIDS)), len(lv & set(STRIPES))) + 1
    return len(lv & set(group)) + 1


def cue_xy(state: dict) -> tuple[float, float]:
    c = state["balls"]["cue"]
    return (c["x"], c["y"])


def with_cue(state: dict, xy) -> dict:
    balls = {b: dict(bl) for b, bl in state["balls"].items()}
    balls["cue"] = {"x": float(xy[0]), "y": float(xy[1]), "pocketed": False}
    return {"balls": balls}


def respot_eight(state: dict) -> dict:
    """8 back on the foot spot (or the nearest free point behind it)."""
    balls = {b: dict(bl) for b, bl in state["balls"].items()}
    x, y = TX / 2, TY * 0.75
    while any(not bl.get("pocketed") and b != "8"
              and math.hypot(bl["x"] - x, bl["y"] - y) < 2 * BALL_R + 1e-3
              for b, bl in balls.items()):
        y += 0.01
    balls["8"] = {"x": x, "y": y, "pocketed": False}
    return {"balls": balls}


def sigma_phi(stroke, cfg: Config = DEFAULT) -> float:
    """Aim noise (deg) for this stroke: harder pace and more english cost
    accuracy, so the policy only reaches for them when position pays."""
    _, v, a, _ = stroke
    return cfg.sigma_phi * (1 + cfg.speed_gain * max(0.0, v - 2.5)
                            + cfg.english_gain * abs(a))


def noisy(stroke, rng: random.Random, cfg: Config = DEFAULT):
    phi, v, a, b = stroke
    return (phi + rng.gauss(0, sigma_phi(stroke, cfg)),
            max(0.3, v * (1 + rng.gauss(0, cfg.sigma_v))),
            max(-0.6, min(0.6, a + rng.gauss(0, cfg.sigma_ab))),
            max(-0.6, min(0.6, b + rng.gauss(0, cfg.sigma_ab))))


# ---------------------------------------------------------------- simulation

def simulate(state: dict, stroke, t_final: float | None = None
             ) -> pt.System | None:
    phi, v, a, b = stroke
    try:
        sys_ = build_system(state)
        sys_.cue.set_state(V0=v, phi=phi, theta=0.0, a=a, b=b)
        if t_final is None:
            pt.simulate(sys_, inplace=True)
        else:
            pt.simulate(sys_, inplace=True, t_final=t_final)
        return sys_
    except Exception:  # noqa: BLE001 — degenerate layouts assert in pooltool
        return None


def post_state(shot: pt.System, state: dict) -> dict:
    down = set(pocket_map(shot))
    balls = {b: dict(bl) for b, bl in state["balls"].items()
             if bl.get("pocketed")}
    for bid, bl in shot.balls.items():
        balls[bid] = {"x": float(bl.state.rvw[0, 0]),
                      "y": float(bl.state.rvw[0, 1]),
                      "pocketed": bid in down}
    return {"balls": balls}


def summarize(shot: pt.System | None, state: dict) -> dict | None:
    """Everything the rules need from one simulated stroke."""
    if shot is None:
        return None
    return {"pm": pocket_map(shot), "fc": first_contact(shot),
            "rail": rail_after_contact(shot),
            "post": post_state(shot, state)}


# --------------------------------------------------------------------- rules

def adjudicate(state: dict, ctx: dict, call: dict, res: dict | None) -> dict:
    """Single 8-ball rules function, shared by the game and the policy's EV.

    ctx = {"me": group|None, "opp": group|None} (None/None = open table).
    call = {"kind": pot|safe|break, "ball", "pocket"}.
    Returns {result: win|loss|continue|turnover|foul, legal, reason,
             state (post), me, opp (groups after the shot)}.
    """
    me, opp = ctx["me"], ctx["opp"]
    if res is None:
        return {"result": "foul", "legal": False, "reason": "no stroke",
                "state": state, "me": me, "opp": opp}
    pm, fc, post = res["pm"], res["fc"], res["post"]
    scratch = "cue" in pm
    potted = set(pm) - {"cue"}
    out = {"state": post, "me": me, "opp": opp}

    if call["kind"] == "break":
        if "8" in potted:
            post = respot_eight(post)
            out["state"] = post
        if scratch:
            return {**out, "result": "foul", "legal": False,
                    "reason": "scratch on the break"}
        # open table unless the breaker pockets a ball: then the breaker
        # takes the group of the first ball down and keeps shooting
        first = next((b for b in pm if b not in ("cue", "8")), None)
        if first is not None:
            out["me"] = group_of(first)
            out["opp"] = SOLIDS if out["me"] == STRIPES else STRIPES
        return {**out, "legal": True,
                "result": "continue" if potted else "turnover",
                "reason": None}

    tg = targets(state, me)
    on8 = tg == {"8"}
    legal = (fc in tg and (res["rail"] or bool(potted)) and not scratch)
    reason = None
    if not legal:
        reason = ("scratch" if scratch else
                  "no contact" if fc is None else
                  f"wrong ball first ({fc})" if fc not in tg else
                  "no rail after contact")
    if "8" in potted:
        won = (on8 and legal and call["kind"] == "pot"
               and call.get("ball") == "8" and pm["8"] == call.get("pocket"))
        return {**out, "legal": legal, "result": "win" if won else "loss",
                "reason": None if won else (
                    "8 early" if not on8 else
                    "scratch on the 8" if scratch else
                    "8 in the wrong pocket" if legal else reason)}
    if not legal:
        return {**out, "legal": False, "result": "foul", "reason": reason}
    made = (call["kind"] == "pot" and pm.get(call.get("ball"))
            == call.get("pocket"))
    own = sorted((potted & set(me)) if me else (potted - {"8"}), key=int)
    keep = made or (RULES == "standard" and bool(own))
    if keep and me is None:
        out["me"] = group_of(call["ball"] if made else own[0])
        out["opp"] = SOLIDS if out["me"] == STRIPES else STRIPES
    return {**out, "legal": True, "reason": None,
            "result": "continue" if keep else "turnover"}


# ------------------------------------------------------------ leave evaluator

def _seg_clear(p1, p2, state: dict, ignore: set[str],
               corridor: float = 2 * BALL_R * 0.97) -> bool:
    x1, y1 = p1
    dx, dy = p2[0] - x1, p2[1] - y1
    L2 = dx * dx + dy * dy
    for bid, b in state["balls"].items():
        if bid in ignore or b.get("pocketed"):
            continue
        bx, by = b["x"], b["y"]
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((bx - x1) * dx
                                                   + (by - y1) * dy) / L2))
        if math.hypot(bx - (x1 + t * dx), by - (y1 + t * dy)) < corridor:
            return False
    return True


def _pocket_width(ball_xy, pk: str) -> float:
    """Effective half-width (m) for this approach angle; 0 = can't enter."""
    P = POCKETS[pk]
    d = math.dist(ball_xy, P) or 1e-6
    ux, uy = (P[0] - ball_xy[0]) / d, (P[1] - ball_xy[1]) / d
    nx, ny = _NORMAL[pk]
    beta = math.degrees(math.acos(max(-1.0, min(1.0, -(ux * nx + uy * ny)))))
    if pk in ("lc", "rc"):
        return W_SIDE if beta <= 30 else W_SIDE * max(0.0, (65 - beta) / 35)
    return W_CORNER if beta <= 50 else 0.0


def pocket_lane_open(state: dict, ball: str, pk: str) -> bool:
    b = state["balls"][ball]
    bxy = (b["x"], b["y"])
    return (_pocket_width(bxy, pk) > 0
            and _seg_clear(bxy, POCKETS[pk], state, {ball, "cue"}))


def shot_prob(state: dict, cxy, ball: str, pk: str,
              cfg: Config = DEFAULT) -> float:
    """Geometric make-probability of a direct pot under the player's aim noise.

    Object-ball tolerance tau = w/d_op; a cue-aim error dphi rotates the
    object line by dphi·(d_c/(2R·cos(cut)) − 1), so the cue-aim tolerance is
    tau / that factor. Used only to value leaves — never the policy's own
    shot, which is sim-verified with a measured basin.
    """
    b = state["balls"][ball]
    B = (b["x"], b["y"])
    P = POCKETS[pk]
    w = _pocket_width(B, pk)
    if w <= 0:
        return 0.0
    d_op = math.dist(B, P) or 1e-6
    ux, uy = (P[0] - B[0]) / d_op, (P[1] - B[1]) / d_op
    G = (B[0] - ux * 2 * BALL_R, B[1] - uy * 2 * BALL_R)
    if not (BALL_R * 0.5 < G[0] < TX - BALL_R * 0.5
            and BALL_R * 0.5 < G[1] < TY - BALL_R * 0.5):
        return 0.0
    vx, vy = G[0] - cxy[0], G[1] - cxy[1]
    d_c = math.hypot(vx, vy) or 1e-6
    cos_a = (vx * ux + vy * uy) / d_c
    if cos_a < math.cos(math.radians(80)):
        return 0.0
    if not _seg_clear(B, P, state, {ball, "cue"}):
        return 0.0
    if not _seg_clear(cxy, G, state, {ball, "cue"}):
        return 0.0
    k = max(0.25, d_c / (2 * BALL_R * cos_a) - 1)
    dphi = math.degrees((w / d_op) / k)
    return min(0.99, math.erf(dphi / (cfg.sigma_phi * math.sqrt(2))))


def bank_prob(state: dict, cxy, ball: str, pk: str,
              cfg: Config = DEFAULT) -> float:
    """Best one-rail bank of `ball` into `pk` (geometric, mirror-pocket
    model, same tolerance formula as shot_prob over the unfolded path),
    discounted by cfg.risk_rail. Lets a leave with "no straight-in pot"
    still count the banks a real opponent would shoot."""
    b = state["balls"][ball]
    B = (b["x"], b["y"])
    P = POCKETS[pk]
    best = 0.0
    for rail in ("L", "R", "B", "T"):
        M = _mirror(P, rail)
        d = math.dist(B, M) or 1e-6
        ux, uy = (M[0] - B[0]) / d, (M[1] - B[1]) / d
        # where the object ball meets the rail line
        if rail in ("L", "R"):
            xr = 0.0 if rail == "L" else TX
            if abs(ux) < 1e-6:
                continue
            t = (xr - B[0]) / ux
        else:
            yr = 0.0 if rail == "B" else TY
            if abs(uy) < 1e-6:
                continue
            t = (yr - B[1]) / uy
        if t <= 0 or t >= d:
            continue
        K = (B[0] + ux * t, B[1] + uy * t)
        if not (0.12 < K[0] < TX - 0.12 or rail in ("L", "R")):
            continue
        if not (0.12 < K[1] < TY - 0.12 or rail in ("B", "T")):
            continue
        if abs(K[1] - TY / 2) < 0.1 and rail in ("L", "R"):
            continue  # into the side-pocket mouth, not a cushion
        w = _pocket_width(K, pk)
        if w <= 0 or not _seg_clear(B, K, state, {ball, "cue"}) \
                or not _seg_clear(K, P, state, {ball, "cue"}):
            continue
        G = (B[0] - ux * 2 * BALL_R, B[1] - uy * 2 * BALL_R)
        vx, vy = G[0] - cxy[0], G[1] - cxy[1]
        d_c = math.hypot(vx, vy) or 1e-6
        cos_a = (vx * ux + vy * uy) / d_c
        if cos_a < math.cos(math.radians(80)) or \
                not _seg_clear(cxy, G, state, {ball, "cue"}):
            continue
        k = max(0.25, d_c / (2 * BALL_R * cos_a) - 1)
        dphi = math.degrees((w / d) / k)
        p = math.erf(dphi / (cfg.sigma_phi * math.sqrt(2)))
        best = max(best, p * (1 - cfg.risk_rail))
    return best


def bih_prob(state: dict, ball: str, kitchen: bool = False,
             cfg: Config = DEFAULT) -> float:
    """Best pot probability with the cue placed freely (geometric): straight
    in from behind the ghost ball, or a one-rail bank with the cue placed
    on the bank line."""
    best = 0.0
    b = state["balls"][ball]
    B = (b["x"], b["y"])
    if kitchen and B[1] < HEAD_STRING:
        return 0.0
    for pk, P in POCKETS.items():
        targets_ = [(P, None)] + [(_mirror(P, r), r) for r in ("L", "R", "B", "T")]
        for T, rail in targets_:
            if rail is None and not pocket_lane_open(state, ball, pk):
                continue
            d_op = math.dist(B, T) or 1e-6
            ux, uy = (T[0] - B[0]) / d_op, (T[1] - B[1]) / d_op
            for dist in (0.25, 0.12):
                C = (B[0] - ux * (2 * BALL_R + dist),
                     B[1] - uy * (2 * BALL_R + dist))
                if kitchen:
                    C = (C[0], min(C[1], HEAD_STRING - 0.01))
                if not placement_ok(state, C, kitchen):
                    continue
                st = with_cue(state, C)
                p = (shot_prob(st, C, ball, pk, cfg) if rail is None
                     else bank_prob(st, C, ball, pk, cfg))
                best = max(best, p)
                break
    return best


def openness(state: dict, balls) -> float:
    balls = list(balls)
    if not balls:
        return 1.0
    return sum(any(pocket_lane_open(state, b, pk) for pk in POCKETS)
               for b in balls) / len(balls)


@functools.lru_cache(maxsize=8192)
def _race_table(q1: float, q2: float):
    """X[n][m] = P(player to move wins) with n vs m shots left, per-shot
    make-rates q1 (mover) / q2 (other), a miss handing the table over."""
    X = [[0.0] * 9 for _ in range(9)]
    Y = [[0.0] * 9 for _ in range(9)]
    for n in range(9):
        X[0][n] = Y[0][n] = 1.0
    for _ in range(400):
        delta = 0.0
        for n in range(1, 9):
            for m in range(1, 9):
                x = q1 * X[n - 1][m] + (1 - q1) * (1 - Y[m][n])
                y = q2 * Y[m - 1][n] + (1 - q2) * (1 - X[n][m])
                delta = max(delta, abs(x - X[n][m]), abs(y - Y[m][n]))
                X[n][m], Y[m][n] = x, y
        if delta < 1e-7:
            break
    return X, Y


def _race(n: int, m: int, q1: float, q2: float) -> float:
    n, m = max(0, min(8, n)), max(0, min(8, m))
    if n == 0:
        return 1.0
    if m == 0:
        return 0.0
    return _race_table(round(q1, 2), round(q2, 2))[0][n][m]


def value(state: dict, me, opp, bih: bool = False,
          kitchen: bool = False, cfg: Config = DEFAULT) -> float:
    """Win probability for the player to move (`me`) in `state`."""
    tg = targets(state, me)
    if not tg:
        return 0.5
    if bih:
        ps = [bih_prob(state, b, kitchen, cfg) for b in tg] + [cfg.bih_floor]
    else:
        cxy = cue_xy(state)
        ps = [max(max(shot_prob(state, cxy, b, pk, cfg),
                      bank_prob(state, cxy, b, pk, cfg)) for pk in POCKETS)
              for b in tg]
    p1 = max(ps)
    if cfg.leave_net and not bih:
        import leave_net
        p1 = leave_net.predict(cfg.leave_net, state, me, opp, p1)
    n, m = n_left(state, me), n_left(state, opp)
    q_me = cfg.q_lo + (cfg.q_hi - cfg.q_lo) * openness(state, tg)
    q_op = cfg.q_lo + (cfg.q_hi - cfg.q_lo) * openness(state, targets(state, opp))
    shoot = p1 * _race(n - 1, m, q_me, q_op) + (1 - p1) * (
        1 - _race(m, n, q_op, q_me))
    if not cfg.safeties:
        return shoot  # house rules: no safety to fall back on
    safe = 1 - (cfg.q_safe * _race(m - 1, n, q_op, q_me)
                + (1 - cfg.q_safe) * (1 - _race(n, m, q_me, q_op)))
    return max(shoot, safe)


def outcome_value(state: dict, ctx: dict, call: dict, res,
                  cfg: Config = DEFAULT) -> tuple[float, dict]:
    """Shooter-perspective value of one simulated outcome."""
    j = adjudicate(state, ctx, call, res)
    r = j["result"]
    if r == "win":
        return 1.0, j
    if r == "loss":
        return 0.0, j
    if r == "continue":
        return value(j["state"], j["me"], j["opp"], cfg=cfg), j
    if r == "turnover":
        return 1 - value(j["state"], j["opp"], j["me"], cfg=cfg), j
    kitchen = call["kind"] == "break"
    return 1 - value(j["state"], j["opp"], j["me"], bih=True,
                     kitchen=kitchen, cfg=cfg), j


# ------------------------------------------------------------- worker tasks

def _pots_called(res, ball, pocket, first) -> bool:
    return (res is not None and res["pm"].get(ball) == pocket
            and "cue" not in res["pm"] and res["fc"] == first)


def pot_check(state: dict, stroke, ball: str, pocket: str, first: str) -> bool:
    """`_pots_called(summarize(simulate(state, stroke)), ...)`, but the sim
    stops as soon as the answer is a sure no: the cue hits the wrong ball
    first, the cue drops, or the called ball drops in another pocket. A run
    that isn't stopped is the full simulation, so the answer is identical."""
    phi, v, a, b = stroke
    seen_contact = []

    def stop(e) -> bool:
        et = e.event_type
        if et == pt.EventType.BALL_BALL and not seen_contact:
            seen_contact.append(True)
            return next(x.id for x in e.agents if x.id != "cue") != first
        if et == pt.EventType.BALL_POCKET:
            bid, pid = e.agents[0].id, e.agents[1].id
            return bid == "cue" or (bid == ball and pid != pocket)
        return False

    try:
        sys_ = build_system(state)
        sys_.cue.set_state(V0=v, phi=phi, theta=0.0, a=a, b=b)
        if fastsim.simulate_until(sys_, stop):
            return False
    except Exception:  # noqa: BLE001 — degenerate layouts assert in pooltool
        return False
    return _pots_called(summarize(sys_, state), ball, pocket, first)


def _rail_at(xy) -> str:
    x, y = xy
    d = {"L": x, "R": TX - x, "B": y, "T": TY - y}
    return min(d, key=d.get)


def classify_route(shot: pt.System, ball: str, pocket: str) -> dict:
    """What the cue and the called ball actually do before the ball drops:
    cue cushions before first contact (kick), first ball hit (combo
    carrier), other balls the called ball touches (kisses), and cushions it
    banks off (pocket-jaw contacts excluded). The label shown to viewers
    comes from this — never from the search lane that found the aim."""
    def pos_at(bid, t):
        sts = shot.balls[bid].history.states
        st = min(sts, key=lambda x: abs(x.t - t))
        return (float(st.rvw[0][0]), float(st.rvw[0][1]))

    first = None
    cue_rails, kisses, rails = [], [], []
    for e in shot.events:
        ids = [a.id for a in e.agents]
        et = e.event_type
        if et == pt.EventType.BALL_POCKET and ball in ids:
            break
        cushion = et in (pt.EventType.BALL_LINEAR_CUSHION,
                         pt.EventType.BALL_CIRCULAR_CUSHION)
        if et == pt.EventType.BALL_BALL:
            if first is None and "cue" in ids:
                first = next(i for i in ids if i != "cue")
            elif ball in ids:
                other = next(i for i in ids if i != ball)
                if other not in ("cue", first) and other not in kisses:
                    kisses.append(other)
        elif cushion and first is None and "cue" in ids:
            cue_rails.append(_rail_at(pos_at("cue", e.time)))
        elif cushion and ball in ids:
            xy = pos_at(ball, e.time)
            if math.dist(xy, POCKETS[pocket]) > 0.15:  # not the jaws
                r = _rail_at(xy)
                if not rails or rails[-1] != r:
                    rails.append(r)
    route = ("kick" if cue_rails else "combo" if first not in (None, ball)
             else "bank" if rails else "direct")
    return {"route": route, "first": first, "cue_rails": cue_rails,
            "kisses": kisses, "rails": rails}


def route_factor(info: dict, cfg: "Config") -> float:
    return ((1 - cfg.risk_kiss) ** len(info["kisses"])
            * (1 - cfg.risk_rail) ** len(info["rails"])
            * (1 - cfg.risk_kick) ** len(info["cue_rails"])
            * (1 - cfg.risk_combo) ** (info["route"] == "combo"))


def _object_rails(shot: pt.System, ball: str) -> int:
    """Cushion hits by `ball` before it drops — a "direct" lane that touches
    a rail is honestly a bank."""
    n = 0
    for e in shot.events:
        ids = [a.id for a in e.agents]
        if ball not in ids:
            continue
        if e.event_type == pt.EventType.BALL_POCKET:
            break
        if e.event_type in (pt.EventType.BALL_LINEAR_CUSHION,
                            pt.EventType.BALL_CIRCULAR_CUSHION):
            n += 1
    return n


def _departure(shot: pt.System, ball: str):
    """(first ball the cue hits, angle in deg the called ball leaves at)."""
    for e in shot.events:
        if e.event_type == pt.EventType.BALL_BALL:
            ids = [a.id for a in e.agents]
            if "cue" not in ids:
                continue
            other = next(i for i in ids if i != "cue")
            if other != ball:
                return other, None
            ag = next(a for a in e.agents if a.id == ball)
            vx, vy = float(ag.final.state.rvw[1][0]), float(ag.final.state.rvw[1][1])
            return ball, math.degrees(math.atan2(vy, vx))
    return None, None


def _aim_solve(state: dict, ball: str, pk: str, phi: float, v: float,
               a: float = 0.0, b: float = 0.0, iters: int = 7, target=None):
    """Aim solver for a direct pot with THIS stroke: shoot, read the object
    ball's departure angle right after contact, correct the aim (Newton
    step from the cut-geometry derivative, then secant). Each probe stops
    just after contact. Throw, squirt and swerve from the stroke's speed and
    spin are absorbed, because it aims on the measured departure.
    Returns (phi, first_ball_hit)."""
    bb = state["balls"][ball]
    B = (bb["x"], bb["y"])
    hole = target or tuple(float(x) for x in _T.pockets[pk].center[:2])
    want = math.degrees(math.atan2(hole[1] - B[1], hole[0] - B[0]))
    c = cue_xy(state)
    g = ghost_ball_aim(B, hole)
    d_c = math.dist(c, g) or 1e-6
    cos_a = max(0.17, ((g[0] - c[0]) * (hole[0] - B[0]) + (g[1] - c[1]) *
                       (hole[1] - B[1])) / (d_c * (math.dist(B, hole) or 1e-6)))
    slope = -(d_c / (2 * BALL_R * cos_a) - 1)  # d(departure)/d(phi)
    t_hit = d_c / (v * 0.7) + 0.08
    hist, first = [], None
    for _ in range(iters):
        shot = simulate(state, (phi, v, a, b), t_final=t_hit)
        if shot is None:
            return None, None
        first, dep = _departure(shot, ball)
        if first != ball or dep is None:
            return None, first  # blocked / missed the ball
        err = (dep - want + 180) % 360 - 180
        hist.append((phi, err))
        if abs(err) < 0.03:
            break
        if len(hist) >= 2 and hist[-1][1] != hist[-2][1]:
            (p0, e0), (p1, e1) = hist[-2], hist[-1]
            slope = (e1 - e0) / (p1 - p0) if p1 != p0 else slope
        phi = phi - err / (slope if abs(slope) > 1e-3 else -1.0)
    return phi, first


SCAN = (0.0, 0.15, -0.15, 0.3, -0.3, 0.45, -0.45)


def mouth_targets(pk: str) -> list:
    """Aim points across the pocket opening: hole centre, mouth centre, and
    both sides of the mouth. Steep entries (side pockets, along-the-rail
    corners) only drop aimed at the far side of the mouth."""
    hole = tuple(float(x) for x in _T.pockets[pk].center[:2])
    M = POCKETS[pk]
    nx, ny = _NORMAL[pk]
    tx, ty = -ny, nx                      # along the mouth
    half = 0.05 if pk in ("lc", "rc") else 0.04
    return [hole, M, (M[0] + tx * half, M[1] + ty * half),
            (M[0] - tx * half, M[1] - ty * half)]


def _solve_direct(state: dict, lane: dict):
    """Seed a direct lane. Lenient pre-check (reject only if a ball sits
    squarely on the object ball's path), then aim-solve toward each target
    across the pocket mouth and scan a little around each solution until a
    stroke really pots on the full simulation. When the ghost line clips a
    ball, thinner or fuller hits get past it."""
    ball, pk = lane["ball"], lane["pocket"]
    bb = state["balls"][ball]
    B = (bb["x"], bb["y"])
    hole = tuple(float(x) for x in _T.pockets[pk].center[:2])
    if not any(_seg_clear(B, t, state, {ball, "cue"}, corridor=1.2 * BALL_R)
               for t in mouth_targets(pk)):
        return None  # every line into the mouth is squarely blocked
    c = cue_xy(state)
    lite = lane.get("lite", False)
    for v in ((2.2,) if lite else (2.2, 1.4)):
        tried = []
        for tgt in mouth_targets(pk):
            g = ghost_ball_aim(B, tgt)
            phi, first = _aim_solve(state, ball, pk, phi_towards(c, g), v,
                                    target=tgt)
            if phi is None:
                if first not in (None, ball):
                    phi = phi_towards(c, g)  # clipped: still scan around it
                else:
                    continue
            if any(abs(phi - t) < 0.1 for t in tried):
                continue
            tried.append(phi)
            for d in SCAN:
                if pot_check(state, (phi + d, v, 0.0, 0.0), ball, pk, ball):
                    return _seed(lane, c, phi + d, v, math.dist(c, g))
    # fallback for open lanes: a wide scan around the ghost line (catches
    # windows the solver's targets miss — throw, jaws, near-miss blockers)
    g = ghost_ball_aim(B, hole)
    base = phi_towards(c, g)
    step, n = (0.2, 8) if lite else (0.2, 13)
    for v in (2.2,):
        for k in range(1, n):
            for d in (step * k, -step * k):
                if pot_check(state, (base + d, v, 0.0, 0.0), ball, pk, ball):
                    return _seed(lane, c, base + d, v, math.dist(c, g))
    return None


def _seed(lane: dict, c, phi: float, v: float, d_c: float) -> dict:
    return {**lane, "phi": phi, "seed_v": v,
            "aim": (c[0] + math.cos(math.radians(phi)) * d_c,
                    c[1] + math.sin(math.radians(phi)) * d_c)}


def _task_seed(job):
    """First sim-verified (phi, v) for a lane. Direct lanes use the aim
    solver; banks/kicks/combos search their aims × speeds."""
    state, lane = job
    if lane["route"] == "direct":
        return _solve_direct(state, lane)
    c = cue_xy(state)
    if lane["route"] == "bank" and lane.get("target"):
        # aim-solve the object ball toward the mirrored pocket, then scan
        # finely: bank windows can be 0.3deg wide and pace-dependent
        ball, pk, M = lane["ball"], lane["pocket"], lane["target"]
        bb = state["balls"][ball]
        g = ghost_ball_aim((bb["x"], bb["y"]), M)
        phi0 = phi_towards(c, g)
        for v in lane["speeds"]:
            phi, _ = _aim_solve(state, ball, pk, phi0, v, target=M, iters=5)
            base = phi if phi is not None else phi0
            for d in (BANK_SCAN_LITE if lane.get("lite") else BANK_SCAN):
                if pot_check(state, (base + d, v, 0.0, 0.0), ball, pk, ball):
                    return _seed(lane, c, base + d, v, math.dist(c, g))
        return None
    for aim in lane["aims"]:
        phi = phi_towards(c, aim)
        for v in lane["speeds"]:
            if pot_check(state, (phi, v, 0.0, 0.0), lane["ball"],
                         lane["pocket"], lane["first"]):
                return {**lane, "phi": phi, "aim": aim, "seed_v": v}
    return None


def _task_exec(job):
    """One stroke of the position sweep: re-solve the aim for this pace and
    spin (direct lanes), then simulate, adjudicate and value the leave."""
    state, ctx, call, (v, a, b), cfg = job
    phi = call["stroke"][0]
    if call["route"] == "direct" and (a or abs(v - 2.2) > 0.6 or b):
        solved, _ = _aim_solve(state, call["ball"], call["pocket"], phi, v, a, b,
                               iters=4)
        if solved is not None:
            phi = solved
    call = {**call, "stroke": (phi, v, a, b)}
    return _task_eval((state, ctx, call, call["stroke"], cfg))


def _task_eval(job):
    """Simulate + adjudicate + value one stroke. Returns a compact record."""
    state, ctx, call, stroke, cfg = job
    res = summarize(simulate(state, stroke), state)
    val, j = outcome_value(state, ctx, call, res, cfg)
    made = (call["kind"] == "pot"
            and _pots_called(res, call["ball"], call["pocket"], call["first"])
            and j["result"] in ("continue", "win"))
    return {"stroke": stroke, "value": val, "made": made,
            "legal": j["legal"], "result": j["result"],
            "fc": None if res is None else res["fc"]}


def _task_basin(job):
    """Aim basin (deg offsets lo, hi around phi) where the called pot holds."""
    state, cand = job
    ball, pk, first = cand["ball"], cand["pocket"], cand["first"]
    phi, v, a, b = cand["stroke"]

    def ok(d):
        return pot_check(state, (phi + d, v, a, b), ball, pk, first)

    def edge(good, bad):
        while abs(good - bad) > 0.012:
            mid = (good + bad) / 2
            if ok(mid):
                good = mid
            else:
                bad = mid
        return good

    if not ok(0.0):
        return None
    lo = -1.5 if ok(-1.5) else edge(0.0, -1.5)
    hi = 1.5 if ok(1.5) else edge(0.0, 1.5)
    centre = (lo + hi) / 2
    shot = simulate(state, (phi + centre, v, a, b))
    if not _pots_called(summarize(shot, state), ball, pk, first):
        shot = simulate(state, (phi, v, a, b))
    return (lo, hi, classify_route(shot, ball, pk))


_POOL = None


def _workers() -> int:
    # POOL_WORKERS caps the search pool (containers see the host's cores but
    # get a fraction of one); 1 = search in-process, no pool
    return (int(os.environ.get("POOL_WORKERS", "0"))
            or max(1, (os.cpu_count() or 2) - 1))


def _pool():
    global _POOL
    if _POOL is None:
        import atexit
        _POOL = mp.get_context("spawn").Pool(processes=_workers())
        atexit.register(_POOL.terminate)  # no teardown traceback at exit
    return _POOL


def pmap(fn, jobs: list) -> list:
    if not jobs:
        return []
    n = _workers()
    if n == 1:
        return [fn(j) for j in jobs]
    return _pool().map(fn, jobs, chunksize=max(1, len(jobs) // (n * 4)))


# ------------------------------------------------------------ lane generation

RAILS = ("L", "R", "B", "T")


def _mirror(p, rail):
    x, y = p
    return {"L": (-x, y), "R": (2 * TX - x, y),
            "B": (x, -y), "T": (x, 2 * TY - y)}[rail]


def _offsets(base, toward, offs):
    """Aim points shifted perpendicular to base->toward by offs × R."""
    dx, dy = toward[0] - base[0], toward[1] - base[1]
    L = math.hypot(dx, dy) or 1.0
    px, py = -dy / L, dx / L
    return [(base[0] + px * BALL_R * o, base[1] + py * BALL_R * o)
            for o in offs]


# direct-lane seeding: aim-angle offsets (deg) around the ghost line, nearest
# first, out to ±3deg (throw can push a cut's basin 2+deg off the ghost line).
# 0.2deg spacing so no basin wider than ~0.2deg falls between samples — the
# audit's brute force caught 0.3-0.6deg lanes a coarser grid skipped.
SEED_PHIS = (0.0,) + tuple(x for k in range(1, 11) for x in (0.3 * k, -0.3 * k))


def direct_lanes(state: dict, tg: set[str]) -> list[dict]:
    """Every legal ball × pocket whose cut is geometrically possible, seeded
    on a dense aim-angle grid plus wide contact offsets (traffic)."""
    lanes = []
    c = cue_xy(state)
    for ball in sorted(tg):
        b = state["balls"][ball]
        B = (b["x"], b["y"])
        for pk, P in POCKETS.items():
            # no "impossible cut" prefilter: a ball near a rail can be driven
            # into the cushion and rebound to a pocket "behind" it (the
            # audit's brute force found these; the seed relabels them banks)
            g = ghost_ball_aim(B, P)
            gx, gy = g[0] - c[0], g[1] - c[1]
            d = math.hypot(gx, gy) or 1e-6
            aims = []
            for dphi in SEED_PHIS:
                t = math.radians(dphi)
                aims.append((c[0] + d * (gx / d * math.cos(t) - gy / d * math.sin(t)),
                             c[1] + d * (gy / d * math.cos(t) + gx / d * math.sin(t))))
            aims += _offsets(g, P, (0.9, -0.9))
            lanes.append({"ball": ball, "pocket": pk, "first": ball,
                          "route": "direct", "aims": aims,
                          "speeds": (2.0, 3.2)})
    return lanes


BANK_OFFS = (0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0)
BANK_SCAN = (0.0,) + tuple(x for k in range(1, 8) for x in (0.15 * k, -0.15 * k))
BANK_SCAN_LITE = (0.0,) + tuple(x for k in range(1, 4) for x in (0.3 * k, -0.3 * k))
KICK_OFFS = (0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9)


def _rail_point(A, M, rail):
    """Where the segment A→M (M mirrored across `rail`) meets that rail, or
    None if it doesn't (or meets it inside a pocket mouth)."""
    ux, uy = M[0] - A[0], M[1] - A[1]
    if rail in ("L", "R"):
        xr = 0.0 if rail == "L" else TX
        if abs(ux) < 1e-9:
            return None
        t = (xr - A[0]) / ux
    else:
        yr = 0.0 if rail == "B" else TY
        if abs(uy) < 1e-9:
            return None
        t = (yr - A[1]) / uy
    if not 0 < t < 1:
        return None
    K = (A[0] + ux * t, A[1] + uy * t)
    along = K[1] if rail in ("L", "R") else K[0]
    span = TY if rail in ("L", "R") else TX
    if along < 0.1 or along > span - 0.1:
        return None  # corner pocket mouth
    if rail in ("L", "R") and abs(K[1] - TY / 2) < 0.09:
        return None  # side pocket mouth
    return K


def deep_lanes(state: dict, tg: set[str]) -> list[dict]:
    """One-rail banks, one-rail cue kicks, and two-ball combos — only lanes
    whose legs are geometrically open (lenient corridor), so a crowded
    table doesn't spend thousands of sims proving blocked lanes blocked."""
    lanes = []
    c = cue_xy(state)
    for ball in sorted(tg):
        b = state["balls"][ball]
        B = (b["x"], b["y"])
        for pk, P in POCKETS.items():
            for rail in RAILS:
                # bank: B → K (rail) → pocket, cue reaches the ghost
                mp_ = _mirror(P, rail)
                K = _rail_point(B, mp_, rail)
                lenient = 1.2 * BALL_R
                if K and _seg_clear(B, K, state, {ball, "cue"}, lenient) \
                        and _seg_clear(K, P, state, {ball, "cue"}, lenient):
                    g = ghost_ball_aim(B, mp_)
                    if _seg_clear(c, g, state, {ball, "cue"}, lenient):
                        # banks have narrow, pace-dependent windows: dense
                        # aim offsets at soft / medium / firm pace
                        lanes.append({"ball": ball, "pocket": pk,
                                      "first": ball, "route": "bank",
                                      "rail": rail, "target": mp_,
                                      "aims": _offsets(g, mp_, BANK_OFFS),
                                      "speeds": (1.8, 2.6, 3.4)})
                # kick: cue → K (rail) → ghost, ball → pocket open
                G = ghost_ball_aim(B, P)
                g2 = _mirror(G, rail)
                K2 = _rail_point(c, g2, rail)
                if K2 and pocket_lane_open(state, ball, pk) \
                        and _seg_clear(c, K2, state, {"cue"}) \
                        and _seg_clear(K2, G, state, {ball, "cue"}):
                    lanes.append({"ball": ball, "pocket": pk, "first": ball,
                                  "route": "kick", "rail": rail,
                                  "aims": _offsets(g2, c, KICK_OFFS),
                                  "speeds": (1.8, 2.6, 3.4)})
            for via in sorted(tg):
                if via == ball:
                    continue
                v_ = state["balls"][via]
                V = (v_["x"], v_["y"])
                # a combo needs the carrier near the target and roughly
                # behind it on the pocket line (≤60° off), with open lines
                vx, vy = B[0] - V[0], B[1] - V[1]
                dv = math.hypot(vx, vy)
                px_, py_ = P[0] - B[0], P[1] - B[1]
                if dv > 0.6 or (vx * px_ + vy * py_) < 0.5 * dv * math.hypot(px_, py_):
                    continue
                g_o = ghost_ball_aim(B, P)
                gv = ghost_ball_aim(V, g_o)
                if not (pocket_lane_open(state, ball, pk)
                        and _seg_clear(V, g_o, state, {ball, via, "cue"})
                        and _seg_clear(c, gv, state, {via, "cue"})):
                    continue
                lanes.append({"ball": ball, "pocket": pk, "first": via,
                              "route": "combo", "via": via,
                              "aims": _offsets(gv, g_o, (0, 0.25, -0.25)),
                              "speeds": (2.0, 2.9)})
    return lanes


# ------------------------------------------------------------------ search

def _pot_call(seed: dict, stroke) -> dict:
    return {"kind": "pot", "ball": seed["ball"], "pocket": seed["pocket"],
            "first": seed["first"], "route": seed["route"],
            "rail": seed.get("rail"), "via": seed.get("via"),
            "aim": seed["aim"], "stroke": stroke}


def _label(c: dict) -> str:
    """Viewer label from the classified path when known (lane family only
    for un-scored seeds, e.g. the audit's lanes_found list)."""
    info = c.get("path")
    if not info:
        r = c.get("route") or "direct"
        tag = {"direct": "", "bank": f" bank {c.get('rail')}",
               "kick": f" kick {c.get('rail')}",
               "combo": f" combo via {c.get('via')}"}[r]
        return f"{c['ball']}→{c['pocket']}{tag}"
    parts = []
    if info["cue_rails"]:
        parts.append(f"{len(info['cue_rails'])}-rail kick")
    if info["route"] == "combo":
        parts.append(f"combo via {info['first']}")
    if info["rails"]:
        parts.append(f"{len(info['rails'])}-rail bank")
    if info["kisses"]:
        parts.append("kiss off " + ",".join(info["kisses"]))
    return f"{c['ball']}→{c['pocket']}" + (f" ({', '.join(parts)})" if parts else "")


def _lane_prior(state: dict, lane: dict, cfg: Config) -> float:
    """Geometric make-odds of a lane, for budgeted searches (max_lanes)."""
    c = cue_xy(state)
    if lane["route"] == "direct":
        return shot_prob(state, c, lane["ball"], lane["pocket"], cfg)
    if lane["route"] == "bank":
        return bank_prob(state, c, lane["ball"], lane["pocket"], cfg)
    return 0.1  # kicks / combos: below any open pot, ties keep lane order


def _lane_key(c: dict) -> tuple:
    return (c["kind"], c.get("ball"), c.get("pocket"), c.get("route"),
            c.get("rail"), c.get("via"))


def _pot_candidates(state: dict, ctx: dict, lanes: list[dict],
                    rng: random.Random, cfg: Config) -> tuple[list[dict], int, list]:
    """Lanes -> verified seeds -> stroke sweep -> basins -> sampled EV.
    Also returns every lane label that sim-verified (for audits)."""
    if cfg.max_lanes and len(lanes) > cfg.max_lanes:
        lanes = sorted(lanes, key=lambda ln: -_lane_prior(state, ln, cfg)
                       )[:cfg.max_lanes]
    if cfg.exec_grid == "lite":
        # budgeted seeding: one working pace per lane and coarser scans
        lanes = [{**ln, "lite": True,
                  "speeds": ln["speeds"][len(ln["speeds"]) // 2:][:1]}
                 for ln in lanes]
    seeds = [s for s in pmap(_task_seed, [(state, ln) for ln in lanes]) if s]
    probed = sum(len(ln["aims"]) * len(ln["speeds"]) for ln in lanes)
    found = sorted({_label(s) for s in seeds})
    if not seeds:
        return [], probed, found
    # stroke sweep: every (speed, spin) at the verified aim, scored by the
    # value of the position it actually leaves
    jobs, owners = [], []
    for s in seeds:
        if not cfg.position:
            grid = [(s["seed_v"], 0.0, 0.0)]
        elif s["route"] == "direct":
            grid = EXEC_LITE if cfg.exec_grid == "lite" else EXEC_GRID
        else:
            grid = EXEC_DEEP
        for v, a, b in grid:
            call = _pot_call(s, (s["phi"], v, a, b))
            jobs.append((state, ctx, call, (v, a, b), cfg))
            owners.append(s)
    res = pmap(_task_exec, jobs)
    per_lane: dict[tuple, list] = {}
    for s, r in zip(owners, res):
        if r["made"]:
            key = _lane_key({"kind": "pot", **s})
            per_lane.setdefault(key, []).append((r["value"], s, r["stroke"]))
    # geometric prior decides which strokes get a measured basin
    shortlist = []
    for rows in per_lane.values():
        # equal leaves (e.g. on the 8, where any make wins) prefer the
        # softer, cleaner stroke — it is the more makeable one
        rows.sort(key=lambda x: (-round(x[0], 2), x[2][1],
                                 abs(x[2][2]) + abs(x[2][3])))
        s = rows[0][1]
        prior = (shot_prob(state, cue_xy(state), s["ball"], s["pocket"], cfg)
                 if s["route"] == "direct" else 0.35)
        picked, paces = [], set()
        for v_cont, s, stroke in rows:  # best leaves, at distinct paces
            if stroke[1] in paces:
                continue
            paces.add(stroke[1])
            picked.append((v_cont, s, stroke))
            if len(picked) == 3:
                break
        for v_cont, s, stroke in picked:
            shortlist.append((max(prior, 0.05) * v_cont, s, stroke, v_cont))
    shortlist.sort(key=lambda x: -x[0])
    shortlist = shortlist[:cfg.shortlist]
    basins = pmap(_task_basin, [
        (state, {"ball": s["ball"], "pocket": s["pocket"],
                 "first": s["first"], "stroke": st})
        for _, s, st, _ in shortlist])
    v_miss_guess = 1 - value(state, ctx["opp"], ctx["me"], cfg=cfg)
    cands = []
    for (_, s, stroke, v_cont), bas in zip(shortlist, basins):
        if bas is None:
            continue
        lo, hi, info = bas
        half = (hi - lo) / 2
        center = stroke[0] + (lo + hi) / 2
        p_aim = math.erf(half / (sigma_phi(stroke, cfg) * math.sqrt(2)))
        call = _pot_call(s, (center, stroke[1], stroke[2], stroke[3]))
        rf = route_factor(info, cfg)
        call.update(margin=hi - lo, p_aim=p_aim, v_cont=v_cont,
                    route=info["route"], path=info, route_factor=rf,
                    rail="→".join(info["cue_rails"] if info["route"] == "kick"
                                  else info["rails"]) or None,
                    est=p_aim * rf * v_cont + (1 - p_aim * rf) * v_miss_guess)
        cands.append(call)
    cands.sort(key=lambda c: -c["est"])
    cands = cands[:cfg.finalists]
    # sampled EV under full execution noise; forced edge misses guarantee the
    # miss branch is valued from real miss states even for near-certain pots
    jobs, owners = [], []
    for i, c in enumerate(cands):
        phi = c["stroke"][0]
        half = c["margin"] / 2
        for _ in range(cfg.n_pot_mc):
            jobs.append((state, ctx, c, noisy(c["stroke"], rng, cfg), cfg))
            owners.append((i, "mc"))
        for sgn in (-1, 1):
            st = (phi + sgn * (half + sigma_phi(c["stroke"], cfg)),
                  *c["stroke"][1:])
            jobs.append((state, ctx, c, st, cfg))
            owners.append((i, "miss"))
    res = pmap(_task_eval, jobs)
    for i, c in enumerate(cands):
        rows = [(kind, r) for (j, kind), r in zip(owners, res) if j == i]
        c["_mc"] = [r for k, r in rows if k == "mc"]
        c["_forced"] = [r for k, r in rows if k == "miss" and not r["made"]]
        c["_v_miss_guess"] = v_miss_guess
        _score(c)
    return cands, probed, found


def _safety_candidates(state: dict, ctx: dict, tg: set[str],
                       rng: random.Random, cfg: Config) -> list[dict]:
    """Contact-fraction × speed × spin sweep on every legal ball (plus a
    360° aim sweep when direct contacts are scarce), scored by the leave
    the opponent actually gets, then sampled under execution noise."""
    c = cue_xy(state)
    lite = cfg.exec_grid == "lite"
    fracs, speeds, spins = (((-0.5, 0.0, 0.5), (1.6, 2.6), (0.3,)) if lite
                            else (SAFE_FRACS, SAFE_SPEEDS, SAFE_SPINS))
    jobs = []
    for ball in sorted(tg):
        b = state["balls"][ball]
        B = (b["x"], b["y"])
        if not _seg_clear(c, B, state, {ball, "cue"}):
            continue  # no direct line: the 360° kick sweep covers it
        d = math.dist(c, B) or 1.0
        px, py = -(B[1] - c[1]) / d, (B[0] - c[0]) / d
        for f in fracs:
            aim = (B[0] + px * f * 2 * BALL_R, B[1] + py * f * 2 * BALL_R)
            phi = phi_towards(c, aim)
            for v in speeds:
                for sb in spins:
                    jobs.append((state, ctx, {"kind": "safe", "ball": ball,
                                              "route": "direct"},
                                 (phi, v, 0.0, sb), cfg))
    res = pmap(_task_eval, jobs)
    legal = [(j[2], r) for j, r in zip(jobs, res) if r["legal"]]
    if len(legal) < 6:
        # snookered or nearly: full-circle aim sweep finds kicks/escapes
        sweep = []
        k = 0
        step = 3 * SWEEP_STEP if lite else SWEEP_STEP
        while k * step < 360:
            for v in ((2.4,) if lite else SWEEP_SPEEDS):
                sweep.append((state, ctx, {"kind": "safe", "route": "kick"},
                              (k * step, v, 0.0, 0.0), cfg))
            k += 1
        res2 = pmap(_task_eval, sweep)
        legal += [(j[2], r) for j, r in zip(sweep, res2) if r["legal"]]
    if not legal:
        return []
    legal.sort(key=lambda x: -x[1]["value"])
    picked, seen = [], set()
    for call, r in legal:
        key = (r["fc"], round(r["stroke"][0] / 1.5), round(r["stroke"][1]))
        if key in seen:
            continue
        seen.add(key)
        picked.append({**call, "ball": r["fc"], "stroke": r["stroke"],
                       "nominal": r["value"]})
        if len(picked) >= (4 if lite else 8):
            break
    jobs, owners = [], []
    for i, s in enumerate(picked):
        for _ in range(cfg.n_safe_mc):
            jobs.append((state, ctx, s, noisy(s["stroke"], rng, cfg), cfg))
            owners.append(i)
    res = pmap(_task_eval, jobs)
    for i, s in enumerate(picked):
        s["_mc"] = [r for j, r in zip(owners, res) if j == i]
        _score(s)
        s["aim"] = (c[0] + math.cos(math.radians(s["stroke"][0])) * 0.5,
                    c[1] + math.sin(math.radians(s["stroke"][0])) * 0.5)
    return picked


def _score(c: dict) -> None:
    """(Re)compute a candidate's EV from its accumulated noisy samples.

    Safety: mean sampled value. Pot, first pass (few samples): p_make = aim
    basin probability × the share of in-basin samples that still make it —
    low variance, but it treats spin/speed error as independent of aim, so
    it reads a few points high. Once refined (≥ REFINE_N samples) p_make is
    the plain sampled make-rate, which is unbiased. The miss branch is
    valued from sampled misses plus two forced edge misses.
    """
    mc = c["_mc"]
    c["n_samples"] = len(mc)
    c["fouls"] = sum(r["result"] == "foul" for r in mc)
    if c["kind"] != "pot":
        c["ev"] = sum(r["value"] for r in mc) / len(mc)
        return
    made = [r["value"] for r in mc if r["made"]]
    miss_pool = ([r["value"] for r in mc if not r["made"]]
                 + [r["value"] for r in c["_forced"]])
    if len(mc) >= REFINE_N:
        p = len(made) / len(mc)
    else:
        inside = [r for r in mc
                  if abs(r["stroke"][0] - c["stroke"][0]) <= c["margin"] / 2]
        keep = (sum(r["made"] for r in inside) / len(inside)) if inside else 1.0
        p = c["p_aim"] * keep
    p *= c.get("route_factor", 1.0)
    v_make = sum(made) / len(made) if made else c["v_cont"]
    v_miss = (sum(miss_pool) / len(miss_pool) if miss_pool
              else c["_v_miss_guess"])
    c.update(p_make=p, v_make=v_make, v_miss=v_miss,
             ev=p * v_make + (1 - p) * v_miss)


REFINE_N = 32     # samples behind any move that is actually chosen
REFINE_STEP = 26


def _refine(state: dict, ctx: dict, cands: list[dict], rng: random.Random,
            cfg: Config, rounds: int = 6) -> None:
    """Sample until the leader is confirmed: while the EV leader, or any
    candidate within `cfg.refine_close` of it, rests on a first-pass
    estimate, give those REFINE_STEP more full-noise executions each and
    re-rank. A candidate that overtakes the leader gets refined in turn."""
    for _ in range(rounds):
        lead = max(cands, key=lambda c: c["ev"])
        todo = [c for c in cands if c["n_samples"] < REFINE_N
                and (c is lead or c["ev"] >= lead["ev"] - cfg.refine_close)]
        if not todo:
            return
        jobs, owners = [], []
        for c in todo:
            call = {k: c.get(k) for k in ("kind", "ball", "pocket", "first")}
            for _ in range(REFINE_STEP):
                jobs.append((state, ctx, call, noisy(c["stroke"], rng, cfg),
                             cfg))
                owners.append(c)
        res = pmap(_task_eval, jobs)
        for c in todo:
            c["_mc"] = c["_mc"] + [r for o, r in zip(owners, res) if o is c]
            _score(c)


def choose(state: dict, ctx: dict, cfg: Config = DEFAULT,
           seed: int = 0, keep: int = 0) -> dict:
    """Best call for the player to move: argmax sampled EV over every pot
    and safety the search finds. Deep routes (banks/kicks/combos) join
    whenever no direct pot reaches `cfg.deep_below`.

    Returns the chosen call (stroke = (phi, V, a, b) to execute) with the
    full ranked `contest` — one row per distinct option — for audit/evals.
    keep > 0 also returns the top `keep` distinct options in full as
    `candidates`, each with its noisy samples (`_mc`), for a deeper search.
    """
    rng = random.Random(seed)
    tg = targets(state, ctx["me"])
    pots, probed, found = _pot_candidates(state, ctx, direct_lanes(state, tg),
                                          rng, cfg)
    deep_ran = not pots or max(p["p_make"] for p in pots) < cfg.deep_below
    if deep_ran:
        deep, dp, dfound = _pot_candidates(state, ctx, deep_lanes(state, tg),
                                           rng, cfg)
        pots += deep
        probed += dp
        found += dfound
    safes = (_safety_candidates(state, ctx, tg, rng, cfg)
             if cfg.safeties or not pots else [])
    if pots or safes:
        _refine(state, ctx, pots + safes, rng, cfg)
    best_ev = max((c["ev"] for c in pots + safes), default=0.0)
    if cfg.tie_eps > 0:  # near-ties go to the more makeable shot
        ranked = sorted(pots + safes, key=lambda c: (
            -c["ev"] if best_ev - c["ev"] > cfg.tie_eps else -best_ev,
            -c.get("p_make", 0.0), -c["ev"]))
    else:
        ranked = sorted(pots + safes, key=lambda c: -c["ev"])
    if not ranked:
        # nothing legal anywhere: the least-bad stroke under the same rules
        # (a foul hands over ball in hand; dropping the 8 loses the game)
        best = _least_bad(state, ctx, tg, cfg)
        return {**best, "no_legal": True, "contest": [],
                "lanes_probed": probed, "lanes_found": found,
                "deep_ran": deep_ran, "n_pots": 0, "n_safes": 0}
    contest, seen, distinct = [], set(), []
    for c in ranked:
        k = _lane_key(c) if c["kind"] == "pot" else ("safe", c.get("ball"),
                                                      c.get("route"))
        if k not in seen:
            seen.add(k)
            contest.append(_brief(c))
            distinct.append(c)
    best = {k: v for k, v in ranked[0].items() if not k.startswith("_")}
    best["plan"] = next_shot_plan(state, ctx, best, cfg)
    best.update(contest=contest[:8], lanes_probed=probed,
                lanes_found=found, deep_ran=deep_ran,
                n_pots=len({_lane_key(p) for p in pots}), n_safes=len(safes))
    if pots:
        best["best_pot"] = _brief(max(pots, key=lambda c: c["p_make"]))
    if keep:
        best["candidates"] = [
            {k: v for k, v in c.items() if k == "_mc" or not k.startswith("_")}
            for c in distinct[:keep]]
    return best


def _least_bad(state: dict, ctx: dict, tg: set[str], cfg: Config) -> dict:
    """With no legal pot or safety found: soft-to-firm hits across every
    target plus a coarse 360° sweep, scored by `_task_eval` (the value of
    what each actually leaves, fouls and losses included)."""
    c = cue_xy(state)
    jobs = []
    for ball in sorted(tg):
        B = (state["balls"][ball]["x"], state["balls"][ball]["y"])
        d = math.dist(c, B) or 1.0
        px, py = -(B[1] - c[1]) / d, (B[0] - c[0]) / d
        for f in (-0.6, 0.0, 0.6):
            phi = phi_towards(c, (B[0] + px * f * 2 * BALL_R,
                                  B[1] + py * f * 2 * BALL_R))
            for v in (0.8, 1.6, 3.0):
                jobs.append((state, ctx, {"kind": "safe", "ball": ball,
                                          "route": "direct"},
                             (phi, v, 0.0, 0.0), cfg))
    jobs += [(state, ctx, {"kind": "safe", "route": "kick"},
              (float(a), 1.2, 0.0, 0.0), cfg) for a in range(0, 360, 10)]
    res = pmap(_task_eval, jobs)
    # the 4 best nominal strokes, re-scored under execution noise
    top = sorted(range(len(jobs)), key=lambda i: -res[i]["value"])[:4]
    rng = random.Random(len(jobs))
    noisy_jobs = [(state, ctx, jobs[i][2], noisy(jobs[i][3], rng, cfg), cfg)
                  for i in top for _ in range(4)]
    vals = [r["value"] for r in pmap(_task_eval, noisy_jobs)]
    ev = {i: sum(vals[n * 4:(n + 1) * 4]) / 4 for n, i in enumerate(top)}
    k = max(top, key=lambda i: ev[i])
    call, stroke = jobs[k][2], jobs[k][3]
    aim = (c[0] + math.cos(math.radians(stroke[0])) * 0.5,
           c[1] + math.sin(math.radians(stroke[0])) * 0.5)
    return {**call, "ball": call.get("ball") or res[k]["fc"], "stroke": stroke,
            "aim": aim, "ev": ev[k]}


def next_shot_plan(state: dict, ctx: dict, call: dict,
                   cfg: Config = DEFAULT) -> dict | None:
    """What the stroke is playing position FOR: simulate the intended
    (noise-free) stroke and report where the cue stops and the best shot
    it leaves — the shooter's own next shot if the turn continues, the
    opponent's best look otherwise."""
    if call["kind"] not in ("pot", "safe"):
        return None
    res = summarize(simulate(state, tuple(call["stroke"])), state)
    if res is None:
        return None
    j = adjudicate(state, ctx, call, res)
    post = j["state"]
    if j["result"] not in ("continue", "turnover") or \
            post["balls"]["cue"].get("pocketed"):
        return {"result": j["result"]}
    mine = j["result"] == "continue"
    grp = j["me"] if mine else j["opp"]
    cxy = cue_xy(post)
    best = (0.0, None, None)
    for b in targets(post, grp):
        for pk in POCKETS:
            p = max(shot_prob(post, cxy, b, pk, cfg),
                    bank_prob(post, cxy, b, pk, cfg))
            if p > best[0]:
                best = (p, b, pk)
    return {"result": j["result"], "for": "self" if mine else "opponent",
            "cue": cxy, "p": round(best[0], 3), "ball": best[1],
            "pocket": best[2]}


def _brief(c: dict) -> dict:
    out = {"kind": c["kind"], "ball": c.get("ball"), "pocket": c.get("pocket"),
           "route": c.get("route"), "ev": round(c["ev"], 3),
           "n": c.get("n_samples")}
    if c["kind"] == "pot":
        out["label"] = _label(c)
        out.update(p_make=round(c["p_make"], 3), margin=round(c["margin"], 2),
                   v_make=round(c["v_make"], 3), v_miss=round(c["v_miss"], 3))
        if c.get("rail"):
            out["rail"] = c["rail"]
        if c.get("via"):
            out["via"] = c["via"]
    else:
        out["fouls"] = c.get("fouls", 0)
    return out


# ------------------------------------------------------------ ball in hand

def placement_ok(state: dict, xy, kitchen: bool = False) -> bool:
    x, y = xy
    if not (BALL_R + 0.003 <= x <= TX - BALL_R - 0.003
            and BALL_R + 0.003 <= y <= TY - BALL_R - 0.003):
        return False
    if kitchen and y >= HEAD_STRING:
        return False
    return all(math.hypot(bl["x"] - x, bl["y"] - y) > 2 * BALL_R + 0.003
               for b, bl in state["balls"].items()
               if b != "cue" and not bl.get("pocketed"))


def choose_bih(state: dict, ctx: dict, cfg: Config = DEFAULT, seed: int = 0,
               kitchen: bool = False) -> dict:
    """Ball in hand: search cue placements (behind each open pot lane at
    several distances/angles for shape), pre-rank by geometric make-odds ×
    the best leave a stroke sweep finds from there, then run the full
    decision from the top placements and keep the best EV. `kitchen` limits
    placement behind the head string (scratch on the break)."""
    tg = targets(state, ctx["me"])
    places = []
    for ball in sorted(tg):
        b = state["balls"][ball]
        B = (b["x"], b["y"])
        for pk, P in POCKETS.items():
            if not pocket_lane_open(state, ball, pk):
                continue
            d = math.dist(B, P) or 1e-6
            ux, uy = (P[0] - B[0]) / d, (P[1] - B[1]) / d
            G = (B[0] - ux * 2 * BALL_R, B[1] - uy * 2 * BALL_R)
            for dist in (0.10, 0.22, 0.40):
                for ang in (0, 20, -20, 40, -40):
                    t = math.radians(ang)
                    dx = -ux * math.cos(t) + uy * math.sin(t)
                    dy = -uy * math.cos(t) - ux * math.sin(t)
                    C = (G[0] + dx * dist, G[1] + dy * dist)
                    if not placement_ok(state, C, kitchen):
                        continue
                    p = shot_prob(with_cue(state, C), C, ball, pk, cfg)
                    if p > 0.3:
                        places.append((p, ball, pk, C))
    spots = []
    if places:
        places.sort(key=lambda x: -x[0])
        per_ball: dict[str, int] = {}
        short = []
        for p, ball, pk, C in places:
            if per_ball.get(ball, 0) >= 4:
                continue
            per_ball[ball] = per_ball.get(ball, 0) + 1
            short.append((p, ball, pk, C))
        short = short[:cfg.max_lanes or 24]
        strokes = ([(v, b) for v, _, b in EXEC_LITE[:9]]
                   if cfg.exec_grid == "lite" else
                   [(v, b) for v in EXEC_SPEEDS
                    for b in (-0.5, -0.25, 0.0, 0.25, 0.5)])
        jobs, owners = [], []
        for i, (p, ball, pk, C) in enumerate(short):
            st = with_cue(state, C)
            bb = st["balls"][ball]
            phi = phi_towards(C, ghost_ball_aim((bb["x"], bb["y"]),
                                                POCKETS[pk]))
            call = {"kind": "pot", "ball": ball, "pocket": pk, "first": ball}
            for v, b in strokes:
                jobs.append((st, ctx, call, (phi, v, 0.0, b), cfg))
                owners.append(i)
        res = pmap(_task_eval, jobs)
        score: dict[int, float] = {}
        for i, r in zip(owners, res):
            if r["made"]:
                score[i] = max(score.get(i, 0.0), r["value"])
        for i in sorted(score, key=lambda i: -short[i][0] * score[i]):
            C = short[i][3]
            if all(math.dist(C, s) > 0.03 for s in spots):
                spots.append(C)
            if len(spots) >= (1 if cfg.max_lanes else 2):
                break
    if not spots:
        # nothing to pot from anywhere: place for the safety search
        grid = [(x, y) for x in (0.2, 0.5, 0.8)
                for y in ((0.15, 0.3, 0.45) if kitchen
                          else (0.3, 0.7, 1.0, 1.3, 1.7))]
        spots = [C for C in grid if placement_ok(state, C, kitchen)][:4]
    best = None
    for k, C in enumerate(spots):
        call = choose(with_cue(state, C), ctx, cfg, seed + k)
        if best is None or call["ev"] > best["ev"]:
            best = {**call, "cue_place": C}
    return best


# ---------------------------------------------------------------- the break

def choose_break(state: dict, cfg: Config = DEFAULT, seed: int = 0,
                 n: int | None = None) -> dict:
    """Pick cue spot (kitchen), speed and spin by sampled break outcomes."""
    n = n or cfg.break_samples
    thin = n < 8  # budgeted: central spots, firm pace, draw only
    rng = random.Random(seed)
    apex = min(live(state), key=lambda b: state["balls"][b]["y"])
    A = (state["balls"][apex]["x"], state["balls"][apex]["y"])
    ctx = {"me": None, "opp": None}
    opts = []
    for x in ((0.40, TX / 2, 0.59) if thin else
              (0.25, 0.40, TX / 2, 0.59, 0.74)):
        C = (x, HEAD_STRING - 0.04)
        phi = phi_towards(C, A)
        for v in ((10.0, 12.0) if thin else (8.0, 10.0, 12.0)):
            for b in ((-0.2,) if thin else (-0.2, 0.1)):
                opts.append({"kind": "break", "cue_place": C, "apex": apex,
                             "stroke": (phi, v, 0.0, b)})
    jobs, owners = [], []
    for i, o in enumerate(opts):
        st = with_cue(state, o["cue_place"])
        for _ in range(n):
            jobs.append((st, ctx, {"kind": "break"},
                         noisy(o["stroke"], rng, cfg), cfg))
            owners.append(i)
    res = pmap(_task_eval, jobs)
    for i, o in enumerate(opts):
        rows = [r for j, r in zip(owners, res) if j == i]
        o["ev"] = sum(r["value"] for r in rows) / len(rows)
        o["p_continue"] = sum(r["result"] == "continue" for r in rows) / len(rows)
        o["p_scratch"] = sum(r["result"] == "foul" for r in rows) / len(rows)
    return max(opts, key=lambda o: o["ev"])
