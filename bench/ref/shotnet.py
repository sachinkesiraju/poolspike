"""ShotNet: a small net that scores candidate shots from simulated features.

A turn becomes up to CANDS (ball, pocket) x STROKES options. Each option is
aimed with a short potting-window probe, simulated once without noise, and
reduced to FEATURES. Net: f -> tanh(8) -> score. The best score is played.
Physics supplies the features; every strategic choice is in the weights.

  shotnet.write(profile)            weights written by hand, no data, no gradients
  ShotNet(weights).act(turn)        wire action
"""
import json
import math
import os
import random
import sys
import zlib

import numpy as np

ROOT = os.environ.get("POOL_COWORLD", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pool8"))
sys.path.insert(0, ROOT)
import baseline  # noqa: E402
import policy as P  # noqa: E402
from pool_env import ghost_ball_aim, phi_towards  # noqa: E402
from rl import encode as E  # noqa: E402

FEATURES = ("window", "odds", "straight", "made", "scratch", "next_odds", "next_open",
            "cue_central", "speed", "spin", "opp_odds", "win", "bias",
            "noisy_make", "foul_risk", "loss_risk", "noisy_next", "opp_after_miss")
F = {k: i for i, k in enumerate(FEATURES)}
N = len(FEATURES)
HIDDEN = 8
CANDS = 3
WIDEN = os.environ.get("SHOTNET_WIDEN", "1") == "1"
DEEP = os.environ.get("SHOTNET_DEEP", "1") == "1"
DEEP_MAX = 8
STROKES = ((1.8, 0.0), (2.8, 0.0), (1.2, 0.0), (1.5, 0.3), (1.5, -0.3), (2.4, 0.3), (2.4, -0.3))  # speed, vertical
NOISY = 4
NOISE = P.DEFAULT
COARSE = 0.4
COARSE_N = 5                                                  # +-2 deg
FINE = 0.1


def _window(st, ball, pk, base, step, n, v, s):
    hits = [P.pot_check(st, (base + k * step, v, 0.0, s), ball, pk, ball)
            for k in range(-n, n + 1)]
    best, cur, end = 0, 0, 0
    for i, h in enumerate(hits):
        cur = cur + 1 if h else 0
        if cur > best:
            best, end = cur, i
    if not best:
        return None, 0.0
    return base + (end - (best - 1) / 2 - n) * step, best * step


def _best_odds(st, group):
    tg = P.targets(st, group)
    if not tg:
        return 0.0, 0.0
    c = P.cue_xy(st)
    ps = [max(P.shot_prob(st, c, b, pk) for pk in P.POCKETS) for b in tg]
    return max(ps), sum(p >= 0.5 for p in ps) / len(ps)


def noisy_stats(f, st, ctx, call, stroke, seed):
    """Fill the noisy features: the same stroke under table noise, NOISY times."""
    rng = random.Random(seed)
    made = foul = loss = 0
    nxt, opp = [], []
    for _ in range(NOISY):
        res = P.summarize(P.simulate(st, P.noisy(stroke, rng, NOISE)), st)
        if res is None:
            foul += 1
            continue
        j = P.adjudicate(st, ctx, call, res)
        r = j["result"]
        made += r in ("continue", "win")
        foul += r == "foul"
        loss += r == "loss"
        if r == "continue":
            nxt.append(_best_odds(j["state"], j["me"])[0])
        elif r in ("turnover", "foul") and j["opp"] is not None and "cue" not in res["pm"]:
            opp.append(_best_odds(j["state"], j["opp"])[0])
    f[F["noisy_make"]] = made / NOISY
    f[F["foul_risk"]] = foul / NOISY
    f[F["loss_risk"]] = loss / NOISY
    f[F["noisy_next"]] = sum(nxt) / NOISY
    f[F["opp_after_miss"]] = sum(opp) / NOISY


def features(st, ctx, ball, pk, q, base, v, s, shot_no=0, hint=None):
    """(feature vector, aim phi) for one option; None if it never pots."""
    if hint is None:
        centre, width = _window(st, ball, pk, base, COARSE, COARSE_N, v, s)
    else:
        centre, width = _window(st, ball, pk, hint, FINE * 3, 3, v, s)
    if centre is None:
        return None
    fine, w2 = _window(st, ball, pk, centre, FINE, 6, v, s)
    if fine is not None:
        centre, width = fine, w2
    res = P.summarize(P.simulate(st, (centre, v, 0.0, s)), st)
    f = np.zeros(N)
    f[F["window"]] = min(width, 2.0) / 2.0
    f[F["odds"]] = q
    B = st["balls"][ball]
    c = P.cue_xy(st)
    hole = P.POCKETS[pk]
    d = math.dist((B["x"], B["y"]), hole) or 1e-6
    g = ghost_ball_aim((B["x"], B["y"]), hole)
    vx, vy = g[0] - c[0], g[1] - c[1]
    f[F["straight"]] = ((vx * (hole[0] - B["x"]) + vy * (hole[1] - B["y"])) / d
                        / (math.hypot(vx, vy) or 1e-6))
    f[F["speed"]] = v / 3.0
    f[F["spin"]] = abs(s)
    f[F["bias"]] = 1.0
    call = {"kind": "pot", "ball": ball, "pocket": pk}
    noisy_stats(f, st, ctx, call, (centre, v, 0.0, s),
                zlib.crc32(f"{shot_no}|{ball}|{pk}|{v}|{s}".encode()))
    if res is None:
        return f, centre
    j = P.adjudicate(st, ctx, call, res)
    f[F["scratch"]] = float("cue" in res["pm"])
    f[F["win"]] = float(j["result"] == "win")
    f[F["made"]] = float(j["result"] in ("continue", "win"))
    post = j["state"]
    if "cue" not in res["pm"]:
        x, y = P.cue_xy(post)
        f[F["cue_central"]] = min(x / P.TX, 1 - x / P.TX, y / P.TY, 1 - y / P.TY) * 2
    if j["result"] == "continue":
        f[F["next_odds"]], f[F["next_open"]] = _best_odds(post, j["me"])
    if "cue" not in res["pm"]:
        f[F["opp_odds"]] = _best_odds(post, j["opp"])[0] if j["opp"] else 0.0
    return f, centre


def options(turn):
    """Every scored option: [(features, wire action)]."""
    st0 = E.to_state(turn)
    me = E.GROUPS[turn.get("my_group")]
    opp = None if me is None else (P.STRIPES if me == P.SOLIDS else P.SOLIDS)
    ctx = {"me": me, "opp": opp}
    phase = turn["phase"]
    legal = turn["legal_targets"]
    cands = []
    if phase == "shoot":
        c = P.cue_xy(st0)
        for b in legal:
            for pk in P.POCKETS:
                q = P.shot_prob(st0, c, b, pk)
                if q > 0.02:
                    cands.append((q, b, pk, None, st0))
    else:
        kitchen = phase == "ball_in_hand_kitchen"
        for b in legal:
            B = st0["balls"][b]
            for pk in P.POCKETS:
                if not P.pocket_lane_open(st0, b, pk):
                    continue
                hole = P.POCKETS[pk]
                d = math.dist((B["x"], B["y"]), hole) or 1e-6
                C0 = (B["x"] - (hole[0] - B["x"]) / d * 0.25, B["y"] - (hole[1] - B["y"]) / d * 0.25)
                C, _ = E.legalize(st0, C0, kitchen)
                st = P.with_cue(st0, C)
                q = P.shot_prob(st, C, b, pk)
                if q > 0.02:
                    cands.append((q, b, pk, C, st))
    cands.sort(key=lambda x: -x[0])
    out = []
    # widen until something really pots: the easiest few first, then the rest
    for i, (q, b, pk, C, st) in enumerate(cands):
        if i >= CANDS and (out or not WIDEN):
            break
        B = st["balls"][b]
        hole = tuple(float(x) for x in P._T.pockets[pk].center[:2])
        base = phi_towards(P.cue_xy(st), ghost_ball_aim((B["x"], B["y"]), hole))
        hint = None
        for si, (v, s) in enumerate(STROKES):
            r = features(st, ctx, b, pk, q, base, v, s, turn.get("shot_no", 0), hint)
            if r is None:
                if si == 0:   # the base stroke finds no window: drop this lane
                    break
                continue
            f, phi = r
            if si == 0:
                hint = phi
            wire = {"call": {"ball": b, "pocket": pk},
                    "stroke": {"phi": float(phi % 360), "speed": v, "side": 0.0, "vertical": s}}
            if C is not None:
                wire["cue"] = {"x": float(C[0]), "y": float(C[1])}
            out.append((f, wire))
    if not out and DEEP and phase == "shoot":
        out = deep_options(st0, ctx, legal)
    return out, st0, me


def deep_options(st, ctx, legal):
    """No direct pot verified: ask the search's own lane finder for thin cuts,
    banks, kicks and combos. Same features; window is measured around the seed."""
    tg = set(legal)
    lanes = P.direct_lanes(st, tg) + P.deep_lanes(st, tg)
    out = []
    for ln in lanes:
        ln["lite"] = True
        seed = P._task_seed((st, ln))
        if not seed:
            continue
        b, pk, first, phi, v = seed["ball"], seed["pocket"], seed["first"], seed["phi"], seed["seed_v"]
        hits = sum(P.pot_check(st, (phi + d, v, 0.0, 0.0), b, pk, first)
                   for d in (-0.4, -0.2, 0.0, 0.2, 0.4))
        f = np.zeros(N)
        f[F["window"]] = min(hits * 0.2, 2.0) / 2.0
        f[F["speed"]] = v / 3.0
        f[F["bias"]] = 1.0
        res = P.summarize(P.simulate(st, (phi, v, 0.0, 0.0)), st)
        if res is not None:
            j = P.adjudicate(st, ctx, {"kind": "pot", "ball": b, "pocket": pk}, res)
            f[F["scratch"]] = float("cue" in res["pm"])
            f[F["win"]] = float(j["result"] == "win")
            f[F["made"]] = float(j["result"] in ("continue", "win"))
            if j["result"] == "continue":
                f[F["next_odds"]], f[F["next_open"]] = _best_odds(j["state"], j["me"])
        noisy_stats(f, st, ctx, {"kind": "pot", "ball": b, "pocket": pk, "first": first},
                    (phi, v, 0.0, 0.0), zlib.crc32(f"deep|{b}|{pk}|{first}|{phi:.4f}".encode()))
        out.append((f, {"call": {"ball": b, "pocket": pk},
                        "stroke": {"phi": float(phi % 360), "speed": float(v), "side": 0.0,
                                   "vertical": 0.0}}))
        if len(out) >= DEEP_MAX:
            break
    return out


def _load_book():
    with open(os.path.join(ROOT, "break_book.json")) as z:
        return json.load(z)["top"][0]


class ShotNet:
    def __init__(self, w):
        if isinstance(w, str):
            with np.load(w) as z:
                w = {k: z[k] for k in z.files}
        self.w = w
        self.book = _load_book()

    def score(self, X):
        h = np.tanh(X @ self.w["W"])
        return (h @ self.w["v"]).ravel()

    def act(self, turn):
        phase = turn["phase"]
        if phase == "break":
            return break_wire(self.book, turn)
        opts, st0, me = options(turn)
        if not opts:
            return baseline.action(st0, phase, me)
        s = self.score(np.array([o[0] for o in opts]))
        return opts[int(np.argmax(s))][1]


def break_wire(top, turn):
    """The book's best recipe. Falls back to the scripted break if its format differs."""
    st0 = E.to_state(turn)
    C = (float(top["x"]), P.HEAD_STRING - 0.04)
    apex = E.apex_ball(st0)
    A = (st0["balls"][apex]["x"], st0["balls"][apex]["y"])
    return {"call": None, "cue": {"x": C[0], "y": C[1]},
            "stroke": {"phi": float(phi_towards(C, A) % 360), "speed": float(top["v"]),
                       "side": float(top["a"]), "vertical": float(top["b"])}}


# Concept units, each one line of strategy. gain keeps tanh close to linear.
GAIN = 0.25
PROFILES = {
    "pot_only": {"potting": {"made": 2.0, "window": 1.5, "odds": 0.5},
                 "foul": {"scratch": -3.0}, "finish": {"win": 5.0}},
    "risk": {"potting": {"made": 2.0, "window": 1.5, "odds": 0.5},
             "foul": {"scratch": -3.0, "foul_risk": -3.0}, "finish": {"win": 5.0, "loss_risk": -5.0}},
    "risk_pos": {"potting": {"made": 2.0, "window": 1.5, "odds": 0.5, "noisy_make": 1.0},
                 "foul": {"scratch": -3.0, "foul_risk": -3.0}, "finish": {"win": 5.0, "loss_risk": -5.0},
                 "position": {"noisy_next": 1.5, "opp_after_miss": -0.5}},
    "full": {"potting": {"made": 2.0, "window": 1.5, "odds": 0.5},
             "foul": {"scratch": -3.0}, "finish": {"win": 5.0},
             "position": {"next_odds": 1.0, "next_open": 0.5, "cue_central": 0.2},
             "touch": {"spin": -0.2}},
}


def write(profile="full"):
    """Weights written from a strategy table. No data, no gradients."""
    W = np.zeros((N, HIDDEN))
    v = np.zeros((HIDDEN, 1))
    for u, (_, terms) in enumerate(PROFILES[profile].items()):
        for k, x in terms.items():
            W[F[k], u] = GAIN * x
        v[u, 0] = 1.0 / GAIN
    return {"W": W, "v": v}


if __name__ == "__main__":
    for name in PROFILES:
        np.savez(sys.argv[1] + f"/shotnet_{name}.npz", **write(name))
    print("features", FEATURES)
