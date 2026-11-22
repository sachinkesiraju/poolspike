"""ShotNet v3: the same small scorer over a wider option set and richer features.

Options: more lanes, EV's lite stroke grid (with side spin), real ball-in-hand
placements, deep lanes when direct pots are poor, the search's fallback when
nothing can be called. Two stages: every option gets NOISY0 noisy executions,
the net shortlists TOP, those get NOISY1 more, the net picks.

New inputs include value_* : the mean of policy.outcome_value (EV's analytic
leave evaluator) over noisy executions. So the net sees what EV sees; only the
weights over these features are learned.
"""
import math
import os
import random
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shotnet as SN  # noqa: E402
from shotnet import E, P, baseline, ghost_ball_aim, phi_towards  # noqa: E402
from rl.env import call_to_wire  # noqa: E402

FEATURES = SN.FEATURES + ("value_mean", "value_make", "value_miss", "value_nominal",
                          "side", "deep", "bih", "make_sq", "value_spread")
F = {k: i for i, k in enumerate(FEATURES)}
N = len(FEATURES)
HIDDEN = 8
LANES = int(os.environ.get("SN3_LANES", 5))
SPOTS = 5
TOP = int(os.environ.get("SN3_TOP", 6))
NOISY0 = int(os.environ.get("SN3_N0", 4))
NOISY1 = int(os.environ.get("SN3_N1", 20))
DEEP_BELOW = 0.6
DEEP_MAX = 8
GRID = ((1.8, 0.0, 0.0),) + tuple((v, 0.0, b) for v in (1.4, 2.4, 3.6) for b in (-0.4, 0.0, 0.4)) + tuple(
    (2.4, a, b) for a in (-0.3, 0.3) for b in (-0.35, 0.35))
BIH_GRID = ((1.8, 0.0, 0.0),) + tuple((v, 0.0, b) for v in (1.4, 2.4) for b in (-0.4, 0.4))
NOISE = P.DEFAULT
VNET = None   # the acting net trained evaluator; None = analytic. Never touches P.value, so opponents are unaffected.


def outcome_value(st, ctx, call, res):
    if VNET is None:
        return P.outcome_value(st, ctx, call, res)
    j = P.adjudicate(st, ctx, call, res)
    r = j["result"]
    if r == "win":
        return 1.0, j
    if r == "loss":
        return 0.0, j
    if r == "continue":
        return VNET(j["state"], j["me"], j["opp"]), j
    if r == "turnover":
        return 1 - VNET(j["state"], j["opp"], j["me"]), j
    return 1 - VNET(j["state"], j["opp"], j["me"], bih=True), j


def window(st, ball, pk, first, base, step, n, v, a, b):
    hits = [P.pot_check(st, (base + k * step, v, a, b), ball, pk, first) for k in range(-n, n + 1)]
    best, cur, end = 0, 0, 0
    for i, h in enumerate(hits):
        cur = cur + 1 if h else 0
        if cur > best:
            best, end = cur, i
    if not best:
        return None, 0.0
    return base + (end - (best - 1) / 2 - n) * step, best * step


class Opt:
    __slots__ = ("st", "ctx", "call", "stroke", "base", "wire", "rng", "n", "made", "foul", "loss",
                 "nxt", "opp", "val", "valm", "valx", "val2", "x")

    def __init__(self, st, ctx, call, stroke, base, wire, seed):
        self.st, self.ctx, self.call, self.stroke, self.base, self.wire = st, ctx, call, stroke, base, wire
        self.rng = random.Random(seed)
        self.n = self.made = self.foul = self.loss = 0
        self.nxt = self.opp = self.val = self.valm = self.valx = self.val2 = 0.0
        self.x = None

    def sample(self, k):
        for _ in range(k):
            res = P.summarize(P.simulate(self.st, P.noisy(self.stroke, self.rng, NOISE)), self.st)
            self.n += 1
            val, j = outcome_value(self.st, self.ctx, self.call, res)
            self.val += val
            self.val2 += val * val
            r = j["result"]
            ok = r in ("continue", "win")
            self.made += ok
            self.foul += r == "foul" or res is None
            self.loss += r == "loss"
            if ok:
                self.valm += val
            else:
                self.valx += val
            if res is None:
                continue
            if r == "continue":
                self.nxt += SN._best_odds(j["state"], j["me"])[0]
            elif r in ("turnover", "foul") and j["opp"] is not None and "cue" not in res["pm"]:
                self.opp += SN._best_odds(j["state"], j["opp"])[0]

    def refine(self):
        """Shortlisted: find the exact centre of the potting window, restart sampling there."""
        phi, v, a, b = self.stroke
        c = self.call
        fine, w = window(self.st, c["ball"], c["pocket"], c["first"], phi, SN.FINE, 6, v, a, b)
        if fine is not None:
            self.stroke = (fine, v, a, b)
            self.wire["stroke"]["phi"] = float(fine % 360)
            self.base[F["window"]] = min(w, 2.0) / 2.0
        self.n = self.made = self.foul = self.loss = 0
        self.nxt = self.opp = self.val = self.valm = self.valx = self.val2 = 0.0

    def vector(self):
        f = self.base.copy()
        n = self.n
        f[F["noisy_make"]] = self.made / n
        f[F["foul_risk"]] = self.foul / n
        f[F["loss_risk"]] = self.loss / n
        f[F["noisy_next"]] = self.nxt / n
        f[F["opp_after_miss"]] = self.opp / n
        f[F["value_mean"]] = self.val / n
        f[F["value_make"]] = self.valm / self.made if self.made else 0.0
        miss = n - self.made
        f[F["value_miss"]] = self.valx / miss if miss else 0.0
        f[F["make_sq"]] = (self.made / n) ** 2
        f[F["value_spread"]] = max(0.0, self.val2 / n - (self.val / n) ** 2) ** .5
        return f


def base_features(st, ctx, call, stroke, q, width, ball, pk):
    """The noiseless part of the feature vector."""
    phi, v, a, s = stroke
    f = np.zeros(N)
    f[F["window"]] = min(width, 2.0) / 2.0
    f[F["odds"]] = q
    f[F["speed"]] = v / 3.0
    f[F["spin"]] = abs(s)
    f[F["side"]] = abs(a)
    f[F["bias"]] = 1.0
    B = st["balls"][ball]
    c = P.cue_xy(st)
    hole = P.POCKETS[pk]
    d = math.dist((B["x"], B["y"]), hole) or 1e-6
    g = ghost_ball_aim((B["x"], B["y"]), hole)
    vx, vy = g[0] - c[0], g[1] - c[1]
    f[F["straight"]] = ((vx * (hole[0] - B["x"]) + vy * (hole[1] - B["y"])) / d
                        / (math.hypot(vx, vy) or 1e-6))
    res = P.summarize(P.simulate(st, stroke), st)
    val, j = outcome_value(st, ctx, call, res)
    f[F["value_nominal"]] = val
    if res is None:
        return f
    f[F["scratch"]] = float("cue" in res["pm"])
    f[F["win"]] = float(j["result"] == "win")
    f[F["made"]] = float(j["result"] in ("continue", "win"))
    post = j["state"]
    if "cue" not in res["pm"]:
        x, y = P.cue_xy(post)
        f[F["cue_central"]] = min(x / P.TX, 1 - x / P.TX, y / P.TY, 1 - y / P.TY) * 2
        f[F["opp_odds"]] = SN._best_odds(post, j["opp"])[0] if j["opp"] else 0.0
    if j["result"] == "continue":
        f[F["next_odds"]], f[F["next_open"]] = SN._best_odds(post, j["me"])
    return f


def lane_options(st, ctx, b, pk, q, C, grid, tag):
    """Every stroke on a grid that verifiably pots ball b in pocket pk."""
    B = st["balls"][b]
    hole = tuple(float(x) for x in P._T.pockets[pk].center[:2])
    base = phi_towards(P.cue_xy(st), ghost_ball_aim((B["x"], B["y"]), hole))
    out, hint = [], None
    call = {"kind": "pot", "ball": b, "pocket": pk, "first": b}
    for si, (v, a, s) in enumerate(grid):
        centre = None
        if hint is not None:
            centre, width = window(st, b, pk, b, hint, SN.FINE * 3, 3, v, a, s)
        if centre is None:
            centre, width = window(st, b, pk, b, base, SN.COARSE, SN.COARSE_N, v, a, s)
        if centre is None:
            if si == 0:
                return out
            continue
        if hint is None:
            hint = centre
        stroke = (centre, v, a, s)
        f = base_features(st, ctx, call, stroke, q, width, b, pk)
        wire = {"call": {"ball": b, "pocket": pk},
                "stroke": {"phi": float(centre % 360), "speed": v, "side": a, "vertical": s}}
        if C is not None:
            wire["cue"] = {"x": float(C[0]), "y": float(C[1])}
            f[F["bih"]] = 1.0
        seed = zlib.crc32(f"{tag}|{b}|{pk}|{v}|{a}|{s}|{C}".encode())
        out.append(Opt(st, ctx, call, stroke, f, wire, seed))
    return out


def deep(st, ctx, legal, tag):
    """The search's own lane finder: thin cuts, banks, kicks, combos."""
    tg = set(legal)
    out = []
    for ln in P.direct_lanes(st, tg) + P.deep_lanes(st, tg):
        ln["lite"] = True
        seed = P._task_seed((st, ln))
        if not seed:
            continue
        b, pk, first, phi, v = seed["ball"], seed["pocket"], seed["first"], seed["phi"], seed["seed_v"]
        hits = sum(P.pot_check(st, (phi + d, v, 0.0, 0.0), b, pk, first) for d in (-0.4, -0.2, 0.0, 0.2, 0.4))
        call = {"kind": "pot", "ball": b, "pocket": pk, "first": first}
        stroke = (phi, float(v), 0.0, 0.0)
        f = base_features(st, ctx, call, stroke, 0.0, hits * 0.2, b, pk)
        f[F["deep"]] = 1.0
        wire = {"call": {"ball": b, "pocket": pk},
                "stroke": {"phi": float(phi % 360), "speed": float(v), "side": 0.0, "vertical": 0.0}}
        out.append(Opt(st, ctx, call, stroke, f, wire, zlib.crc32(f"{tag}|deep|{b}|{pk}|{first}|{phi:.4f}".encode())))
        if len(out) >= DEEP_MAX:
            break
    return out


def placements(st0, legal, kitchen):
    """Cue spots behind open lanes, same geometry as the search's ball in hand."""
    places = []
    for b in legal:
        B = st0["balls"][b]
        for pk, hole in P.POCKETS.items():
            if not P.pocket_lane_open(st0, b, pk):
                continue
            d = math.dist((B["x"], B["y"]), hole) or 1e-6
            ux, uy = (hole[0] - B["x"]) / d, (hole[1] - B["y"]) / d
            G = (B["x"] - ux * 2 * P.BALL_R, B["y"] - uy * 2 * P.BALL_R)
            for dist in (0.10, 0.22, 0.40):
                for ang in (0, 20, -20, 40, -40):
                    t = math.radians(ang)
                    dx, dy = -ux * math.cos(t) + uy * math.sin(t), -uy * math.cos(t) - ux * math.sin(t)
                    C = (G[0] + dx * dist, G[1] + dy * dist)
                    if not P.placement_ok(st0, C, kitchen):
                        continue
                    q = P.shot_prob(P.with_cue(st0, C), C, b, pk)
                    if q > 0.3:
                        places.append((q, b, pk, C))
    places.sort(key=lambda x: -x[0])
    seen, out = {}, []
    for q, b, pk, C in places:
        if seen.get((b, pk), 0) >= 2:
            continue
        seen[(b, pk)] = seen.get((b, pk), 0) + 1
        out.append((q, b, pk, C))
        if len(out) >= SPOTS:
            break
    return out


def options(turn):
    st0 = E.to_state(turn)
    me = E.GROUPS[turn.get("my_group")]
    opp = None if me is None else (P.STRIPES if me == P.SOLIDS else P.SOLIDS)
    ctx = {"me": me, "opp": opp}
    phase, legal = turn["phase"], turn["legal_targets"]
    tag = turn.get("shot_no", 0)
    out = []
    if phase == "shoot":
        c = P.cue_xy(st0)
        cands = sorted(((P.shot_prob(st0, c, b, pk), b, pk) for b in legal for pk in P.POCKETS),
                       key=lambda x: -x[0])
        best = 0.0
        used = 0
        for q, b, pk in cands:
            if q <= 0.02 or (used >= LANES and out):
                break
            o = lane_options(st0, ctx, b, pk, q, None, GRID, tag)
            used += bool(o)
            best = max(best, q if o else 0.0)
            out += o
        if not out or best < DEEP_BELOW:
            out += deep(st0, ctx, legal, tag)
    else:
        for q, b, pk, C in placements(st0, legal, phase == "ball_in_hand_kitchen"):
            out += lane_options(P.with_cue(st0, C), ctx, b, pk, q, C, BIH_GRID, tag)
    return out, st0, me, ctx


class ShotNet5:
    def __init__(self, w, tau=0.0, seed=0, log=None):
        if isinstance(w, str):
            with np.load(w) as z:
                w = {k: z[k] for k in z.files}
        self.w, self.tau, self.log = w, tau, log
        self.rng = np.random.default_rng(seed)
        self.book = SN._load_book()
        self.vnet = None
        if "v_W" in self.w:
            import vnet
            self.vnet = vnet.VNet(self.w)
        if "v2_A" in self.w:
            import vnet2
            self.vnet = vnet2.VNet2(self.w)

    def score(self, X):
        return (np.tanh(X @ self.w["W"]) @ self.w["v"]).ravel()

    def fallback(self, turn, st0, me, ctx):
        """Nothing callable: the lite search, which knows kicks and safeties."""
        phase = turn["phase"]
        try:
            if phase == "shoot":
                call = P.choose(st0, ctx, P.LITE, turn.get("shot_no", 0))
            else:
                call = P.choose_bih(st0, ctx, P.LITE, turn.get("shot_no", 0),
                                    kitchen=phase == "ball_in_hand_kitchen")
            return call_to_wire(call)
        except Exception:
            return baseline.action(st0, phase, me)

    def act(self, turn):
        global VNET
        VNET = self.vnet
        phase = turn["phase"]
        if phase == "break":
            return SN.break_wire(self.book, turn)
        opts, st0, me, ctx = options(turn)
        if not opts:
            return self.fallback(turn, st0, me, ctx)
        for o in opts:
            o.sample(NOISY0)
        s = self.score(np.array([o.vector() for o in opts]))
        keep = [opts[i] for i in np.argsort(-s)[:TOP]]
        for o in keep:
            o.refine()
            o.sample(NOISY0 + NOISY1)
        X = np.array([o.vector() for o in keep])
        s = self.score(X)
        if self.tau > 0:
            p = np.exp((s - s.max()) / self.tau)
            p /= p.sum()
            i = int(self.rng.choice(len(keep), p=p))
        else:
            i = int(np.argmax(s))
        if self.log is not None:
            self.log.append({"X": X.astype(np.float32), "i": i, "shot": turn.get("shot_no", 0)})
        return keep[i].wire


def linear(weights, g=0.05):
    """A one-unit net close to linear: score = sum(weights[k] * feature k)."""
    W, v = np.zeros((N, HIDDEN)), np.zeros((HIDDEN, 1))
    for k, x in weights.items():
        W[F[k], 0] = g * x
    v[0, 0] = 1.0 / g
    return {"W": W, "v": v}


if __name__ == "__main__":
    np.savez(sys.argv[1], **linear({"value_mean": 30.0}, g=0.02))
    print("features", FEATURES)
