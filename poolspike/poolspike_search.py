"""Poolspike-deep: Poolspike plus a search on its finalists, for when compute is cheap.

Poolspike (poolspike_player.ShotNet5) builds candidate
shots, plays each NOISY0 times with noise, shortlists TOP and plays those NOISY1
more, then takes the best mean win chance. This player keeps all of that and the
same judge, and adds two steps on the finalists:

1. Stroke refinement (every shot). Around the chosen pot, try speed +-0.6 m/s,
   draw/follow +-0.3 R and side +-0.25 R, re-aiming each. All variants are scored
   on the same 24 noise draws (common random numbers); a variant replaces the
   stroke only if its paired gain beats 2 standard errors.
2. Late-rack lookahead (4 or fewer of our balls left, the 8 counting as one).
   The top 3 finalists are refined, then played 32 times with noise. For the
   noisy executions that make the ball (up to 4 kept tables), the judge's value
   is replaced by the value of our best next pot from that table: a light
   search, then the pick is re-scored on fresh draws, so the max over noisy
   estimates does not bias it upward.

About 3.9x Poolspike's CPU per decision, measured in the same games.
`rla_late` in the logs (its development name).
"""
import math
import random
import zlib

import numpy as np

import poolspike_player as N5
from shotnet import P

LATE = 4                               # lookahead once this many of our balls (or fewer) are left
REFINE_K = 24                          # common noise draws per stroke variant
LA_TOP, LA_K, LA_KEEP = 3, 32, 4       # finalists, executions, made tables kept
LA_LANES, LA_S, LA_SHORT, LA_RE = 3, 4, 3, 12   # the next-shot search


def evaluate(st, ctx, call, stroke, seed, k, keep=0):
    """k noisy executions on a fixed draw stream; values, up to `keep` made tables, made mask."""
    rng = random.Random(seed)
    vals, cont, made = np.empty(k), [], np.zeros(k, bool)
    for i in range(k):
        res = P.summarize(P.simulate(st, P.noisy(stroke, rng, N5.NOISE)), st)
        v, j = N5.outcome_value(st, ctx, call, res)
        vals[i] = v
        if j["result"] == "continue":
            made[i] = True
            if len(cont) < keep:
                cont.append((i, j["state"], j["me"], j["opp"]))
    return vals, cont, made


def neighbours(o):
    phi, v, a, b = o.stroke
    out = []
    for dv, da, db in ((-.6, 0, 0), (.6, 0, 0), (0, 0, -.3), (0, 0, .3), (0, -.25, 0), (0, .25, 0)):
        v2, a2, b2 = min(6.0, max(0.6, v + dv)), max(-.6, min(.6, a + da)), max(-.6, min(.6, b + db))
        if (v2, a2, b2) == (v, a, b) or math.hypot(a2, b2) > 0.7:
            continue
        c = o.call
        fine, _ = N5.window(o.st, c["ball"], c["pocket"], c["first"], phi, N5.SN.FINE, 6, v2, a2, b2)
        if fine is not None:
            out.append((fine, v2, a2, b2))
    return out


def wire_for(o, stroke):
    w = {k: (dict(x) if isinstance(x, dict) else x) for k, x in o.wire.items()}
    w["stroke"] = {"phi": float(stroke[0] % 360), "speed": float(stroke[1]),
                   "side": float(stroke[2]), "vertical": float(stroke[3])}
    return w


def refine_stroke(o, seed, k=REFINE_K):
    """Best stroke around o's on common draws: (stroke, values)."""
    base, _, _ = evaluate(o.st, o.ctx, o.call, o.stroke, seed, k)
    if o.call.get("kind") != "pot":
        return o.stroke, base
    best, bv, gain = o.stroke, base, 0.0
    for s in neighbours(o):
        v, _, _ = evaluate(o.st, o.ctx, o.call, s, seed, k)
        d = v - base
        se = d.std(ddof=1) / math.sqrt(k) if k > 1 else 1.0
        if d.mean() > 2 * se and d.mean() > gain:
            best, bv, gain = s, v, d.mean()
    return best, bv


def best_next(st, me, opp, guess, tag):
    """Value of our best next pot from table st (select, then re-score on fresh draws).
    No pot available: the judge's value `guess`."""
    tg = P.targets(st, me)
    if not tg:
        return guess
    ctx = {"me": me, "opp": opp}
    c = P.cue_xy(st)
    cands = sorted(((P.shot_prob(st, c, b, pk), b, pk) for b in tg for pk in P.POCKETS), key=lambda x: -x[0])
    opts = []
    for q, b, pk in cands[:LA_LANES]:
        if q <= 0.02:
            break
        opts += N5.lane_options(st, ctx, b, pk, q, None, N5.GRID, tag)
    if not opts:
        return guess
    for o in opts:
        o.sample(LA_S)
    top = sorted(opts, key=lambda o: -o.val / o.n)[:LA_SHORT]
    for o in top:
        o.refine()
        o.sample(LA_RE)
    pick = max(top, key=lambda o: o.val / o.n)
    v, _, _ = evaluate(pick.st, pick.ctx, pick.call, pick.stroke, zlib.crc32(f"re|{tag}".encode()), LA_RE)
    return float(v.mean())


class PoolspikeDeep(N5.ShotNet5):
    def __init__(self, w, seed=0):
        super().__init__(w, seed=seed)
        self.seed = seed
        self.stats = {"decisions": 0, "switched": 0, "late": 0}

    def act(self, turn):
        N5.VNET = self.vnet
        phase = turn["phase"]
        if phase == "break":
            return N5.SN.break_wire(self.book, turn)
        me = N5.E.GROUPS[turn.get("my_group")]
        late = me is not None and len(P.targets(N5.E.to_state(turn), me)) <= LATE
        n_refine, n_look = (LA_TOP, LA_TOP) if late else (1, 0)
        self.stats["decisions"] += 1
        self.stats["late"] += late
        opts, st0, me, ctx = N5.options(turn)
        if not opts:
            return self.fallback(turn, st0, me, ctx)
        for o in opts:
            o.sample(N5.NOISY0)
        s = self.score(np.array([o.vector() for o in opts]))
        keep = [opts[i] for i in np.argsort(-s)[:N5.TOP]]
        for o in keep:
            o.refine()
            o.sample(N5.NOISY0 + N5.NOISY1)
        keep.sort(key=lambda o: -o.val / o.n)
        tag = turn.get("shot_no", 0)
        best, best_v = None, -1.0
        for r, o in enumerate(keep[:max(n_refine, n_look)]):
            seed = zlib.crc32(f"{self.seed}|{tag}|fam".encode())
            stroke, vals = refine_stroke(o, seed) if r < n_refine else (o.stroke, None)
            if n_look:
                vals, cont, made = evaluate(o.st, o.ctx, o.call, stroke,
                                            zlib.crc32(f"{self.seed}|{tag}|la".encode()), LA_K, keep=LA_KEEP)
                if cont:
                    la = np.mean([best_next(s_, m_, p_, vals[i], tag) for i, s_, m_, p_ in cont])
                    vals = np.where(made, la, vals)
            v = float(vals.mean()) if vals is not None else o.val / o.n
            if v > best_v:
                best, best_v = (o, stroke), v
        o, stroke = best
        if stroke != o.stroke:
            self.stats["switched"] += 1
            return wire_for(o, stroke)
        return o.wire
