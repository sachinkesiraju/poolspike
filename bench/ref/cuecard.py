"""CueCard rebuild: the 2008 Computer Olympiad computational-pool champion
(Archibald, Altman & Shoham, IJCAI 2009; AI Magazine 2010), rebuilt on our
pooltool simulator, noise model and rules. PickPocket (Smith, MSc thesis,
U. Alberta 2006) supplies the evaluation function and the lookup tables that
CueCard reused. Every number below is tagged with its source:
  [IJCAI] IJCAI 2009 paper, [AIM] AI Magazine 2010, [PP] PickPocket thesis,
  [dev] our choice where the papers are silent or our budget forces it.

Decision for a normal shot (IJCAI Sec. 3.1 steps 1-7):
  1. For each legal ball x pocket, a straight-in direction phi with minimum
     speed v0 (velocity table + 0.2 m/s boost [PP 3.1.6]). Banks, kicks and
     combos are generated when no straight-in variant is feasible or the
     best level-1 score is below B0 [PP 3.1.8 activation; dev: CueCard
     generated them always, our budget cannot].
  2. Speeds v_i from v0 up to VMAX = 4.5 in 1 m/s steps [PP 3.1.1]; each
     (phi, v_i) kept only if the noiseless shot pots the called ball [IJCAI].
  3. Variants: random a, b for each feasible (phi, v_i) [IJCAI step 3],
     each re-checked noiselessly [AIM]; each variant simulated N1 times with
     noise and every resulting state scored by the PickPocket evaluation
     1*p1 + 0.33*p2 + 0.15*p3 over the best straight-in shots [AIM, PP 3.2],
     failed samples scoring 0 [PP Fig 3.10]. Generation runs until the
     level-1 budget is spent (CueCard ran until a time limit [AIM]).
  3c/d. Top two variants per (phi, v_i) pair kept; their successful samples
     clustered into representative states [IJCAI 3.4].
  4. Top K variants go to level 2 [IJCAI: K = 20; dev: K = 4 for budget].
  5. Level 2 from each representative state: early game repeats steps 1-3
     with smaller constants and takes the best shot's mean evaluation
     [IJCAI 5a]; late game is a noiseless search to the end of the game,
     shots weighted by their table probability, misses valued by a
     Markov-chain (race) model of the opponent [IJCAI 3.4].
  6. Re-score the K variants with the level-2 values, play the best.
  7. If the best score is below LR_T or nothing is feasible: last-resort
     shot over all directions, valued vCC - vOPP [IJCAI 3.1].
Break: one precomputed shot from an offline search weighting keep-the-turn
over spread [IJCAI 3.3, AIM]. Ball in hand: grid of cue spots valued by the
best table probability, k-best local maxima by hill climbing, a shortened
single-level search from each [AIM, PP 3.4.2].

Tables (build once per noise model, `cuecard.py build`): a 4-D straight-in
success-probability table over (cut, object-pocket distance, cue-ghost
distance, pocket approach angle) for corner and side pockets, filled by
noisy simulation [PP 3.1.5], and the minimum-speed table [PP 3.1.6].

  cuecard.py build [WORKERS]       tables + break for the active noise model
  cuecard.py sanity                simple-position and timing checks
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("POOL_COWORLD", os.path.join(os.path.dirname(HERE), "pool8"))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("POOL_WORKERS", "1")

import noise_models  # noqa: E402
import policy as P  # noqa: E402
from pool_env import ghost_ball_aim, phi_towards  # noqa: E402

DATA = os.environ.get("CC_DATA", os.path.join(HERE, "cc_data"))
R = P.BALL_R
PC = {pk: tuple(float(x) for x in P._T.pockets[pk].center[:2]) for pk in P.POCKETS}
CORNERS = ("lb", "lt", "rb", "rt")

# ---------------------------------------------------------------- parameters
EVAL_W = (1.0, 0.33, 0.15)       # [PP 3.2, AIM] discount weights, n = 3
MAX_EVAL = sum(EVAL_W)           # 1.48 [PP 3.1.7]
V_BOOST = 0.2                    # [PP 3.1.6] added to the table's minimum speed
V_STEP = 1.0                     # [PP 3.1.1] speed increments
VMAX = 4.5                       # [PP 3.1.1] POOLFIZ maximum
TIP_R = 0.5                      # [dev] random english radius (tip units of R)
B0 = 0.8                         # [PP 3.1.8] bank/kick/combo activation
LR_T = 0.10                      # [dev] last-resort threshold (paper: "a certain threshold")
CLUSTER_D = 0.05                 # [dev] max ball displacement (m) inside a cluster
CLUSTER_V = 0.15                 # [dev] max evaluation gap inside a cluster
LATE_LEFT = 3                    # [dev] "late game": own balls + 8 left <= 3
Q_RACE = float(os.environ.get("CC_Q", 0.80))   # [dev] Markov-chain make rate, from self-play
# compute budget (units ~ one full simulation); scaled by the spec's multiplier
N1 = 25                          # [IJCAI] samples per variant: 25-100 (we use the minimum)
N_VAR = 8                        # [dev] max random-english variants per (phi, v) pair (budget-limited)
N_SCREEN, TOPUP = 6, 6           # [dev] screening samples per variant; variants sampled to N1
B1 = 330                         # [dev] level-1 budget
K_TOP = 3                        # [dev] variants to level 2 (paper: 20)
C_MAX = 3                        # [dev] representative states per variant
N2, L2_LANES, L2_SPEEDS = 4, 3, 2  # [dev] "smaller constants" at level 2
B2_FRAC = 0.75                   # [dev] level-2 budget as a fraction of B1
BIH_K, BIH_C = 4, 24             # [dev] k-best local maxima from c hill climbs [PP 3.4.2]
LR_DIRS, LR_SPEEDS, LR_KEEP, LR_N = 72, (1.6, 3.0), 6, 6  # [dev] last-resort sweep

# probability / velocity tables [PP 3.1.5-6; dev: coarser grid, s = 48]
AL = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80], float)     # cut (deg)
D1 = np.array([0.06, 0.15, 0.3, 0.5, 0.75, 1.05, 1.4, 1.8])   # object-pocket (m)
D2 = np.array([0.06, 0.15, 0.3, 0.5, 0.75, 1.05, 1.4, 1.8])   # cue-ghost (m)
BE = {"corner": np.array([0, 15, 30, 42], float),             # approach angle (deg)
      "side": np.array([0, 15, 30, 45, 60], float)}
TABLE_S = 40


def noise_name() -> str:
    return noise_models.ACTIVE["noise"]


# ------------------------------------------------------------------ geometry

def _rot(u, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return (u[0] * c - u[1] * s, u[0] * s + u[1] * c)


def _normal(pk):
    return P._NORMAL[pk]


def shot_params(C, B, pk):
    """(cut, d1, d2, beta, G) for a straight-in shot, or None if impossible."""
    Pp = PC[pk]
    d1 = math.dist(B, Pp)
    if d1 < 1e-6:
        return None
    u = ((Pp[0] - B[0]) / d1, (Pp[1] - B[1]) / d1)
    G = (B[0] - 2 * R * u[0], B[1] - 2 * R * u[1])
    if not (R * 0.5 < G[0] < P.TX - R * 0.5 and R * 0.5 < G[1] < P.TY - R * 0.5):
        return None
    w = (G[0] - C[0], G[1] - C[1])
    d2 = math.hypot(*w)
    if d2 < 1e-6:
        return None
    cos_a = (w[0] * u[0] + w[1] * u[1]) / d2
    cut = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))
    if cut >= 85:
        return None
    n = _normal(pk)
    beta = math.degrees(math.acos(max(-1.0, min(1.0, -(u[0] * n[0] + u[1] * n[1])))))
    lim = 47 if pk in CORNERS else 70
    if beta > lim:
        return None
    return cut, d1, d2, beta, G


class Tables:
    def __init__(self, path):
        z = np.load(path)
        self.p = {k: z["p_" + k] for k in ("corner", "side")}
        self.v = {k: z["v_" + k] for k in ("corner", "side")}

    @staticmethod
    def _interp(T, axes, x):
        """Multilinear interpolation (PickPocket used nearest entry [PP 3.1.5])."""
        idx, wts = [], []
        for a, xi in zip(axes, x):
            xi = min(max(xi, a[0]), a[-1])
            j = int(np.searchsorted(a, xi, side="right") - 1)
            j = min(max(j, 0), len(a) - 2)
            t = (xi - a[j]) / (a[j + 1] - a[j])
            idx.append(j)
            wts.append(t)
        out = 0.0
        for corner in range(16):
            w, ix = 1.0, []
            for d in range(4):
                bit = (corner >> d) & 1
                w *= wts[d] if bit else 1 - wts[d]
                ix.append(idx[d] + bit)
            if w:
                out += w * T[tuple(ix)]
        return float(out)

    def lookup(self, cut, d1, d2, beta, pk):
        kind = "corner" if pk in CORNERS else "side"
        ax = (AL, D1, D2, BE[kind])
        x = (cut, d1, d2, beta)
        return self._interp(self.p[kind], ax, x), self._interp(self.v[kind], ax, x)


_TABLES = {}


def tables() -> Tables:
    name = noise_name()
    if name not in _TABLES:
        _TABLES[name] = Tables(os.path.join(DATA, f"{name}_tables.npz"))
    return _TABLES[name]


def straight_shots(state, me, cxy=None, T=None):
    """Feasible straight-in shots for the mover: [(p, vmin, ball, pocket, G)]."""
    T = T or tables()
    cxy = cxy or P.cue_xy(state)
    out = []
    for b in P.targets(state, me):
        bl = state["balls"][b]
        B = (bl["x"], bl["y"])
        for pk in P.POCKETS:
            sp = shot_params(cxy, B, pk)
            if sp is None:
                continue
            cut, d1, d2, beta, G = sp
            if not P._seg_clear(B, PC[pk], state, {b, "cue"}):
                continue
            if not P._seg_clear(cxy, G, state, {b, "cue"}):
                continue
            p, vmin = T.lookup(cut, d1, d2, beta, pk)
            out.append((p, vmin, b, pk, G))
    out.sort(key=lambda r: -r[0])
    return out


def evaluate(state, me, T=None) -> float:
    """PickPocket evaluation [PP 3.2]: 1*p1 + 0.33*p2 + 0.15*p3 over the
    mover's straight-in shots, one per ball-pocket pair."""
    ps = [s[0] for s in straight_shots(state, me, T=T)[:3]]
    return sum(w * p for w, p in zip(EVAL_W, ps))


def pots_ball(state, stroke, ball, pocket, first) -> bool:
    """Noiseless feasibility [IJCAI step 2, AIM]: the called ball drops in the
    called pocket after the cue first hits `first`. A scratch does not make
    the direction infeasible (the sampled variants pay for scratches)."""
    import fastsim
    import pooltool as pt
    from pool_env import build_system
    phi, v, a, b = stroke
    seen = []

    def stop(e):
        et = e.event_type
        if et == pt.EventType.BALL_BALL and not seen:
            seen.append(True)
            return next(x.id for x in e.agents if x.id != "cue") != first
        if et == pt.EventType.BALL_POCKET:
            bid, pid = e.agents[0].id, e.agents[1].id
            return bid == ball
        return False
    try:
        sys_ = build_system(state)
        sys_.cue.set_state(V0=v, phi=phi, theta=0.0, a=a, b=b)
        fastsim.simulate_until(sys_, stop)
    except Exception:  # noqa: BLE001
        return False
    res = P.summarize(sys_, state)
    return res is not None and res["pm"].get(ball) == pocket and res["fc"] == first


def solve_to(st, b, pk, tgt, v, a=0.0, bb=0.0, phi0=None):
    """Aim solver toward one target point in the pocket mouth."""
    c = P.cue_xy(st)
    bl = st["balls"][b]
    g = ghost_ball_aim((bl["x"], bl["y"]), tgt)
    return P._aim_solve(st, b, pk, phi_towards(c, g) if phi0 is None else phi0, v, a, bb, target=tgt)[0]


AIM_STEP, AIM_NMAX = 0.06, 6


def aim_direct(st, b, pk, v, a=0.0, bb=0.0):
    """Direction that sinks b in pk at this stroke [IJCAI step 1], robust to
    pooltool's throw and pocket jaws [dev]: aim-solve toward points across the
    pocket mouth, scan a little around each solution, then centre the aim in
    the potting window (up to +-0.36 deg). Returns (phi, target, offset) or
    None; offset = centred phi minus the solver's phi for that target."""
    for tgt in P.mouth_targets(pk):
        phi = solve_to(st, b, pk, tgt, v, a, bb)
        if phi is None:
            continue
        for d in P.SCAN:
            if pots_ball(st, (phi + d, v, a, bb), b, pk, b):
                base = phi + d
                lo = hi = 0
                while lo < AIM_NMAX and pots_ball(st, (base - (lo + 1) * AIM_STEP, v, a, bb), b, pk, b):
                    lo += 1
                while hi < AIM_NMAX and pots_ball(st, (base + (hi + 1) * AIM_STEP, v, a, bb), b, pk, b):
                    hi += 1
                centre = base + (hi - lo) * AIM_STEP / 2
                return centre, tgt, centre - phi
    return None


# ------------------------------------------------------------- race model

_RACE = {}


def opp_wins(n_opp: int, n_me: int, q: float = Q_RACE) -> float:
    """Markov-chain estimate [IJCAI 3.4] that the player to move with n_opp
    shots left beats one with n_me left (per-shot make rate q, both sides)."""
    return P._race(n_opp, n_me, q, q)


# ------------------------------------------------------------- the player

class Budget:
    def __init__(self, units):
        self.left = units

    def spend(self, u=1.0):
        self.left -= u
        return self.left >= 0


class CueCard:
    def __init__(self, spec="cc", seed=0):
        name, _, arg = str(spec).partition(":")
        self.mult = float(arg) if arg else 1.0
        self.seed = seed
        self.stats = {"decisions": 0, "last_resort": 0, "deep": 0, "late": 0, "bih": 0, "level2_changed": 0}
        self.T = tables()
        with open(os.path.join(DATA, f"{noise_name()}_break.json")) as f:
            self.brk = json.load(f)["best"]
        self.cfg = P.DEFAULT
        self.trace = {}

    # -------------------------------------------------- outcome of one sample
    def _sample(self, st, ctx, call, stroke, rng):
        res = P.summarize(P.simulate(st, P.noisy(stroke, rng, self.cfg)), st)
        j = P.adjudicate(st, ctx, call, res)
        return j

    def _late(self, st, me):
        return me is not None and P.n_left(st, me) <= LATE_LEFT

    def _fail_value(self, j, late, ctx_after):
        """Value of a sample where the turn is lost [PP: 0 early; IJCAI late:
        the Markov chain for the opponent]."""
        if j["result"] == "loss":
            return 0.0
        if j["result"] == "win":
            return 1.0 if late else MAX_EVAL
        if not late:
            return 0.0
        st = j["state"]
        n_me = P.n_left(st, ctx_after["me"])
        n_opp = P.n_left(st, ctx_after["opp"])
        q = min(0.97, Q_RACE + 0.1) if j["result"] == "foul" else Q_RACE
        return 1.0 - P._race(n_opp, n_me, q, Q_RACE)

    # -------------------------------------------------- level 1 generation
    def _pairs_direct(self, st, me, bud, speeds_max=None, lanes_max=None):
        """Steps 1-2: feasible (ball, pocket, phi, v) for straight-in shots.
        The direction is found once per lane (aim_direct at a medium speed),
        then re-solved toward the same mouth point for each speed v_i."""
        out = []
        shots = straight_shots(st, me, T=self.T)
        if lanes_max:
            shots = shots[:lanes_max]
        for p, vmin, b, pk, G in shots:
            v0 = max(0.5, vmin + V_BOOST)
            vs = []
            v = v0
            while v <= VMAX + 1e-9:
                vs.append(round(v, 3))
                v += V_STEP
            if not vs:
                vs = [min(VMAX, v0)]
            if speeds_max:
                vs = vs[:speeds_max]
            if not bud.spend(4.0):
                return out
            vb = min(vs, key=lambda x: abs(x - 2.0))
            ad = aim_direct(st, b, pk, vb)
            if ad is None:
                continue
            phi_b, tgt, off = ad
            for v in vs:
                if v == vb:
                    phi = phi_b
                else:
                    if not bud.spend(1.0):
                        return out
                    phi = solve_to(st, b, pk, tgt, v, phi0=phi_b)
                    if phi is None:
                        continue
                    phi += off
                    if not pots_ball(st, (phi, v, 0.0, 0.0), b, pk, b):
                        continue
                out.append({"ball": b, "pocket": pk, "first": b, "phi": phi, "v": v,
                            "p_tab": p, "route": "direct", "tgt": tgt, "off": off})
        return out

    def _pairs_deep(self, st, me, bud):
        """Banks, kicks, combos [PP 3.1.2-3.1.4] from the shared geometric lane
        finder; a lane is kept at the first aim/speed that pots noiselessly."""
        out = []
        try:
            lanes = P.deep_lanes(st, P.targets(st, me))
        except Exception:
            return out
        c = P.cue_xy(st)
        for ln in lanes:
            found = None
            for aim in ln["aims"][:5]:
                for v in ln["speeds"][1:2] + ln["speeds"][:1] + ln["speeds"][2:]:
                    if not bud.spend(1.0):
                        return out
                    phi = phi_towards(c, aim)
                    if pots_ball(st, (phi, v, 0.0, 0.0), ln["ball"], ln["pocket"], ln["first"]):
                        found = (phi, v)
                        break
                if found:
                    break
            if found:
                for v in sorted({found[1], max(1.0, found[1] - 0.8), min(VMAX, found[1] + 0.8)}):
                    if v != found[1]:
                        if not bud.spend(1.0) or not pots_ball(
                                st, (found[0], v, 0.0, 0.0), ln["ball"], ln["pocket"], ln["first"]):
                            continue
                    out.append({"ball": ln["ball"], "pocket": ln["pocket"], "first": ln["first"],
                                "phi": found[0], "v": v, "p_tab": 0.0, "route": ln["route"]})
        return out

    def _run_samples(self, st, ctx, v, k, rng, bud, late):
        for _ in range(k):
            bud.spend(1.0)
            j = self._sample(st, ctx, v["call"], v["stroke"], rng)
            if j["result"] == "continue":
                val = evaluate(j["state"], j["me"], T=self.T)
                v["samples"].append({"st": j["state"], "me": j["me"], "opp": j["opp"], "val": val})
            else:
                # level 1 ranks PickPocket-style (failure = 0); the late-game
                # value of the same failure is kept for the level-2 re-score
                ca = {"me": j["me"], "opp": j["opp"]}
                val = self._fail_value(j, False, ca)
                v["samples"].append({"st": None, "val": val,
                                     "late": self._fail_value(j, True, ca) if late else val})
            v["tot"] += val
        v["n"] = len(v["samples"])
        v["l1"] = v["tot"] / v["n"]

    def _level1(self, st, ctx, pairs, rng, bud, n, keep_states=True):
        """Step 3: variants of each pair (plain first, then random english),
        each noiselessly re-checked then sampled. [dev] Budget: every variant
        is first screened with N_SCREEN samples (PickPocket's root sampling,
        PP 3.1.9); the TOPUP best are then sampled up to n = 25 [IJCAI], and
        only those can be chosen."""
        me = ctx["me"]
        late = self._late(st, me)
        variants = []
        topup = max(1, min(round(TOPUP * self.mult), int(0.4 * max(bud.left, 0) // max(1, n - N_SCREEN))))
        reserve = topup * max(0, n - N_SCREEN)
        rounds = [(0.0, 0.0)] + [None] * N_VAR
        done = False
        for r_i, ab in enumerate(rounds):
            if done:
                break
            for pi, pr in enumerate(pairs):
                if bud.left - reserve < N_SCREEN:
                    done = True
                    break
                if ab is None:
                    rr = TIP_R * math.sqrt(rng.random())
                    th = rng.random() * 2 * math.pi
                    a, b = rr * math.cos(th), rr * math.sin(th)
                    bud.spend(1.0)
                    phi = pr["phi"]
                    if pr["route"] == "direct":
                        # [dev] direction re-solved for this english (squirt / throw in
                        # pooltool); CueCard kept phi and re-checked noiselessly [AIM]
                        phi = solve_to(st, pr["ball"], pr["pocket"], pr["tgt"], pr["v"], a, b, phi0=pr["phi"])
                        if phi is None:
                            continue
                        phi += pr["off"]
                    if not pots_ball(st, (phi, pr["v"], a, b), pr["ball"], pr["pocket"], pr["first"]):
                        continue
                else:
                    a, b = ab
                    phi = pr["phi"]
                call = {"kind": "pot", "ball": pr["ball"], "pocket": pr["pocket"], "first": pr["first"]}
                v = {"pair": pi, "call": call, "stroke": (phi, pr["v"], a, b), "route": pr["route"],
                     "samples": [], "tot": 0.0, "n": 0, "l1": 0.0, "full": False}
                self._run_samples(st, ctx, v, min(N_SCREEN, n), rng, bud, late)
                variants.append(v)
        for v in sorted(variants, key=lambda v: -v["l1"])[:topup]:
            self._run_samples(st, ctx, v, n - v["n"], rng, bud, late)
            v["full"] = True
        return [v for v in variants if v["full"]]

    # -------------------------------------------------- clustering (3d)
    @staticmethod
    def _dist(s1, s2):
        b1, b2 = s1["balls"], s2["balls"]
        d = 0.0
        for k, x in b1.items():
            y = b2.get(k)
            if y is None or bool(x.get("pocketed")) != bool(y.get("pocketed")):
                return math.inf
            if not x.get("pocketed"):
                d = max(d, abs(x["x"] - y["x"]), abs(x["y"] - y["y"]))
        return d

    def _clusters(self, samples):
        reps = []      # [state, me, opp, val, weight]
        for sm in samples:
            st, me, opp, val = sm["st"], sm.get("me"), sm.get("opp"), sm["val"]
            if st is None:
                continue
            for r in reps:
                if abs(r[3] - val) <= CLUSTER_V and self._dist(r[0], st) <= CLUSTER_D:
                    r[4] += 1
                    break
            else:
                reps.append([st, me, opp, val, 1])
        reps.sort(key=lambda r: -r[4])
        cmax = min(10, round(C_MAX * math.sqrt(self.mult)))
        keep, rest = reps[:cmax], reps[cmax:]
        for r in rest:      # fold small clusters into the nearest kept one
            if keep:
                best = min(keep, key=lambda k: (self._dist(k[0], r[0]), abs(k[3] - r[3])))
                best[4] += r[4]
        return keep

    # -------------------------------------------------- level 2
    def _level2_early(self, st, me, opp, rng, bud):
        ctx = {"me": me, "opp": opp}
        pairs = self._pairs_direct(st, me, bud, speeds_max=L2_SPEEDS, lanes_max=L2_LANES)
        if not pairs:
            return 0.0
        # samples per pair from this state's share of the level-2 budget (>= 2)
        n2 = max(2, min(N2, int(max(bud.left, 0) // len(pairs))))
        pairs = sorted(pairs, key=lambda q: -q["p_tab"])[:max(1, int(max(bud.left, 2 * len(pairs)) // n2))]
        best = 0.0
        for pr in pairs:
            call = {"kind": "pot", "ball": pr["ball"], "pocket": pr["pocket"], "first": pr["first"]}
            tot = 0.0
            for _ in range(n2):
                bud.spend(1.0)
                j = self._sample(st, ctx, call, (pr["phi"], pr["v"], 0.0, 0.0), rng)
                if j["result"] == "continue":
                    tot += evaluate(j["state"], j["me"], T=self.T)
                elif j["result"] == "win":
                    tot += MAX_EVAL
            best = max(best, tot / n2)
        return best

    def _win_search(self, st, me, opp, depth, bud):
        """Late game [IJCAI 3.4]: noiseless search to the end, each shot
        weighted by its table probability; a miss leaves the same balls and
        the opponent's chances come from the Markov chain."""
        if me is None:
            return 0.5
        tg = P.targets(st, me)
        n_me, n_opp = P.n_left(st, me), P.n_left(st, opp)
        miss = 1.0 - P._race(n_opp, n_me, Q_RACE, Q_RACE)
        shots = straight_shots(st, me, T=self.T)[: (3 if depth == 0 else 2)]
        best = miss * 0.5   # nothing makeable: roughly a safety / give-up
        for p, vmin, b, pk, G in shots:
            if b == "8" and tg == {"8"}:
                best = max(best, p + (1 - p) * miss)
                continue
            if depth >= 3 or bud.left <= 0:
                best = max(best, p * 0.9 + (1 - p) * miss)
                continue
            for v in ((max(0.5, vmin + V_BOOST), max(0.5, vmin + V_BOOST) + V_STEP) if depth == 0
                      else (max(0.5, vmin + V_BOOST) + 0.5,)):
                if v > VMAX:
                    continue
                bud.spend(4.0)
                ad = aim_direct(st, b, pk, v)
                if ad is None:
                    break
                phi = ad[0]
                res = P.summarize(P.simulate(st, (phi, v, 0.0, 0.0)), st)
                j = P.adjudicate(st, {"me": me, "opp": opp},
                                 {"kind": "pot", "ball": b, "pocket": pk, "first": b}, res)
                if j["result"] == "win":
                    w = 1.0
                elif j["result"] == "continue":
                    w = self._win_search(j["state"], j["me"], j["opp"], depth + 1, bud)
                else:
                    continue
                best = max(best, p * w + (1 - p) * miss)
        return best

    def _refine(self, st, ctx, variants, rng, bud):
        """Steps 4-6: top-2 per pair, top K overall, level-2 re-score."""
        me = ctx["me"]
        late = self._late(st, me)
        by_pair = {}
        for v in variants:
            by_pair.setdefault(v["pair"], []).append(v)
        pool = []
        for vs in by_pair.values():
            pool += sorted(vs, key=lambda v: -v["l1"])[:2]
        top = sorted(pool, key=lambda v: -v["l1"])[:min(20, round(K_TOP * self.mult))]   # paper: 20
        if late:
            self.stats["late"] += 1
        reps_of = {id(v): self._clusters(v["samples"]) for v in top}
        n_reps = max(1, sum(len(r) for r in reps_of.values()))
        share = bud.left / n_reps            # equal level-2 budget per representative state
        for v in top:
            succ = [s for s in v["samples"] if s["st"] is not None]
            fails = sum(s["late"] if late else s["val"] for s in v["samples"] if s["st"] is None)
            tot = fails
            for stc, mec, oppc, val, w in reps_of[id(v)]:
                b = Budget(share)
                if late:
                    l2 = self._win_search(stc, mec, oppc, 0, b)
                else:
                    l2 = self._level2_early(stc, mec, oppc, rng, b)
                tot += w * l2
            v["l2"] = tot / v["n"] if succ else fails / v["n"]
        return top

    # -------------------------------------------------- last resort
    def _last_resort(self, st, ctx, rng, bud):
        self.stats["last_resort"] += 1
        me, opp = ctx["me"], ctx["opp"]
        tg = P.targets(st, me)
        cands = []
        for k in range(LR_DIRS):
            phi = 360.0 * k / LR_DIRS
            for v in LR_SPEEDS:
                res = P.summarize(P.simulate(st, (phi, v, 0.0, 0.0)), st)
                if res is None:
                    continue
                call = {"kind": "safe"}
                pots = [b for b in res["pm"] if b != "cue" and b in tg]
                if len(pots) == 1 and "8" not in res["pm"]:
                    call = {"kind": "pot", "ball": pots[0], "pocket": res["pm"][pots[0]], "first": res["fc"]}
                j = P.adjudicate(st, ctx, call, res)
                if j["result"] in ("foul", "loss"):
                    continue
                cands.append((self._lr_value(j, ctx), phi, v, call))
        if not cands:
            return None
        cands.sort(key=lambda c: -c[0])
        best, bv = None, -9.0
        for _, phi, v, call in cands[:LR_KEEP]:
            tot = 0.0
            for _ in range(LR_N):
                j = self._sample(st, ctx, call, (phi, v, 0.0, 0.0), rng)
                tot += self._lr_value(j, ctx)
            if tot / LR_N > bv:
                bv, best = tot / LR_N, (phi, v, call)
        return best

    def _lr_value(self, j, ctx):
        """vLR = vCC - vOPP [IJCAI 3.1]."""
        r = j["result"]
        if r == "win":
            return MAX_EVAL
        if r == "loss":
            return -MAX_EVAL
        if r == "continue":
            return evaluate(j["state"], j["me"], T=self.T)
        if r == "foul":
            return -0.9 * MAX_EVAL      # [dev] opponent's ball in hand
        return -evaluate(j["state"], j["opp"], T=self.T)

    # -------------------------------------------------- decisions
    def _decide(self, st, ctx, rng, mult=1.0, level2=True):
        """Returns (call, stroke, score) or None when nothing is feasible."""
        bud = Budget(B1 * self.mult * mult)
        me = ctx["me"]
        pairs = self._pairs_direct(st, me, bud)
        n = N1
        variants = self._level1(st, ctx, pairs, rng, bud, n)
        best1 = max((v["l1"] for v in variants), default=-1.0)
        if best1 < B0 * (1.0 if not self._late(st, me) else 0.0) or not variants:
            self.stats["deep"] += 1
            dbud = Budget(0.3 * B1 * self.mult * mult)
            deep = self._pairs_deep(st, me, dbud)
            if deep:
                base = len(pairs)
                dv = self._level1(st, ctx, deep, rng, Budget(0.4 * B1 * self.mult * mult), n)
                for v in dv:
                    v["pair"] += base
                variants += dv
        self.trace = {"pairs": len(pairs), "variants": len(variants), "best1": round(best1, 3),
                      "late": self._late(st, me), "left1": round(bud.left, 1)}
        if not variants:
            return None
        if level2:
            top = self._refine(st, ctx, variants, rng, Budget(B2_FRAC * B1 * self.mult * mult))
            pick = max(top, key=lambda v: (round(v["l2"], 9), v["l1"]))
            l1pick = max(variants, key=lambda v: v["l1"])
            if pick is not l1pick:
                self.stats["level2_changed"] += 1
            self.trace["succ"] = sum(x["st"] is not None for x in pick["samples"]) / pick["n"]
            self.trace["l2s"] = [round(v["l2"], 3) for v in top]
            self.trace["l1s"] = [round(v["l1"], 3) for v in top]
            self.trace["nvar"] = len(variants)
            return pick["call"], pick["stroke"], pick["l1"], pick["l2"]
        pick = max(variants, key=lambda v: v["l1"])
        return pick["call"], pick["stroke"], pick["l1"], pick["l1"]

    def _wire(self, call, stroke, place=None, ev=None):
        phi, v, a, b = stroke
        w = {"stroke": {"phi": float(phi) % 360.0, "speed": float(min(max(v, 0.3), 8.0)),
                        "side": float(a), "vertical": float(b)},
             "call": ({"ball": call["ball"], "pocket": call["pocket"]} if call["kind"] == "pot" else None),
             "meta": {"kind": call["kind"], "ev": ev, "p_make": None}}
        if place is not None:
            w["cue"] = {"x": float(place[0]), "y": float(place[1])}
        return w

    def _break(self, st):
        b = self.brk
        C = (float(b["x"]), float(b["y"]))
        apex = min(P.live(st), key=lambda k: st["balls"][k]["y"])
        A = (st["balls"][apex]["x"], st["balls"][apex]["y"])
        phi = phi_towards(C, A) + float(b.get("dphi", 0.0))
        return {"stroke": {"phi": phi % 360.0, "speed": float(b["v"]), "side": float(b["a"]),
                           "vertical": float(b["b"])}, "call": None, "cue": {"x": C[0], "y": C[1]},
                "meta": {"kind": "break", "ev": b.get("score"), "p_make": None}}

    def _bih_spots(self, st, me, kitchen, rng):
        """Grid of cue spots valued by the best straight-in table probability,
        k-best local maxima from c random hill climbs [PP 3.4.2]."""
        nx, ny = 10, 20
        xs = [(i + 0.5) * P.TX / nx for i in range(nx)]
        ys = [(j + 0.5) * P.TY / ny for j in range(ny)]
        grid = np.full((nx, ny), -1.0)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if not P.placement_ok(st, (x, y), kitchen):
                    continue
                s2 = P.with_cue(st, (x, y))
                ss = straight_shots(s2, me, (x, y), T=self.T)
                grid[i, j] = ss[0][0] if ss else 0.0
        if grid.max() < 0:
            return []
        maxima = set()
        valid = [(i, j) for i in range(nx) for j in range(ny) if grid[i, j] >= 0]
        for _ in range(BIH_C):
            i, j = valid[rng.randrange(len(valid))]
            while True:
                nb = [(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                      if 0 <= i + di < nx and 0 <= j + dj < ny]
                bi, bj = max(nb, key=lambda q: grid[q])
                if grid[bi, bj] <= grid[i, j]:
                    break
                i, j = bi, bj
            maxima.add((i, j))
        best = sorted(maxima, key=lambda q: -grid[q])[:BIH_K]
        return [(xs[i], ys[j]) for i, j in best]

    def act(self, turn):
        w = self._act(turn)
        tr = os.environ.get("CC_TRACE")
        if tr:
            with open(tr, "a") as f:
                f.write(json.dumps({"seed": self.seed, "shot": turn.get("shot_no"), "phase": turn["phase"],
                                    "kind": w["meta"]["kind"], "ev": w["meta"].get("ev"),
                                    "call": w.get("call"), "speed": w["stroke"]["speed"],
                                    **getattr(self, "trace", {}),
                                    **({"turn": turn} if os.environ.get("CC_TRACE_STATES") and (
                                        self.trace.get("lr") or self.trace.get("pairs") == 0) else {})}) + "\n")
        self.trace = {}
        return w

    def _act(self, turn):
        from rl import encode as E
        self.stats["decisions"] += 1
        st0 = E.to_state(turn)
        phase = turn["phase"]
        rng = random.Random(zlib.crc32(f"cc{self.seed}:{turn.get('shot_no', 0)}:{turn.get('slot', 0)}".encode()))
        if phase == "break":
            return self._break(st0)
        me, opp = E.GROUPS[turn["my_group"]], E.GROUPS[turn["opp_group"]]
        ctx = {"me": me, "opp": opp}
        if phase.startswith("ball_in_hand"):
            self.stats["bih"] += 1
            kitchen = phase == "ball_in_hand_kitchen"
            spots = self._bih_spots(st0, me, kitchen, rng)
            best = None
            for xy in spots:
                st = P.with_cue(st0, xy)
                d = self._decide(st, ctx, rng, mult=1.0 / max(1, len(spots)) * 1.2, level2=False)
                if d and (best is None or d[3] > best[0][3]):
                    best = (d, xy)
            if best and best[0][3] >= LR_T:
                d, xy = best
                return self._wire(d[0], d[1], xy, d[3])
            # nothing good: place centrally-ish at the best spot and take the last resort
            xy = best[1] if best else next((s for s in spots), None)
            if xy is None:
                xy = self._any_spot(st0, kitchen)
            st = P.with_cue(st0, xy)
            lr = self._last_resort(st, ctx, rng, None)
            if lr:
                return self._wire(lr[2], (lr[0], lr[1], 0.0, 0.0), xy)
            if best:
                return self._wire(best[0][0], best[0][1], xy, best[0][3])
            return self._wire({"kind": "safe"}, (90.0, 2.0, 0.0, 0.0), xy)
        d = self._decide(st0, ctx, rng)
        # [dev] the last-resort test uses the better of the shot's level-1 and
        # level-2 values: a near-sure pot whose 2-ply leave has no straight-in
        # shot (PickPocket evaluation 0) is not a "no suitable shot" state
        if d and max(d[2], d[3]) >= LR_T:
            return self._wire(d[0], d[1], None, d[3])
        lr = self._last_resort(st0, ctx, rng, None)
        self.trace["lr"] = True
        if lr:
            return self._wire(lr[2], (lr[0], lr[1], 0.0, 0.0))
        if d:
            return self._wire(d[0], d[1], None, d[3])
        # no legal non-fouling stroke found: hit the nearest legal ball
        tg = sorted(P.targets(st0, me))
        c = P.cue_xy(st0)
        b = min(tg, key=lambda k: math.dist(c, (st0["balls"][k]["x"], st0["balls"][k]["y"])))
        return self._wire({"kind": "safe"}, (phi_towards(c, (st0["balls"][b]["x"], st0["balls"][b]["y"])),
                                             2.0, 0.0, 0.0))

    @staticmethod
    def _any_spot(st, kitchen):
        for y in np.linspace(0.1, (P.HEAD_STRING - 0.05) if kitchen else P.TY - 0.1, 30):
            for x in np.linspace(0.1, P.TX - 0.1, 12):
                if P.placement_ok(st, (x, y), kitchen):
                    return (float(x), float(y))
        return (P.TX / 2, P.HEAD_STRING / 2)


# ================================================================ offline build

def _table_geom(kind, ia, i1, i2, ib):
    """Two-ball layout for one table entry, or None if it doesn't fit."""
    pk = "lb" if kind == "corner" else "lc"
    cut, d1, d2, beta = AL[ia], D1[i1], D2[i2], BE[kind][ib]
    n = _normal(pk)
    for sb in (1, -1):
        dirn = _rot(n, sb * beta)
        B = (PC[pk][0] + d1 * dirn[0], PC[pk][1] + d1 * dirn[1])
        m = R + 0.004
        if not (m < B[0] < P.TX - m and m < B[1] < P.TY - m):
            continue
        u = (-dirn[0], -dirn[1])
        G = (B[0] - 2 * R * u[0], B[1] - 2 * R * u[1])
        for sa in (1, -1):
            w = _rot(u, sa * cut)
            C = (G[0] - d2 * w[0], G[1] - d2 * w[1])
            if not (m < C[0] < P.TX - m and m < C[1] < P.TY - m):
                continue
            if math.dist(C, B) < 2 * R + 0.002:
                continue
            st = {"balls": {"cue": {"x": C[0], "y": C[1], "pocketed": False},
                            "1": {"x": B[0], "y": B[1], "pocketed": False}}}
            return st, pk, G
    return None


def _obj_in(st, stroke, pk):
    """Table entries count the object ball dropping in the called pocket,
    scratch or not ("the percentage ... the ball is successfully pocketed" [PP])."""
    shot = P.simulate(st, stroke)
    if shot is None:
        return False
    from pool_env import pocket_map
    return pocket_map(shot).get("1") == pk


def _entry(job):
    """One table entry [PP 3.1.5-6]: minimum speed (noiseless bisection) and
    the success probability under the noise model. [dev] The aim is the
    centre of the potting window (aim_direct), and p is the best of a few
    speeds (vmin + 0.3, 2.0, 3.2 m/s), since CueCard plays several."""
    kind, ia, i1, i2, ib, s, seed = job
    noise_models.apply_from_env()
    g = _table_geom(kind, ia, i1, i2, ib)
    if g is None:
        return job, np.nan, np.nan
    st, pk, G = g
    ad = aim_direct(st, "1", pk, 2.5) or aim_direct(st, "1", pk, 5.0)
    if ad is None:
        return job, 0.0, 6.0
    phi_r, tgt, off = ad
    lo, hi = 0.3, 6.0
    if not _obj_in(st, (phi_r, 6.0, 0.0, 0.0), pk) and aim_direct(st, "1", pk, 6.0) is None:
        hi = 6.0
    for _ in range(8):
        mid = (lo + hi) / 2
        if aim_direct(st, "1", pk, mid) is not None:
            hi = mid
        else:
            lo = mid
    vmin = hi
    best = 0.0
    rng = random.Random(seed)
    for v in sorted({min(VMAX, vmin + 0.3), 2.0, 3.2}):
        if v < vmin:
            continue
        a2 = aim_direct(st, "1", pk, v)
        if a2 is None:
            continue
        made = sum(_obj_in(st, P.noisy((a2[0], v, 0.0, 0.0), rng, P.DEFAULT), pk) for _ in range(s))
        best = max(best, made / s)
    return job, best, vmin


def _fill(T):
    """Entries whose layout doesn't fit take the nearest valid neighbour."""
    T = T.copy()
    for _ in range(20):
        bad = np.argwhere(np.isnan(T))
        if not len(bad):
            break
        for ix in bad:
            vals = []
            for d in range(T.ndim):
                for s in (-1, 1):
                    j = list(ix)
                    j[d] += s
                    if 0 <= j[d] < T.shape[d] and not np.isnan(T[tuple(j)]):
                        vals.append(T[tuple(j)])
            if vals:
                T[tuple(ix)] = np.mean(vals)
    return np.nan_to_num(T, nan=0.0)


def _break_eval(job):
    x, y, v, b, dphi, seeds = job
    noise_models.apply_from_env()
    import game
    keep = spread = scratch = 0
    for s in seeds:
        st = game.rack_state(s)
        st = P.with_cue(st, (x, y))
        apex = min(P.live(st), key=lambda k: st["balls"][k]["y"])
        A = (st["balls"][apex]["x"], st["balls"][apex]["y"])
        cx = sum(st["balls"][k]["x"] for k in P.live(st)) / 15
        cy = sum(st["balls"][k]["y"] for k in P.live(st)) / 15
        rng = random.Random(s * 31 + 7)
        stroke = (phi_towards((x, y), A) + dphi, v, 0.0, b)
        res = P.summarize(P.simulate(st, P.noisy(stroke, rng, P.DEFAULT)), st)
        j = P.adjudicate(st, {"me": None, "opp": None}, {"kind": "break"}, res)
        keep += j["result"] == "continue"
        scratch += j["result"] == "foul"
        post = j["state"]["balls"]
        spread += sum(1 for k in P.live(st) if post[k].get("pocketed") or
                      math.hypot(post[k]["x"] - cx, post[k]["y"] - cy) > 0.35) / 15
    n = len(seeds)
    return job[:5], keep / n, spread / n, scratch / n


def build(workers="8"):
    import multiprocessing as mp
    os.makedirs(DATA, exist_ok=True)
    name = noise_name()
    t0 = time.time()
    out = {}
    jobs = []
    for kind in ("corner", "side"):
        for ia in range(len(AL)):
            for i1 in range(len(D1)):
                for i2 in range(len(D2)):
                    for ib in range(len(BE[kind])):
                        jobs.append((kind, ia, i1, i2, ib, TABLE_S,
                                     zlib.crc32(f"{kind}{ia}{i1}{i2}{ib}".encode())))
    with mp.get_context("spawn").Pool(int(workers)) as pool:
        res = pool.map(_entry, jobs, chunksize=16)
    for kind in ("corner", "side"):
        shp = (len(AL), len(D1), len(D2), len(BE[kind]))
        Pt, Vt = np.full(shp, np.nan), np.full(shp, np.nan)
        for job, p, v in res:
            if job[0] == kind:
                Pt[job[1:5]] = p
                Vt[job[1:5]] = v
        out["p_" + kind] = _fill(Pt)
        out["v_" + kind] = _fill(Vt)
        print(f"{kind}: {np.isnan(Pt).sum()} of {Pt.size} entries filled from neighbours; "
              f"mean p {np.nanmean(Pt):.3f}", flush=True)
    np.savez(os.path.join(DATA, f"{name}_tables.npz"), **out)
    print(f"tables [{name}] {len(jobs)} entries x {TABLE_S} samples [{time.time() - t0:.0f}s]", flush=True)
    if os.environ.get("CC_TABLES_ONLY"):
        return
    # break: offline search over cue spot, speed, aim offset and draw [IJCAI 3.3]
    t0 = time.time()
    seeds = list(range(609000, 609040))
    cands = [(x, P.HEAD_STRING - 0.05, v, b, dphi)
             for x in (0.15, 0.3, P.TX / 2, 0.69, 0.84)
             for v in (7.0, 9.0, 11.0, 12.0)
             for b in (-0.3, 0.0, 0.3)
             for dphi in (-0.6, 0.0, 0.6)]
    with mp.get_context("spawn").Pool(int(workers)) as pool:
        r1 = pool.map(_break_eval, [c + (seeds,) for c in cands], chunksize=2)
    score = lambda r: r[1] + 0.2 * r[2]     # noqa: E731 [AIM: keep the turn weighs more than spread]
    top = sorted(r1, key=lambda r: -score(r))[:8]
    seeds2 = list(range(609100, 609300))
    with mp.get_context("spawn").Pool(int(workers)) as pool:
        r2 = pool.map(_break_eval, [t[0] + (seeds2,) for t in top], chunksize=1)
    best = max(r2, key=score)
    x, y, v, b, dphi = best[0]
    rec = {"best": {"x": x, "y": y, "v": v, "a": 0.0, "b": b, "dphi": dphi, "keep": best[1],
                    "spread": best[2], "scratch": best[3], "score": score(best)},
           "refined": [{"cand": list(r[0]), "keep": r[1], "spread": r[2], "scratch": r[3]} for r in r2]}
    with open(os.path.join(DATA, f"{name}_break.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"break [{name}]: {rec['best']} [{time.time() - t0:.0f}s]", flush=True)


# ================================================================ sanity

def sanity():
    from rl import encode as E
    cc = CueCard("cc", seed=1)
    T = cc.T
    # 1. simple positions: a straight-in hanger, a 30-degree cut, a long shot
    cases = [((0.5, 0.5), (0.5, 1.2), "1"), ((0.3, 0.6), (0.6, 1.5), "1"), ((0.5, 0.2), (0.25, 1.6), "1")]
    for C, B, b in cases:
        st = {"balls": {"cue": {"x": C[0], "y": C[1], "pocketed": False},
                        b: {"x": B[0], "y": B[1], "pocketed": False},
                        "9": {"x": 0.8, "y": 0.3, "pocketed": False},
                        "8": {"x": 0.9, "y": 1.0, "pocketed": False}}}
        turn = E.make_turn(st, "shoot", P.SOLIDS, P.STRIPES, 5)
        t = time.process_time()
        w = cc.act(turn)
        dt = time.process_time() - t
        made = 0
        rng = random.Random(3)
        call = {"kind": "pot", "ball": w["call"]["ball"], "pocket": w["call"]["pocket"],
                "first": w["call"]["ball"]} if w["call"] else {"kind": "safe"}
        stroke = (w["stroke"]["phi"], w["stroke"]["speed"], w["stroke"]["side"], w["stroke"]["vertical"])
        for _ in range(50):
            j = cc._sample(st, {"me": P.SOLIDS, "opp": P.STRIPES}, call, stroke, rng)
            made += j["result"] in ("continue", "win")
        ss = straight_shots(st, P.SOLIDS, T=T)
        print(f"case cue {C} ball {B}: call {w['call']} speed {stroke[1]:.2f} spin ({stroke[2]:.2f},{stroke[3]:.2f}) "
              f"made {made}/50, table p {ss[0][0] if ss else None:.2f}, {dt:.2f}s", flush=True)
    print("stats", cc.stats)


if __name__ == "__main__":
    noise_models.apply_from_env()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    {"build": build, "sanity": sanity}[cmd](*sys.argv[2:])
