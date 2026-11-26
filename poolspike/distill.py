"""Student nets distilled from Poolspike.

  python distill.py build DATADIR OUT.npz [B]     features + labels from teacher logs (B = with 1 noiseless sim)
  python distill.py train DATA.npz OUT.npz         fit the student
  python distill.py report DATA.npz NET.npz        holdout accuracy

Student = candidate scorer. Candidates come from cheap geometry (Poolspike's own lane
order and ball-in-hand spots, its 13/5-stroke grids); aim = ghost-ball angle +
a learned correction (no window search). Per candidate it predicts:
  feas  : does this stroke pot at all (Poolspike's window search found a window)
  dphi  : aim correction, degrees (Poolspike's window centre / refined aim - ghost-ball angle)
  s1    : Poolspike's first-stage score for it
  pick  : Poolspike's final choice (listwise softmax within a decision)
Modes at play (DS_MODE):
  A  no simulation at all
  B  one noiseless simulation per candidate adds leave features (Poolspike's base_features)
  C  B's top DS_K candidates are checked Poolspike's way (window, refine, 28 noisy sims, Poolspike's scorer)
"""
import glob
import gzip
import math
import os
import pickle
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shotnet as SN  # noqa: E402
import poolspike_player as N5  # noqa: E402
import numpy as np  # noqa: E402
from shotnet import E, P  # noqa: E402

GRIDS = N5.GRID + N5.BIH_GRID
POST = ("value_nominal", "scratch", "win", "made", "cue_central", "opp_odds", "next_odds", "next_open")
MAX_LANES = 8


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def gvec(st, me, opp, bih, kitchen):
    import judge_net
    e = judge_net.encode(st, me, opp, bih, kitchen)
    return None if e is None else e


def cand_feats(st, b, pk, gi, bih, q, lane_rank, G, Bs, cat, order):
    """Analytic features of one candidate (st already has the cue placed)."""
    v, a, s = GRIDS[gi]
    c = P.cue_xy(st)
    B = st["balls"][b]
    hole = P.POCKETS[pk]
    g = SN.ghost_ball_aim((B["x"], B["y"]), hole)
    u1 = (g[0] - c[0], g[1] - c[1])
    u2 = (hole[0] - B["x"], hole[1] - B["y"])
    d1, d2 = math.hypot(*u1) or 1e-6, math.hypot(*u2) or 1e-6
    cr = (u1[0] * u2[1] - u1[1] * u2[0]) / (d1 * d2)
    dt = (u1[0] * u2[0] + u1[1] * u2[1]) / (d1 * d2)
    th = math.degrees(math.atan2(cr, dt))          # signed cut angle
    side = pk in ("lc", "rc")
    oh = np.zeros(len(GRIDS), np.float32)
    oh[gi] = 1
    i = order.get(b)
    brow = Bs[i] if i is not None else np.zeros(Bs.shape[1], np.float32)
    f = np.r_[q, th / 90, math.sin(math.radians(th)), math.cos(math.radians(th)), d1 / 1.27, d2 / 1.27,
              v / 3, a, s, a * math.sin(math.radians(th)), float(side), float(bih), lane_rank / 8,
              (c[0] / P.TX) * 2 - 1, (c[1] / P.TY) * 2 - 1, oh, brow, G]
    return f.astype(np.float32), th


def post_feats(st, ctx, b, pk, gi, phi, q):
    v, a, s = GRIDS[gi]
    call = {"kind": "pot", "ball": b, "pocket": pk, "first": b}
    f = N5.base_features(st, ctx, call, (phi, v, a, s), q, 0.0, b, pk)
    return np.array([f[N5.F[k]] for k in POST], np.float32)


def position(turn_or_rec):
    """(st0, me, opp, ctx, G, Bs, cat, order) for the mover."""
    t = turn_or_rec
    if "turn" in t:
        t = t["turn"]
    st0 = E.to_state(t)
    me = E.GROUPS[t.get("my_group")]
    opp = None if me is None else (P.STRIPES if me == P.SOLIDS else P.SOLIDS)
    ph = t["phase"]
    e = gvec(st0, me, opp, ph.startswith("ball_in_hand"), ph == "ball_in_hand_kitchen")
    if e is None:
        return None
    G, Bs, cat = e
    lv = sorted(b for b in P.live(st0) if b != "cue")[:15]
    order = {b: i for i, b in enumerate(lv)}
    return st0, me, opp, {"me": me, "opp": opp}, G, Bs, cat, order


# ----------------------------------------------------------------- dataset
def _build_file(job):
    path, with_post = job
    N5.VNET = _vnet()
    rows = []
    for rec in pickle.load(gzip.open(path)):
        pos = position(rec)
        if pos is None or not rec["strokes"]:
            continue
        st0, me, opp, ctx, G, Bs, cat, order = pos
        lanes = [(b, pk) for _, b, pk in rec.get("lanes", [])]
        opts = rec["opts"]
        ch = rec["chosen"]
        for s in rec["strokes"]:
            st = st0 if s["C"] is None else P.with_cue(st0, s["C"])
            gi = s["gi"] + (len(N5.GRID) if s["bih"] else 0)
            rank = lanes.index((s["b"], s["pk"])) if (s["b"], s["pk"]) in lanes else 0
            x, _ = cand_feats(st, s["b"], s["pk"], gi, s["bih"], s["q"], rank, G, Bs, cat, order)
            o = None if s["opt"] is None else opts[s["opt"]]
            feas = o is not None
            aim = None
            if feas:
                aim = o.get("phi2") if o.get("kept") is not None else s["centre"]
            dphi = wrap(aim - s["base"]) if feas else 0.0
            if with_post:
                x = np.r_[x, post_feats(st, ctx, s["b"], s["pk"], gi, s["base"] + dphi if feas else s["base"], s["q"])]
            rows.append((x, feas, dphi, o["s1"] if feas else 0.0, float(feas and s["opt"] == ch),
                         o is not None and o.get("kept") is not None, rec["seed"], rec["shot_no"]))
    return rows


_V = None


def _vnet():
    global _V
    if _V is None:
        import judge_net
        z = np.load(os.environ.get("TEACHER_NET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nets", "poolspike.npz")))
        _V = judge_net.VNet2({k: z[k] for k in z.files})
    return _V


def build(datadir, out, with_post=""):
    import multiprocessing as mp
    files = sorted(glob.glob(os.path.join(datadir, "g*.pkl.gz")))
    t = time.time()
    with mp.get_context("spawn").Pool(int(os.environ.get("DS_WORKERS", 10))) as pool:
        parts = pool.map(_build_file, [(f, bool(with_post)) for f in files], chunksize=4)
    rows = [r for p in parts for r in p]
    X = np.stack([r[0] for r in rows])
    cols = list(zip(*[r[1:] for r in rows]))
    dec = np.array([r[6] * 1000 + r[7] for r in rows])
    _, gid = np.unique(dec, return_inverse=True)
    np.savez_compressed(out, X=X, feas=np.array(cols[0], np.float32), dphi=np.array(cols[1], np.float32),
                        s1=np.array(cols[2], np.float32), pick=np.array(cols[3], np.float32),
                        kept=np.array(cols[4], np.float32), seed=np.array(cols[5]), gid=gid)
    print(f"{len(files)} games, {gid.max() + 1} decisions, {len(X)} candidates, {X.shape[1]} inputs "
          f"[{time.time() - t:.0f}s]", flush=True)


# -------------------------------------------------------------------- model
class MLP:
    def __init__(self, d, hid=(128, 64), seed=0, z=None):
        if z is not None:
            self.p = {k[3:]: z[k] for k in z.files if k.startswith("ds_")}
            return
        r = np.random.default_rng(seed)
        dims = (d,) + hid
        self.p = {}
        for i in range(len(hid)):
            self.p[f"W{i}"] = r.normal(0, (2 / dims[i]) ** .5, (dims[i], dims[i + 1])).astype(np.float32)
            self.p[f"b{i}"] = np.zeros(dims[i + 1], np.float32)
        self.p["Wo"] = r.normal(0, .05, (hid[-1], 4)).astype(np.float32)
        self.p["bo"] = np.zeros(4, np.float32)
        self.p["mu"] = np.zeros(d, np.float32)
        self.p["sd"] = np.ones(d, np.float32)
        self.L = len(hid)

    @property
    def nl(self):
        return sum(1 for k in self.p if k.startswith("W") and k != "Wo")

    def forward(self, X, want=False):
        h = (X - self.p["mu"]) / self.p["sd"]
        hs = [h]
        for i in range(self.nl):
            h = np.maximum(0, h @ self.p[f"W{i}"] + self.p[f"b{i}"])
            hs.append(h)
        o = h @ self.p["Wo"] + self.p["bo"]          # pick, feas, dphi, s1
        return (o, hs) if want else o

    def backward(self, hs, do):
        g = {"Wo": hs[-1].T @ do, "bo": do.sum(0)}
        dh = do @ self.p["Wo"].T
        for i in range(self.nl - 1, -1, -1):
            dh = dh * (hs[i + 1] > 0)
            g[f"W{i}"] = hs[i].T @ dh
            g[f"b{i}"] = dh.sum(0)
            dh = dh @ self.p[f"W{i}"].T
        return g

    def save(self):
        return {"ds_" + k: v for k, v in self.p.items()}


def seg_softmax(l, gid, ng):
    m = np.full(ng, -1e9, np.float32)
    np.maximum.at(m, gid, l)
    e = np.exp(l - m[gid])
    z = np.zeros(ng, np.float32)
    np.add.at(z, gid, e)
    return e / z[gid]


DPHI_SCALE = 0.5


def losses(o, y, gid, ng, want_grad=False):
    feas, dphi, s1, pick = y
    n = len(feas)
    # pick: listwise over feasible candidates of each decision
    lp = np.where(feas > 0, o[:, 0], -30.0)
    p = seg_softmax(lp, gid, ng)
    has = np.zeros(ng, np.float32)
    np.add.at(has, gid, pick)
    w = has[gid]
    L_pick = -(pick * np.log(p + 1e-9)).sum() / max(1, has.sum())
    q = 1 / (1 + np.exp(-o[:, 1]))
    L_feas = float(np.mean(np.logaddexp(0, o[:, 1]) - feas * o[:, 1]))
    e_phi = (o[:, 2] - dphi / DPHI_SCALE) * feas
    L_phi = float((e_phi ** 2).sum() / max(1, feas.sum()))
    e_s1 = (o[:, 3] - s1 / 30.0) * feas
    L_s1 = float((e_s1 ** 2).sum() / max(1, feas.sum()))
    if not want_grad:
        return L_pick, L_feas, L_phi, L_s1
    d = np.zeros_like(o)
    d[:, 0] = (w * p - pick) * (feas > 0) / max(1, has.sum())
    d[:, 1] = (q - feas) / n
    d[:, 2] = 2 * e_phi / max(1, feas.sum())
    d[:, 3] = 2 * e_s1 / max(1, feas.sum())
    return (L_pick, L_feas, L_phi, L_s1), d


def split(Z):
    seed = Z["seed"]
    hold = (seed % 10) == 0
    return ~hold, hold


def load(data):
    zs = [np.load(f) for f in data.split(",")]
    Z, off = {}, 0
    for z in zs:
        for k in z.files:
            v = z[k] + off if k == "gid" else z[k]
            Z[k] = v if k not in Z else np.concatenate([Z[k], v])
        off = Z["gid"].max() + 1
    return Z


def train(data, out, epochs="40", lr="1e-3"):
    Z = load(data)
    X = Z["X"].astype(np.float32)
    tr, ho = split(Z)
    gid = Z["gid"]
    ys = [Z[k] for k in ("feas", "dphi", "s1", "pick")]
    net = MLP(X.shape[1])
    net.p["mu"] = X[tr].mean(0)
    net.p["sd"] = X[tr].std(0) + 1e-3
    m = {k: np.zeros_like(v) for k, v in net.p.items()}
    v2 = {k: np.zeros_like(v) for k, v in net.p.items()}
    r = np.random.default_rng(0)
    dec_tr = np.unique(gid[tr])
    order = np.argsort(gid, kind="stable")
    starts = np.searchsorted(gid[order], np.arange(gid.max() + 2))
    t, step, best = time.time(), 0, 1e9
    for ep in range(int(epochs)):
        r.shuffle(dec_tr)
        for i in range(0, len(dec_tr), 256):
            ds = dec_tr[i:i + 256]
            idx = np.concatenate([order[starts[d]:starts[d + 1]] for d in ds])
            _, g = np.unique(gid[idx], return_inverse=True)
            o, hs = net.forward(X[idx], True)
            _, do = losses(o, [y[idx] for y in ys], g, len(ds), True)
            grads = net.backward(hs, do.astype(np.float32))
            step += 1
            for k, gk in grads.items():
                m[k] = .9 * m[k] + .1 * gk
                v2[k] = .999 * v2[k] + .001 * gk * gk
                net.p[k] -= float(lr) * (m[k] / (1 - .9 ** step)) / (np.sqrt(v2[k] / (1 - .999 ** step)) + 1e-8)
        L = evaluate(net, X, ys, gid, ho)
        tot = sum(L[:4])
        flag = ""
        if tot < best:
            best, flag = tot, " *"
            np.savez(out, **net.save())
        print(f"ep {ep + 1}: holdout pick {L[0]:.3f} feas {L[1]:.3f} aim-mse {L[2]:.3f} s1 {L[3]:.4f} | "
              f"top1 {L[4]:.1%} feas-acc {L[5]:.1%} aim-err {L[6]:.2f} deg [{time.time() - t:.0f}s]{flag}", flush=True)


def evaluate(net, X, ys, gid, mask):
    _, g = np.unique(gid[mask], return_inverse=True)
    ng = g.max() + 1
    o = net.forward(X[mask])
    y = [v[mask] for v in ys]
    L = losses(o, y, g, ng)
    feas, dphi, _, pick = y
    lp = np.where(o[:, 1] > 0, o[:, 0], -1e9)    # play rule: predicted feasible, best pick logit
    best = np.full(ng, -np.inf)
    np.maximum.at(best, g, lp)
    hit = np.zeros(ng)
    np.add.at(hit, g, (lp == best[g]) & (pick > 0))
    has = np.zeros(ng)
    np.add.at(has, g, pick)
    top1 = hit[has > 0].mean()
    acc = ((o[:, 1] > 0) == (feas > 0)).mean()
    err = np.abs(o[:, 2] * DPHI_SCALE - dphi)[feas > 0].mean()
    return (*L, top1, acc, err)


def report(data, netf):
    Z = np.load(data)
    net = MLP(0, z=np.load(netf))
    tr, ho = split(Z)
    ys = [Z[k] for k in ("feas", "dphi", "s1", "pick")]
    print(evaluate(net, Z["X"].astype(np.float32), ys, Z["gid"], ho))


# ------------------------------------------------------------------ playing
class Student:
    """Mode A uses net A only. Modes B/C: net A gives feasibility + aim, then one
    noiseless sim per plausible candidate feeds net B, which picks."""
    def __init__(self, path_a, path_b=None, mode=None, k=None):
        self.a = MLP(0, z=np.load(path_a))
        self.b = MLP(0, z=np.load(path_b)) if path_b else None
        self.mode = mode or os.environ.get("DS_MODE", "A" if path_b is None else "B")
        self.k = int(k or os.environ.get("DS_K", 4))
        self.ns = int(os.environ.get("DS_NS", N5.NOISY0 + N5.NOISY1))   # noisy sims per checked candidate
        self.refine = os.environ.get("DS_REFINE", "1") == "1"
        self.teacher = N5.ShotNet5(os.environ.get("TEACHER_NET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nets", "poolspike.npz")))
        self.stats = {"student": 0, "fallback": 0, "checked": 0}

    def candidates(self, turn, pos):
        st0, me, opp, ctx, G, Bs, cat, order = pos
        ph = turn["phase"]
        legal = turn["legal_targets"]
        out = []
        if ph == "shoot":
            c = P.cue_xy(st0)
            lanes = sorted(((P.shot_prob(st0, c, b, pk), b, pk) for b in legal for pk in P.POCKETS), key=lambda x: -x[0])
            lanes = [x for x in lanes if x[0] > 0.02][:MAX_LANES]
            for rank, (q, b, pk) in enumerate(lanes):
                for gi in range(len(N5.GRID)):
                    out.append((st0, b, pk, gi, False, None, q, rank))
        else:
            for q, b, pk, C in N5.placements(st0, legal, ph == "ball_in_hand_kitchen"):
                st = P.with_cue(st0, C)
                for j in range(len(N5.BIH_GRID)):
                    out.append((st, b, pk, len(N5.GRID) + j, True, C, q, 0))
        return out

    def act(self, turn):
        if turn["phase"] == "break":
            return self.teacher.act(turn)
        pos = position(turn)
        cands = [] if pos is None else self.candidates(turn, pos)
        if not cands:
            self.stats["fallback"] += 1
            return self.teacher.act(turn)
        st0, me, opp, ctx, G, Bs, cat, order = pos
        N5.VNET = self.teacher.vnet
        X, bases = [], []
        for st, b, pk, gi, bih, C, q, rank in cands:
            x, _ = cand_feats(st, b, pk, gi, bih, q, rank, G, Bs, cat, order)
            B = st["balls"][b]
            hole = tuple(float(u) for u in P._T.pockets[pk].center[:2])
            X.append(x)
            bases.append(SN.phi_towards(P.cue_xy(st), SN.ghost_ball_aim((B["x"], B["y"]), hole)))
        X = np.stack(X)
        o = self.a.forward(X)
        phis = [bs + o[i, 2] * DPHI_SCALE for i, bs in enumerate(bases)]
        if self.mode != "A":
            live = [i for i in range(len(cands)) if o[i, 1] > -1.0]
            o = np.full((len(cands), 4), -1e9, np.float32)
            if live:
                Xp = np.stack([np.r_[X[i], post_feats(cands[i][0], ctx, cands[i][1], cands[i][2], cands[i][3], phis[i], cands[i][6])]
                               for i in live])
                o[live] = self.b.forward(Xp)
        ok = [i for i in range(len(cands)) if o[i, 1] > 0]
        if not ok:
            self.stats["fallback"] += 1
            return self.teacher.act(turn)
        ok.sort(key=lambda i: -o[i, 0])
        if self.mode == "C":
            w = self.check(turn, cands, phis, ok[:self.k], ctx)
            if w is not None:
                self.stats["checked"] += 1
                return w
        if self.mode == "W":                       # aim fix: exact window search on the top K, no noisy sims
            for i in ok[:self.k]:
                st, b, pk, gi, bih, C, q, rank = cands[i]
                v, a, s_ = GRIDS[gi]
                centre, _ = N5.window(st, b, pk, b, phis[i], SN.FINE * 3, 3, v, a, s_)
                if centre is not None:
                    self.stats["checked"] += 1
                    return self.wire(cands[i], centre)
        self.stats["student"] += 1
        i = ok[0]
        return self.wire(cands[i], phis[i])

    @staticmethod
    def wire(c, phi):
        st, b, pk, gi, bih, C, q, rank = c
        v, a, s = GRIDS[gi]
        w = {"call": {"ball": b, "pocket": pk}, "stroke": {"phi": float(phi % 360), "speed": v, "side": a, "vertical": s}}
        if C is not None:
            w["cue"] = {"x": float(C[0]), "y": float(C[1])}
        return w

    def check(self, turn, cands, phis, top, ctx):
        """Poolspike's second stage on the student's shortlist: exact window, refine, 28 noisy sims, teacher scorer."""
        opts = []
        for i in top:
            st, b, pk, gi, bih, C, q, rank = cands[i]
            v, a, s = GRIDS[gi]
            centre, width = N5.window(st, b, pk, b, phis[i], SN.FINE * 3, 3, v, a, s)
            if centre is None:
                continue
            stroke = (centre, v, a, s)
            call = {"kind": "pot", "ball": b, "pocket": pk, "first": b}
            f = N5.base_features(st, ctx, call, stroke, q, width, b, pk)
            if C is not None:
                f[N5.F["bih"]] = 1.0
            wire = self.wire(cands[i], centre)
            o = N5.Opt(st, ctx, call, stroke, f, wire, zlib.crc32(f"{turn['shot_no']}|{b}|{pk}|{gi}|{C}".encode()))
            if self.refine:
                o.refine()
            o.sample(self.ns)
            opts.append(o)
        if not opts:
            return None
        s = self.teacher.score(np.array([o.vector() for o in opts]))
        return opts[int(np.argmax(s))].wire


if __name__ == "__main__":
    {"build": build, "train": train, "report": report}[sys.argv[1]](*sys.argv[2:])
