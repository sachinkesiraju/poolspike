"""Position evaluator that sees the whole table, trained from play.

Each ball -> small encoder -> pooled separately for mine / theirs / the 8, joined
with whole-position numbers (including the analytic value as one input) -> win
probability for the player to move. Targets blend the game result with the net's
own next-position estimate (lambda return). Table mirror symmetry is augmented.

  judge_net.py gen NET OPP LO HI WORKERS TAU OUT.npz
  judge_net.py fit OUT.npz PREV.npz|- DATA...
  judge_net.py eval NET OPP LO HI WORKERS OUT.jsonl
  judge_net.py gradcheck
OPP: lite | ev | greedy | n5:<npz> | cc[:budget multiplier] (CueCard rebuild)
"""
import collections as C
import json
import multiprocessing as mp
import os
import statistics as S
import sys
import time
import zlib

os.environ["POOL_WORKERS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

D_BALL = 12
K = 16
HID = 32
LAM = 0.7
# indices that flip under a table mirror: ball (x, y, dx_cue, dy_cue), global (cue x, cue y)
MIRROR_X, MIRROR_Y = (0, 2), (1, 3)


def encode(state, me, opp, bih, kitchen):
    """(globals, balls[15, D_BALL], category[15]: 0 none, 1 mine, 2 theirs, 3 the 8), or None."""
    import policy as P
    import vnet as V1
    g = V1.inputs(state, me, opp, bih, kitchen)
    if g is None:
        return None
    cx, cy = P.cue_xy(state)
    hx, hy = P.TX / 2, P.TY / 2
    B = np.zeros((15, D_BALL), np.float32)
    cat = np.zeros(15, np.int8)
    lv = P.live(state)
    mine = set(P.group_of(me[0])) if me else set()
    theirs = set(P.group_of(opp[0])) if opp else set()
    pts = {b: (state["balls"][b]["x"], state["balls"][b]["y"]) for b in lv if b != "cue"}
    for i, (b, (x, y)) in enumerate(sorted(pts.items())[:15]):
        odds = [P.shot_prob(state, (cx, cy), b, pk) for pk in P.POCKETS]
        lanes = sum(P.pocket_lane_open(state, b, pk) for pk in P.POCKETS)
        near = min((np.hypot(x - u, y - w) for c, (u, w) in pts.items() if c != b), default=1.0)
        dx, dy = x - cx, y - cy
        B[i] = ((x - hx) / hx, (y - hy) / hy, dx / hx, dy / hy, np.hypot(dx, dy) / hy, max(odds), sum(o > .5 for o in odds) / 6,
                lanes / 6, min(near, .5) * 2, min(x, P.TX - x) / hx, min(y, P.TY - y) / hy, 1.0)
        cat[i] = 3 if b == "8" else 1 if b in mine else 2 if b in theirs else 0
    G = np.r_[g, (cx - hx) / hx, (cy - hy) / hy].astype(np.float32)
    return G, B, cat


class Net:
    """All parameters in one dict so the numeric gradient check can walk them."""
    def __init__(self, g_dim=None, seed=0, z=None):
        if z is not None:
            self.p = {k[3:]: z[k].astype(np.float64) for k in z.keys() if k.startswith("v2_")}
            return
        r = np.random.default_rng(seed)
        self.p = {"A": r.normal(0, .15, (D_BALL, K)), "c": np.zeros(K), "W": r.normal(0, .05, (g_dim + 3 * K, HID)),
                  "d": np.zeros(HID), "o": r.normal(0, .02, HID), "a": np.array(0.5), "c0": np.array(0.0),
                  "mu": np.zeros(g_dim), "sd": np.ones(g_dim)}

    def forward(self, G, B, cat, want=False):
        p = self.p
        h = np.tanh(B @ p["A"] + p["c"])
        pools = [(h * (cat == k)[..., None]).sum(-2) / 8 for k in (1, 2, 3)]
        Gn = (G - p["mu"]) / p["sd"]
        X = np.concatenate([Gn] + pools, -1)
        u = np.tanh(X @ p["W"] + p["d"])
        l = p["a"] * G[..., 0] + p["c0"] + u @ p["o"]
        return (l, h, X, u) if want else l

    def grads(self, G, B, cat, t):
        p = self.p
        l, h, X, u = self.forward(G, B, cat, True)
        r = 1 / (1 + np.exp(-l)) - t
        n = len(t)
        du = r[:, None] * p["o"]                         # dL/du
        s = (1 - u ** 2) * du
        dX = s @ p["W"].T                                # dL/dX
        g_dim = G.shape[1]
        dh = np.zeros_like(h)
        for j, k in enumerate((1, 2, 3)):
            dp = dX[:, g_dim + j * K: g_dim + (j + 1) * K]
            dh += (cat == k)[..., None] * dp[:, None, :] / 8
        sh = (1 - h ** 2) * dh
        return {"A": np.einsum("nbd,nbk->dk", B, sh) / n, "c": sh.sum((0, 1)) / n, "W": X.T @ s / n, "d": s.sum(0) / n,
                "o": u.T @ r / n, "a": np.array((r * G[:, 0]).mean()), "c0": np.array(r.mean())}

    def save(self):
        return {"v2_" + k: v for k, v in self.p.items()}


class VNet2:
    def __init__(self, z):
        self.net = Net(z=z)

    def __call__(self, state, me, opp, bih=False, kitchen=False):
        e = encode(state, me, opp, bih, kitchen)
        if e is None:
            return 0.5
        return float(1 / (1 + np.exp(-self.net.forward(e[0].astype(np.float64), e[1].astype(np.float64), e[2]))))


def loss(l, t):
    return float(np.mean(np.logaddexp(0, l) - t * l))


def bloss(net, G, B, cat, t, bs=4096):
    return float(np.concatenate([np.logaddexp(0, l) - t[i:i + bs] * l for i, l in (
        (i, net.forward(G[i:i + bs], B[i:i + bs], cat[i:i + bs])) for i in range(0, len(t), bs))]).mean())


def blogit(net, G, B, cat, bs=4096):
    return np.concatenate([net.forward(G[i:i + bs], B[i:i + bs], cat[i:i + bs]) for i in range(0, len(G), bs)])


def targets(pred, y, who, g):
    """Lambda return for each position, in that position's mover's perspective."""
    t = y.astype(np.float64).copy()
    for k in range(len(y) - 2, -1, -1):
        if g[k] == g[k + 1]:
            same = who[k] == who[k + 1]
            t[k] = (1 - LAM) * (pred[k + 1] if same else 1 - pred[k + 1]) + LAM * (t[k + 1] if same else 1 - t[k + 1])
    return t


def train(net, G, B, cat, t, T, Dv, yD, rng, epochs, lr=2e-3):
    """Adam with mirror augmentation; keeps the weights with the best dev loss on real game results."""
    p = net.p
    names = ["A", "c", "W", "d", "o", "a", "c0"]
    m1 = {k: np.zeros_like(p[k]) for k in names}
    m2 = {k: np.zeros_like(p[k]) for k in names}
    best, keep, stale, step, idx = 1e9, None, 0, 0, T.copy()
    gx, gy = G.shape[1] - 2, G.shape[1] - 1
    for ep in range(epochs):
        rng.shuffle(idx)
        for s in range(0, len(idx), 256):
            q = idx[s:s + 256]
            Gb, Bb = G[q].copy(), B[q].copy()
            sx, sy = rng.choice((-1., 1.), len(q)), rng.choice((-1., 1.), len(q))
            for j in MIRROR_X:
                Bb[:, :, j] *= sx[:, None]
            for j in MIRROR_Y:
                Bb[:, :, j] *= sy[:, None]
            Gb[:, gx] *= sx
            Gb[:, gy] *= sy
            gr = net.grads(Gb, Bb, cat[q], t[q])
            step += 1
            for k in names:
                g_ = gr[k] + (1e-4 * p[k] if k in ("A", "W", "o") else 0)
                m1[k] = .9 * m1[k] + .1 * g_
                m2[k] = .999 * m2[k] + .001 * g_ ** 2
                p[k] = p[k] - lr * (m1[k] / (1 - .9 ** step)) / (np.sqrt(m2[k] / (1 - .999 ** step)) + 1e-8)
        d = bloss(net, G[Dv], B[Dv], cat[Dv], yD)
        if d < best:
            best, keep, stale = d, {k: p[k].copy() for k in p}, 0
        else:
            stale += 1
            if stale >= 12:
                break
    net.p = keep
    return best, ep + 1


def fit(out, prev, *data):
    Ds = [np.load(f) for f in data]
    G = np.concatenate([d["G"] for d in Ds]).astype(np.float64)
    B = np.concatenate([d["B"] for d in Ds]).astype(np.float64)
    cat = np.concatenate([d["cat"] for d in Ds])
    y = np.concatenate([d["y"] for d in Ds]).astype(np.float64)
    who = np.concatenate([d["who"] for d in Ds])
    g = np.concatenate([d["g"] + 10**9 * k for k, d in enumerate(Ds)])
    gs = np.unique(g)
    rng = np.random.default_rng(5)
    r2 = np.random.default_rng(6)
    r2.shuffle(gs)
    n = max(1, len(gs) // 10)
    H = np.flatnonzero(np.isin(g, gs[:n]))
    Dv = np.flatnonzero(np.isin(g, gs[n:2 * n]))
    T = np.flatnonzero(~np.isin(g, gs[:2 * n]))
    if prev != "-":
        with np.load(prev) as z:
            if "v2_A" in z:
                net = Net(z=z)
            else:
                net = Net(G.shape[1], 1)
    else:
        net = Net(G.shape[1], 1)
    net.p["mu"][1:-2] = G[T][:, 1:-2].mean(0)   # logit and cue coordinates stay unshifted
    net.p["sd"][1:-2] = G[T][:, 1:-2].std(0) + 1e-6
    # baseline: analytic value rescaled on training games
    a, c = 1.0, 0.0
    x = G[T, 0]
    for _ in range(40):
        l = a * x + c
        p = 1 / (1 + np.exp(-l))
        A = np.c_[x, np.ones(len(x))]
        s = np.linalg.solve((A.T * (p * (1 - p) + 1e-6)) @ A, A.T @ (p - y[T]))
        a, c = a - s[0], c - s[1]
    t0 = time.time()
    pred = 1 / (1 + np.exp(-(a * G[:, 0] + c)))
    for it in range(3):                         # refresh the bootstrapped targets
        tg = targets(pred, y, who, g)
        d, ep = train(net, G, B, cat, tg, T, Dv, y[Dv], rng, 60)
        pred = 1 / (1 + np.exp(-blogit(net, G, B, cat)))
        print(f"  pass {it}: dev loss {d:.4f}, {ep} epochs", flush=True)
    base = a * G[H, 0] + c
    L = blogit(net, G[H], B[H], cat[H])
    u = (np.logaddexp(0, base) - y[H] * base) - (np.logaddexp(0, L) - y[H] * L)
    hg = g[H]
    per = np.array([u[hg == q].sum() for q in np.unique(hg)]) * len(np.unique(hg)) / len(u)
    print(f"fit: {len(y)} positions, {len(gs)} games [{time.time() - t0:.0f}s]; holdout loss analytic {loss(G[H, 0], y[H]):.4f}, "
          f"rescaled (slope {a:.2f}) {loss(base, y[H]):.4f}, net {loss(L, y[H]):.4f}; "
          f"gain {per.mean():+.4f} +- {1.96 * per.std() / len(per) ** .5:.4f}", flush=True)
    import poolspike_player as N5
    z = dict(N5.linear({"value_mean": 30.0}, g=0.02))
    z.update(net.save())
    np.savez(out, **z)


def opponent(spec, seed, seat, tau):
    import poolspike_player as N5
    from rl import env as ENV
    if spec == "cc" or spec.startswith("cc:"):      # CueCard rebuild (cuecard.py)
        import cuecard
        return cuecard.CueCard(spec, seed=zlib.crc32(f"o{seed}{seat}".encode())).act
    if spec.startswith("stu") and spec[4:5] == ":":
        import distill
        ps = spec[5:].split(",")
        return distill.Student(ps[0], ps[1] if len(ps) > 1 else None, mode=spec[3]).act
    if spec.startswith("n3:"):
        import shotnet3 as N3
        return N3.ShotNet3(spec[3:], tau=tau, seed=zlib.crc32(f"o{seed}{seat}".encode())).act
    if spec[:1] == "n" and spec[2:3] == ":" and spec[1] in "67":
        import importlib
        mod = importlib.import_module("shotnet" + spec[1])
        return getattr(mod, "ShotNet" + spec[1])(spec[3:], tau=tau, seed=zlib.crc32(f"o{seed}{seat}".encode())).act
    if spec.startswith("n5:"):
        return N5.ShotNet5(spec[3:], tau=tau, seed=zlib.crc32(f"o{seed}{seat}".encode())).act
    if spec in ("plus", "pattern", "safe", "pro"):
        os.environ["POOL_WORKERS"] = "1"
        return ENV.search_policy(spec)
    return ENV.make_policy(spec)


def play(job):
    try:
        return play_(job)
    except Exception as e:            # one broken game must not cost the whole batch
        print("game failed", job[2], job[3], repr(e)[:200], flush=True)
        return None


def play_(job):
    import poolspike_player as N5
    from rl import encode as E
    from rl import env as ENV
    a, opp, seed, seat, tau, log = job
    import noise_models                 # POOL_NOISE / POOL_RULESET; no-op when unset
    noise_models.apply_from_env()
    rows = []

    def wrap(fn, who):
        def act(turn):
            ph = turn["phase"]
            if log and ph != "break":
                e = encode(E.to_state(turn), E.GROUPS[turn["my_group"]], E.GROUPS[turn["opp_group"]],
                           ph.startswith("ball_in_hand"), ph == "ball_in_hand_kitchen")
                if e is not None:
                    rows.append((e, who))
            return fn(turn)
        return act
    if a == "cc" or a.startswith("cc:"):
        import cuecard
        me = cuecard.CueCard(a, seed=zlib.crc32(f"{seed}{seat}".encode()))
    elif a.startswith("stu") and a[4:5] == ":":
        import distill
        ps = a[5:].split(",")
        me = distill.Student(ps[0], ps[1] if len(ps) > 1 else None, mode=a[3])
    elif a[:1] == "n" and a[2:3] == ":" and a[1].isdigit():
        import importlib
        mod = importlib.import_module("shotnet" + a[1])
        me = getattr(mod, "ShotNet" + a[1])(a[3:], tau=tau, seed=zlib.crc32(f"{seed}{seat}".encode()))
    else:
        me = N5.ShotNet5(a, tau=tau, seed=zlib.crc32(f"{seed}{seat}".encode()))
    env = ENV.PoolEnv(opponent=wrap(opponent(opp, seed, seat, tau), 1), agent_seat=seat)
    agent = wrap(me.act, 0)
    t = time.process_time()
    obs = env.reset(seed)
    n = 0
    while obs is not None:
        obs, _, _, _ = env.step(agent(obs))
        n += 1
    s = (env.outcome() + 1) / 2
    side = {}
    for who, st in (("a", env.seat), ("b", "B" if env.seat == "A" else "A")):
        hs = [h for h in env.history if h["seat"] == st]
        pots = [h for h in hs if h["kind"] == "pot"]
        side[who] = {"shots": len(hs), "pots": len(pots), "made": sum(bool(h["made"]) for h in pots),
                     "fouls": sum(h["result"] == "foul" for h in hs),
                     "scratch": sum("scratch" in str(h["reason"]) for h in hs),
                     "invalid": sum(h["invalid"] is not None for h in hs)}
    return {"seed": seed, "seat": seat, "score": s, "end": env.m.end_reason, "sides": side, "n": n,
            "cpu": time.process_time() - t, "rows": rows}


def gen(a, opp, lo, hi, workers, tau, out):
    jobs = [(a, opp, s, seat, float(tau), True) for s in range(int(lo), int(hi)) for seat in "AB"]
    t = time.time()
    with mp.get_context("spawn").Pool(int(workers)) as pool:
        rs = [r for r in pool.map(play, jobs, chunksize=1) if r]
    G, B, cat, y, who, g = [], [], [], [], [], []
    for i, r in enumerate(rs):
        for (e, w) in r["rows"]:
            G.append(e[0])
            B.append(e[1])
            cat.append(e[2])
            y.append(r["score"] if w == 0 else 1 - r["score"])
            who.append(w)
            g.append(i + int(lo) * 1000)
    np.savez_compressed(out, G=np.array(G), B=np.array(B), cat=np.array(cat), y=np.array(y, np.float32),
                        who=np.array(who, np.int8), g=np.array(g))
    print(f"gen {os.path.basename(a)} vs {os.path.basename(opp)}: {len(rs)} games, score "
          f"{np.mean([r['score'] for r in rs]):.3f}, {len(y)} positions [{time.time() - t:.0f}s]", flush=True)


def eval_(a, opp, lo, hi, workers, out):
    jobs = [(a, opp, s, seat, 0.0, False) for s in range(int(lo), int(hi)) for seat in "AB"]
    t = time.time()
    with mp.get_context("spawn").Pool(int(workers)) as pool:
        rows = [r for r in pool.map(play, jobs, chunksize=1) if r]
    with open(out, "w") as f:
        for r in rows:
            r.pop("rows")
            f.write(json.dumps(r) + "\n")
    sc = [r["score"] for r in rows]
    print(f"{os.path.basename(a)} vs {os.path.basename(opp)}: {len(rows)} games, score {S.mean(sc):.3f} "
          f"+- {1.96 * S.pstdev(sc) / len(sc) ** .5:.3f} [{time.time() - t:.0f}s]")
    for w, name in (("a", "net"), ("b", "opp")):
        x = {k: sum(r["sides"][w][k] for r in rows) for k in rows[0]["sides"][w]}
        print(f"  {name}: make {x['made'] / max(1, x['pots']):.1%} ({x['made']}/{x['pots']}), fouls "
              f"{x['fouls'] / max(1, x['shots']):.1%}, scratches {x['scratch']}, invalid {x['invalid']}")
    print("  ends:", dict(C.Counter(("W " if r["score"] > .5 else "L ") + str(r["end"]) for r in rows).most_common(6)))
    print(f"  cpu per decision {sum(r['cpu'] for r in rows) / sum(r['n'] for r in rows) * 1000:.0f} ms (includes opponent)",
          flush=True)


def gradcheck():
    r = np.random.default_rng(0)
    n, gd = 6, 19
    G, B = r.normal(size=(n, gd)), r.normal(size=(n, 15, D_BALL))
    cat, t = r.integers(0, 4, (n, 15)), r.random(n)
    net = Net(gd, 3)
    an = net.grads(G, B, cat, t)
    worst = 0.0
    for k, g_ in an.items():
        p = net.p[k]
        it = np.nditer(p, flags=["multi_index"]) if p.ndim else [None]
        for _ in range(min(p.size, 12)):
            ix = tuple(r.integers(0, s) for s in p.shape)
            e = 1e-5
            if p.ndim:
                o = p[ix]
                p[ix] = o + e
                a = loss(net.forward(G, B, cat), t)
                p[ix] = o - e
                b = loss(net.forward(G, B, cat), t)
                p[ix] = o
                ana = g_[ix]
            else:
                net.p[k] = p + e
                a = loss(net.forward(G, B, cat), t)
                net.p[k] = p - e
                b = loss(net.forward(G, B, cat), t)
                net.p[k] = p
                ana = float(g_)
            worst = max(worst, abs((a - b) / (2 * e) - ana))
    print("worst gradient error", worst)
    assert worst < 1e-6


if __name__ == "__main__":
    {"gen": gen, "fit": fit, "eval": eval_, "gradcheck": gradcheck}[sys.argv[1]](*sys.argv[2:])
