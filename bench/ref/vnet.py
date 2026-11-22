"""A trained position evaluator: win probability for the player to move.

Same inputs the analytic race model looks at, plus a few it ignores; output is a
correction learned from real game results. Used in place of policy.value when
the net scores what a shot leaves. No game knowledge beyond these inputs.

  vnet.py gen  A.npz OPP LO HI WORKERS TAU OUT.npz   games, every position from both sides
  vnet.py fit  OUT.npz HIDDEN DATA...                train, print holdout vs analytic
  vnet.py pack NET_IN.npz VNET.npz NET_OUT.npz       attach evaluator to a shot net
"""
import math
import multiprocessing as mp
import os
import sys
import time
import zlib

os.environ["POOL_WORKERS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

INPUTS = ("logit_analytic", "p1", "p2", "p3", "n", "m", "n_is_1", "m_is_1", "diff", "open_me", "open_opp",
          "opp_p1", "bih", "table_open", "n_x_p1")
Q = len(INPUTS)


def inputs(state, me, opp, bih, kitchen):
    """Features of a position for the mover. Analytic value included as one input."""
    import policy as P
    cfg = P.DEFAULT
    tg = P.targets(state, me)
    if not tg:
        return None
    cxy = P.cue_xy(state)

    def odds(balls):
        return sorted((max(max(P.shot_prob(state, cxy, b, pk, cfg), P.bank_prob(state, cxy, b, pk, cfg))
                           for pk in P.POCKETS) for b in balls), reverse=True)
    if bih:
        ps = sorted((P.bih_prob(state, b, kitchen, cfg) for b in tg), reverse=True)
        ps = [max(ps[0], cfg.bih_floor)] + ps[1:]
    else:
        ps = odds(tg)
    otg = P.targets(state, opp)
    op = odds(otg)[0] if otg and not bih else 0.0
    n, m = P.n_left(state, me), P.n_left(state, opp)
    v = min(max(P.value(state, me, opp, bih=bih, kitchen=kitchen, cfg=cfg), .01), .99)
    p = (ps + [0.0, 0.0])[:3]
    return np.array([math.log(v / (1 - v)), p[0], p[1], p[2], n / 8, m / 8, float(n == 1), float(m == 1), (m - n) / 8,
                     P.openness(state, tg), P.openness(state, otg) if otg else 1.0, op, float(bih), float(me is None),
                     p[0] * n / 8])


class VNet:
    def __init__(self, z):
        self.mu, self.sd, self.W, self.b, self.o, self.a, self.c = (z[k] for k in (
            "v_mu", "v_sd", "v_W", "v_b", "v_o", "v_a", "v_c"))

    def logit(self, X):
        Z = (X - self.mu) / self.sd
        return float(self.a) * X[..., 0] + float(self.c) + np.tanh(Z @ self.W + self.b) @ self.o

    def __call__(self, state, me, opp, bih=False, kitchen=False):
        x = inputs(state, me, opp, bih, kitchen)
        if x is None:
            return 0.5
        return float(1 / (1 + np.exp(-self.logit(x))))


def play(job):
    import shotnet3 as N3
    from rl import encode as E
    from rl import env as ENV
    a, opp, seed, seat, tau = job
    rows = []

    def wrap(fn, who):
        def act(turn):
            ph = turn["phase"]
            if ph != "break":
                st = E.to_state(turn)
                x = inputs(st, E.GROUPS[turn["my_group"]], E.GROUPS[turn["opp_group"]],
                           ph.startswith("ball_in_hand"), ph == "ball_in_hand_kitchen")
                if x is not None:
                    rows.append((x, who, int(turn["shot_no"])))
            return fn(turn)
        return act
    me = N3.ShotNet3(a, tau=tau, seed=zlib.crc32(f"{seed}{seat}".encode()))
    other = N3.ShotNet3(opp[3:], tau=tau, seed=zlib.crc32(f"o{seed}{seat}".encode())).act if opp.startswith("n3:") \
        else ENV.make_policy(opp)
    env = ENV.PoolEnv(opponent=wrap(other, 1), agent_seat=seat)
    agent = wrap(me.act, 0)
    obs = env.reset(seed)
    while obs is not None:
        obs, _, _, _ = env.step(agent(obs))
    s = (env.outcome() + 1) / 2
    return {"X": np.array([r[0] for r in rows]), "y": np.array([s if r[1] == 0 else 1 - s for r in rows]),
            "shot": np.array([r[2] for r in rows]), "score": s, "seed": seed, "seat": seat}


def gen(a, opp, lo, hi, workers, tau, out):
    jobs = [(a, opp, s, seat, float(tau)) for s in range(int(lo), int(hi)) for seat in "AB"]
    t = time.time()
    with mp.get_context("spawn").Pool(int(workers)) as pool:
        rs = pool.map(play, jobs, chunksize=1)
    rs = [r for r in rs if len(r["X"])]
    np.savez(out, X=np.concatenate([r["X"] for r in rs]), y=np.concatenate([r["y"] for r in rs]),
             g=np.concatenate([np.full(len(r["X"]), i + int(lo) * 1000) for i, r in enumerate(rs)]))
    print(f"gen {a} vs {opp}: {len(rs)} games, score {np.mean([r['score'] for r in rs]):.3f}, "
          f"{sum(len(r['X']) for r in rs)} positions [{time.time() - t:.0f}s]", flush=True)


def fit(out, hidden, *data):
    D = [np.load(f) for f in data]
    X = np.concatenate([d["X"] for d in D])
    y = np.concatenate([d["y"] for d in D])
    g = np.concatenate([d["g"] + 10**8 * k for k, d in enumerate(D)])
    gs = np.unique(g)
    rng = np.random.default_rng(5)
    rng.shuffle(gs)
    n = max(1, len(gs) // 5)
    H, Dv = np.isin(g, gs[:n]), np.isin(g, gs[n:2 * n])
    T = ~(H | Dv)
    mu, sd = X[T].mean(0), X[T].std(0) + 1e-6
    k = int(hidden)
    W, b, o = rng.normal(0, .1, (Q, k)), np.zeros(k), np.zeros(k)
    a, c = np.array(1.0), np.array(0.0)

    def lg(X, W, b, o, a, c):
        return a * X[:, 0] + c + np.tanh((X - mu) / sd @ W + b) @ o

    def loss(l, y):
        return float(np.mean(np.logaddexp(0, l) - y * l))
    base = loss(X[H, 0], y[H])
    # scale-only baseline: how miscalibrated is the analytic value on its own?
    aa, cc = 1.0, 0.0
    for _ in range(50):
        l = aa * X[T, 0] + cc
        p = 1 / (1 + np.exp(-l))
        r = p * (1 - p) + 1e-6
        A = np.c_[X[T, 0], np.ones(T.sum())]
        s = np.linalg.solve((A.T * r) @ A, A.T @ (p - y[T]))
        aa, cc = aa - s[0], cc - s[1]
    params = [W, b, o, a, c]
    m1 = [np.zeros_like(q) for q in params]
    m2 = [np.zeros_like(q) for q in params]
    best, keep, idx, t, stale = 1e9, [q.copy() for q in params], np.flatnonzero(T), 0, 0
    for ep in range(400):
        rng.shuffle(idx)
        for s in range(0, len(idx), 256):
            B = idx[s:s + 256]
            Z = (X[B] - mu) / sd
            h = np.tanh(Z @ W + b)
            l = a * X[B, 0] + c + h @ o
            e = 1 / (1 + np.exp(-l)) - y[B]
            d = (1 - h ** 2) * o
            g_ = [Z.T @ (e[:, None] * d) / len(B) + 1e-3 * W, (e[:, None] * d).mean(0), h.T @ e / len(B) + 1e-3 * o,
                  np.array(np.mean(e * X[B, 0])), np.array(e.mean())]
            t += 1
            for i, q in enumerate(params):
                m1[i] = .9 * m1[i] + .1 * g_[i]
                m2[i] = .999 * m2[i] + .001 * g_[i] ** 2
                q -= 2e-3 * (m1[i] / (1 - .9 ** t)) / (np.sqrt(m2[i] / (1 - .999 ** t)) + 1e-8)
        dv = loss(lg(X[Dv], W, b, o, a, c), y[Dv])
        if dv < best:
            best, keep, stale = dv, [q.copy() for q in params], 0
        else:
            stale += 1
            if stale > 40:
                break
    W, b, o, a, c = keep
    L = lg(X[H], W, b, o, a, c)
    # bootstrap holdout improvement by game, so the gain carries an error bar
    hg = g[H]
    u = np.logaddexp(0, X[H, 0] * aa + cc) - y[H] * (X[H, 0] * aa + cc) - (np.logaddexp(0, L) - y[H] * L)
    per = np.array([u[hg == q].sum() for q in np.unique(hg)]) / max(1, len(u)) * len(np.unique(hg))
    print(f"positions {len(y)} games {len(gs)} epochs {ep + 1}; holdout loss: analytic {base:.4f}, "
          f"analytic rescaled (slope {aa:.2f}) {loss(X[H, 0] * aa + cc, y[H]):.4f}, net {loss(L, y[H]):.4f}; "
          f"gain over rescaled {per.mean():+.4f} +- {1.96 * per.std() / len(per) ** .5:.4f}; a={float(a):.2f}", flush=True)
    np.savez(out, v_mu=mu, v_sd=sd, v_W=W, v_b=b, v_o=o, v_a=a, v_c=c)


def pack(a, v, out):
    z = dict(np.load(a))
    z.update(dict(np.load(v)))
    np.savez(out, **z)


if __name__ == "__main__":
    {"gen": gen, "fit": fit, "pack": pack}[sys.argv[1]](*sys.argv[2:])
