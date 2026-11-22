"""Leave net: a learned next-shot probability for policy.value.

EV full judges every leave with p1, the best geometric next-shot odds.
On self-play positions p1 reads 0.86 where a real noisy search finds
0.76, and a p1 of 0.70–0.85 is really ~0.43 (tools/leave_check.py). The
net predicts the searched best-pot probability from the encoded table
plus p1 itself, so policy.value can price leaves the way search would.

  python leave_net.py train DATA_DIR OUT.npz [--epochs=60] [--hidden=128,64]
      [--rank=1.0]   add a pairwise loss over leaves of the same position

Choosing a leave only needs the right order among one position's leaves;
plain MSE learned the level but flattened that order. --rank adds a
logistic loss on pairs from the same root (needs the g array from
tools/leave_data.py), keeping MSE so the output stays a probability.

Pure numpy; predict() costs well under a millisecond.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

_NETS: dict[str, dict] = {}


def _forward(net: dict, x: np.ndarray) -> np.ndarray:
    h = (x - net["mu"]) / net["sd"]
    i = 0
    while f"W{i + 1}" in net:
        h = h @ net[f"W{i}"] + net[f"b{i}"]
        h = np.maximum(h, 0.0)
        i += 1
    z = h @ net[f"W{i}"] + net[f"b{i}"]
    return 1.0 / (1.0 + np.exp(-np.clip(z[..., 0], -30, 30)))


def load(path: str) -> dict:
    if path not in _NETS:
        d = np.load(path)
        _NETS[path] = {k: d[k] for k in d.files}
    return _NETS[path]


def predict(path: str, state: dict, me, opp, p1: float) -> float:
    from rl import encode as E
    x = E.encode_obs(E.make_turn(state, "shoot", me, opp, 10))
    x = np.append(x, np.float32(p1))
    return float(_forward(load(path), x[None])[0])


def _data(root: str):
    files = sorted(glob.glob(os.path.join(root, "chunk_*.npz")))
    parts = [np.load(f) for f in files]
    X = [np.hstack([d["X"], d["p1"][:, None]]) for d in parts]
    Y = [d["y"] for d in parts]
    G = []
    for d in parts:
        g = d["g"].copy() if "g" in d.files else np.full(len(d["y"]), -1)
        # each group's first row is the root itself, not a leave to choose
        first = np.r_[True, g[1:] != g[:-1]]
        g[first] = -1
        G.append(g)
    return X, Y, G


def _pairs(g: np.ndarray, y: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """(better, worse) row pairs from the same root position."""
    out = []
    for r in np.unique(g[g >= 0]):
        idx = np.flatnonzero(g == r)
        for a in idx:
            for b in idx:
                if y[a] > y[b] + margin:
                    out.append((a, b))
    return np.array(out, int).reshape(-1, 2)


def _rank_stats(net: dict, X: np.ndarray, y: np.ndarray, g: np.ndarray) -> dict:
    """Per root: does the net pick the best-labelled row, and its regret."""
    p = _forward(net, X)
    hit, reg = [], []
    for r in np.unique(g[g >= 0]):
        idx = np.flatnonzero(g == r)
        if len(idx) < 3:
            continue
        best = y[idx].max()
        reg.append(best - y[idx[np.argmax(p[idx])]])
        hit.append(reg[-1] < 0.02)
    return {"roots": len(hit), "picked_best": float(np.mean(hit)) if hit else 0.0,
            "regret": float(np.mean(reg)) if reg else 0.0}


def train(root: str, out: str, epochs: int = 60, hidden=(128, 64),
          lr: float = 1e-3, batch: int = 128, wd: float = 1e-4,
          seed: int = 0, rank: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    X, Y, G = _data(root)
    # hold out whole chunks (different positions), not rows of one chunk
    order = rng.permutation(len(X))
    n_hold = max(1, len(X) // 5)
    hold, tr = order[:n_hold], order[n_hold:]
    Xt = np.vstack([X[i] for i in tr]); Yt = np.concatenate([Y[i] for i in tr])
    Xh = np.vstack([X[i] for i in hold]); Yh = np.concatenate([Y[i] for i in hold])
    Gt = np.concatenate([G[i] for i in tr]); Gh = np.concatenate([G[i] for i in hold])
    PR = _pairs(Gt, Yt) if rank else np.zeros((0, 2), int)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    sizes = [Xt.shape[1], *hidden, 1]
    net = {"mu": mu, "sd": sd}
    for i in range(len(sizes) - 1):
        net[f"W{i}"] = rng.normal(0, np.sqrt(2 / sizes[i]),
                                  (sizes[i], sizes[i + 1])).astype(np.float64)
        net[f"b{i}"] = np.zeros(sizes[i + 1])
    keys = [k for k in net if k[0] in "Wb"]
    m = {k: np.zeros_like(net[k]) for k in keys}
    v = {k: np.zeros_like(net[k]) for k in keys}
    L = len(sizes) - 1
    step, best, best_net = 0, (1e9, -1), None

    def mse(Xs, Ys):
        return float(((_forward(net, Xs) - Ys) ** 2).mean())

    base = {"p1_mae": float(np.abs(Xh[:, -1] - Yh).mean()),
            "p1_mse": float(((Xh[:, -1] - Yh) ** 2).mean()),
            "mean_mse": float(((Yt.mean() - Yh) ** 2).mean())}
    print(f"train {len(Yt)} rows, holdout {len(Yh)}; holdout mse of p1 "
          f"{base['p1_mse']:.4f}, of the mean {base['mean_mse']:.4f}")
    for ep in range(epochs):
        perm = rng.permutation(len(Yt))
        n_b = max(1, len(perm) // batch)
        pp = rng.permutation(len(PR)) if len(PR) else None
        for bi in range(n_b):
            idx = perm[bi * batch:(bi + 1) * batch]
            rows, dz = [idx], []
            # MSE on sigmoid output: dL/dz
            p = None
            if len(PR):
                pr = PR[pp[(bi * batch + np.arange(batch)) % len(PR)]]
                rows += [pr[:, 0], pr[:, 1]]
            Xb = np.concatenate([Xt[r] for r in rows])
            hs = [(Xb - mu) / sd]
            for i in range(L - 1):
                hs.append(np.maximum(hs[-1] @ net[f"W{i}"] + net[f"b{i}"], 0))
            z = (hs[-1] @ net[f"W{L - 1}"] + net[f"b{L - 1}"])[:, 0]
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            n0 = len(idx)
            gz = np.zeros_like(z)
            gz[:n0] = 2 * (p[:n0] - Yt[idx]) * p[:n0] * (1 - p[:n0]) / n0
            if len(PR):
                k = len(pr)
                d = z[n0:n0 + k] - z[n0 + k:]
                sg = -rank / (1 + np.exp(np.clip(d, -30, 30))) / k
                gz[n0:n0 + k] += sg
                gz[n0 + k:] -= sg
            g = gz[:, None]
            grads = {}
            for i in range(L - 1, -1, -1):
                grads[f"W{i}"] = hs[i].T @ g + wd * net[f"W{i}"]
                grads[f"b{i}"] = g.sum(0)
                if i:
                    g = (g @ net[f"W{i}"].T) * (hs[i] > 0)
            step += 1
            for k_ in keys:
                m[k_] = 0.9 * m[k_] + 0.1 * grads[k_]
                v[k_] = 0.999 * v[k_] + 0.001 * grads[k_] ** 2
                mh = m[k_] / (1 - 0.9 ** step)
                vh = v[k_] / (1 - 0.999 ** step)
                net[k_] -= lr * mh / (np.sqrt(vh) + 1e-8)
        h = mse(Xh, Yh)
        rs = _rank_stats(net, Xh, Yh, Gh) if rank else None
        crit = rs["regret"] + 0.1 * h if rank else h
        if crit < best[0]:
            best, best_net = (crit, ep), {k: x.copy() for k, x in net.items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d} train mse {mse(Xt, Yt):.4f} holdout mse "
                  f"{h:.4f} {rs or ''} (best {best[0]:.4f} @ {best[1]})", flush=True)
    np.savez(out, **best_net)
    pred = _forward(best_net, Xh)
    res = {**base, "net_mse": float(((pred - Yh) ** 2).mean()),
           "net_mae": float(np.abs(pred - Yh).mean()),
           "epoch": best[1], "rows": len(Yt), "holdout": len(Yh)}
    if (Gh >= 0).any():
        res["net_rank"] = _rank_stats(best_net, Xh, Yh, Gh)
        res["p1_rank"] = _rank_stats({"mu": 0, "sd": 1, "W0": np.eye(Xh.shape[1])[:, -1:] * 1.0,
                                       "b0": np.zeros(1)}, Xh, Yh, Gh)
    print(res)
    return res


def main():
    if len(sys.argv) < 4 or sys.argv[1] != "train":
        print(__doc__)
        return 2
    opts = dict((a[2:].split("=", 1) + [""])[:2] for a in sys.argv[4:]
                if a.startswith("--"))
    hidden = tuple(int(h) for h in (opts.get("hidden") or "128,64").split(","))
    train(sys.argv[2], sys.argv[3], epochs=int(opts.get("epochs") or 60),
          hidden=hidden, rank=float(opts.get("rank") or 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
