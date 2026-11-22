"""Pattern player: the full EV search plus a second ply on its best pots.

EV full scores a pot as p_make·V(leave) + (1 − p_make)·V(miss), where
V(leave) is the analytic `policy.value`: the best next-shot odds from a
geometric model plus a race over balls left. It never plays the next shot,
so two leaves with the same straight-in odds look equal even when one runs
out and the other dead-ends after one more ball.

The pattern player keeps EV full's top pots (anything within `BAND` of the
leader), replays the made samples that search already drew (same noisy
strokes, deterministic physics), and from each leave runs a real, cheaper
search for the next shot. That search's EV — which itself prices the leave
after next — replaces V(leave). The pick is argmax of the two-ply EV, the
way a player chooses shape for the ball after next. The miss branch keeps
EV full's value: what the opponent faces is unchanged by looking deeper.

Breaks and ball in hand play the full search unchanged.
"""

from __future__ import annotations

import time
from dataclasses import replace

import policy as P

K = 4         # pots that get the second ply
LEAVES = 3    # made leaves searched per pot
BAND = 0.15   # ... if within this EV of the leader
MIN_P = 0.25  # ... and makeable enough for the leave to matter


def look_cfg(cfg: P.Config) -> P.Config:
    """The second ply's search: the lite budget, direct pots only unless
    nothing is pottable, refining only the leader. Same noise and value."""
    return replace(cfg, name="look", shortlist=4, finalists=2, n_pot_mc=4,
                   n_safe_mc=3, max_lanes=6, exec_grid="lite",
                   deep_below=0.0, refine_close=0.0)


def answers(state: dict, ctx: dict, cfg: P.Config, seed: int,
            deadline: float | None = None, limit: int | None = None):
    """Anytime: yield ("ev", call) as soon as the first ply is done, then
    ("pattern", call) after the second ply. `deadline` (time.monotonic())
    stops the second ply early; it then compares only the pots it reached,
    and the call's `looked` says how many (`limit` replays that cut)."""
    base = P.choose(state, ctx, cfg, seed, keep=K + 2)
    yield "ev", _strip(base)
    yield "pattern", lookahead(state, ctx, base, cfg, seed, deadline, limit)


def choose(state: dict, ctx: dict, cfg: P.Config, seed: int,
           deadline: float | None = None, limit: int | None = None) -> dict:
    out = None
    for _, out in answers(state, ctx, cfg, seed, deadline, limit):
        pass
    return out


def _strip(call: dict, **extra) -> dict:
    return {**{k: v for k, v in call.items() if k != "candidates"}, **extra}


def leave_value(state: dict, ctx: dict, c: dict, cfg: P.Config,
                seed: int) -> tuple[float, int]:
    """Mean two-ply value of a pot's made leaves: (value, leaves used)."""
    call = {k: c.get(k) for k in ("kind", "ball", "pocket", "first")}
    strokes = [r["stroke"] for r in c.get("_mc", []) if r["made"]][:LEAVES]
    look = look_cfg(cfg)
    vals = []
    for i, s in enumerate(strokes):
        res = P.summarize(P.simulate(state, tuple(s)), state)
        j = P.adjudicate(state, ctx, call, res)
        if j["result"] == "win":
            vals.append(1.0)
        elif j["result"] == "continue":
            nxt = P.choose(j["state"], {"me": j["me"], "opp": j["opp"]},
                           look, seed * 31 + i)
            vals.append(nxt["ev"])
    if not vals:  # no made sample to replay: keep the one-ply value
        return c["v_make"], 0
    return sum(vals) / len(vals), len(vals)


def lookahead(state: dict, ctx: dict, base: dict, cfg: P.Config,
              seed: int, deadline: float | None = None,
              limit: int | None = None) -> dict:
    cands = base.get("candidates") or []
    pots = [c for c in cands if c["kind"] == "pot"]
    if not pots or cands[0]["kind"] != "pot":
        return _strip(base, looked=0)  # nothing to pot, or a safety leads
    lead = max(c["ev"] for c in pots)
    pool = [c for c in pots if c["ev"] >= lead - BAND
            and c["p_make"] >= MIN_P][:K]
    if len(pool) < 2:  # one real option: a deeper look can't change it
        return _strip(base, looked=0)
    t = time.monotonic()
    rows = []
    for n, c in enumerate(pool):
        if limit is not None and len(rows) >= limit:
            break
        if deadline is not None and rows:
            per = (time.monotonic() - t) / len(rows)
            if time.monotonic() + per > deadline:
                break
        v2, used = leave_value(state, ctx, c, cfg, seed * 7 + n)
        ev2 = c["p_make"] * v2 + (1 - c["p_make"]) * c["v_miss"]
        rows.append((ev2, v2, used, c))
    if len(rows) < 2:
        return _strip(base, looked=len(rows))
    # same near-tie rule as the first ply: the more makeable shot
    top = max(r[0] for r in rows)
    ev2, v2, used, pick = max(rows, key=lambda r: (
        r[0] if top - r[0] > cfg.tie_eps else top, r[3]["p_make"], r[0]))
    best = {k: v for k, v in pick.items() if not k.startswith("_")}
    best.update(ev1=pick["ev"], ev=ev2, v_make1=pick["v_make"], v_make=v2,
                contest=base.get("contest", []),
                lanes_probed=base.get("lanes_probed"),
                lanes_found=base.get("lanes_found"),
                deep_ran=base.get("deep_ran"), n_pots=base.get("n_pots"),
                n_safes=base.get("n_safes"), best_pot=base.get("best_pot"),
                pattern=[{"label": P._label(c), "p_make": round(c["p_make"], 3),
                          "ev1": round(c["ev"], 3), "ev2": round(e, 3),
                          "v_make1": round(c["v_make"], 3),
                          "v_make2": round(v, 3), "leaves": u}
                         for e, v, u, c in rows],
                pattern_s=round(time.monotonic() - t, 1), looked=len(rows),
                changed=P._lane_key(pick) != P._lane_key(pool[0]))
    best["plan"] = P.next_shot_plan(state, ctx, best, cfg)
    return best
