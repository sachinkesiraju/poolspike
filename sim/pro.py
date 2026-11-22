"""Pro player: EV full's search, spent differently, plus two things it skips.

1. Safeties. EV full plays house rules: it never plays safe while any pot
   exists. Pro weighs deliberate safeties against every pot (Config
   safeties=True), so a thin pot that leaves the opponent a run loses to a
   hit that leaves them nothing. The analytic leave model overrates
   safeties (the opponent potted next after ~78% of them in testing), so a
   chosen safety is checked: a real lite search from the opponent's side
   on SAFE_LEAVES of its noisy leaves. If their best pot averages above
   OPP_MAX the safety is dropped for the best pot.
2. Clock. EV full spends the same search on every shot. Pro answers easy
   shots (lite search finds a pot it makes ≥ EASY, not the 8) with that
   lite answer, and gives everything else a search wider than full (BIG).
   Over a game the CPU is about the same; it goes where the choice is hard.
3. Break book. tools/break_book.py scores hundreds of break recipes over
   many racks offline; pro re-ranks the book's best few on the actual rack.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import replace

import policy as P

EASY = 0.90
OPP_MAX = 0.50   # a safety must leave the opponent's best pot below this
SAFE_LEAVES = 3  # leaves of a safety that get a real opponent search
BOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "break_book.json")
BREAK_N = 16  # noisy samples per book recipe on the actual rack
NOISE = ("sigma_phi", "sigma_v", "sigma_ab", "speed_gain", "english_gain")


def stages(cfg: P.Config) -> tuple[P.Config, P.Config]:
    """(quick, big) searches, both weighing safeties, with cfg's noise."""
    quick = replace(P.LITE, name="pro-lite", safeties=True,
                    **{k: getattr(cfg, k) for k in NOISE})
    big = replace(cfg, name="pro-big", safeties=True, shortlist=16,
                  finalists=8, n_pot_mc=12, n_safe_mc=10)
    return quick, big


def easy(call: dict) -> bool:
    return (call["kind"] == "pot" and call.get("ball") != "8"
            and call.get("p_make", 0.0) >= EASY)


def answers(state: dict, ctx: dict, cfg: P.Config, seed: int):
    """Anytime: ("pro-lite", call) first; ("pro-big", call) only when the
    shot isn't easy."""
    quick, big = stages(cfg)
    call = checked(state, ctx, P.choose(state, ctx, quick, seed, keep=4),
                   quick, seed)
    yield "pro-lite", call
    if not easy(call):
        yield "pro-big", checked(
            state, ctx, P.choose(state, ctx, big, seed, keep=4), quick, seed)


def _strip(call: dict, **extra) -> dict:
    return {**{k: v for k, v in call.items() if k != "candidates"}, **extra}


def opp_chance(state: dict, ctx: dict, c: dict, quick: P.Config,
               seed: int) -> float:
    """Mean best-pot odds the opponent gets after safety `c`'s leaves."""
    call = {"kind": "safe", "ball": c.get("ball")}
    ps = []
    for i, r in enumerate(c.get("_mc", [])[:SAFE_LEAVES]):
        res = P.summarize(P.simulate(state, tuple(r["stroke"])), state)
        j = P.adjudicate(state, ctx, call, res)
        if j["result"] != "turnover":
            ps.append(1.0)  # a foul hands over ball in hand
            continue
        opp = {"me": j["opp"], "opp": j["me"]}
        nxt = P.choose(j["state"], opp, replace(quick, safeties=False),
                       seed * 31 + i)
        ps.append(nxt.get("p_make", 0.0) if nxt["kind"] == "pot" else 0.0)
    return sum(ps) / len(ps) if ps else 1.0


def checked(state: dict, ctx: dict, call: dict, quick: P.Config,
            seed: int) -> dict:
    """Drop a voluntary safety whose real leave is still makeable."""
    cands = call.get("candidates") or []
    pots = [c for c in cands if c["kind"] == "pot"]
    if call["kind"] != "safe" or not pots or not cands:
        return _strip(call)
    p = opp_chance(state, ctx, cands[0], quick, seed)
    if p <= OPP_MAX:
        return _strip(call, opp_check=round(p, 3))
    best = {k: v for k, v in pots[0].items() if not k.startswith("_")}
    best["plan"] = P.next_shot_plan(state, ctx, best, quick)
    return _strip(call, **best, opp_check=round(p, 3), safety_dropped=True)


def choose(state: dict, ctx: dict, cfg: P.Config, seed: int) -> dict:
    out = None
    for _, out in answers(state, ctx, cfg, seed):
        pass
    return out


def choose_bih(state: dict, ctx: dict, cfg: P.Config, seed: int,
               kitchen: bool = False) -> dict:
    return P.choose_bih(state, ctx, replace(cfg, safeties=True), seed,
                        kitchen=kitchen)


def _book() -> list[dict]:
    try:
        return json.load(open(BOOK))["top"]
    except (OSError, ValueError, KeyError):
        return []


def choose_break(state: dict, cfg: P.Config, seed: int) -> dict:
    book = _book()
    if not book:
        return P.choose_break(state, cfg, seed)
    rng = random.Random(seed)
    apex = min(P.live(state), key=lambda b: state["balls"][b]["y"])
    A = (state["balls"][apex]["x"], state["balls"][apex]["y"])
    ctx = {"me": None, "opp": None}
    opts, jobs, owners = [], [], []
    for i, r in enumerate(book):
        C = (r["x"], P.HEAD_STRING - 0.04)
        o = {"kind": "break", "cue_place": C, "apex": apex,
             "stroke": (P.phi_towards(C, A), r["v"], r["a"], r["b"])}
        opts.append(o)
        st = P.with_cue(state, C)
        for _ in range(BREAK_N):
            jobs.append((st, ctx, {"kind": "break"},
                         P.noisy(o["stroke"], rng, cfg), cfg))
            owners.append(i)
    res = P.pmap(P._task_eval, jobs)
    for i, o in enumerate(opts):
        rows = [r for j, r in zip(owners, res) if j == i]
        o["ev"] = sum(r["value"] for r in rows) / len(rows)
        o["p_continue"] = sum(r["result"] == "continue" for r in rows) / len(rows)
        o["p_scratch"] = sum(r["result"] == "foul" for r in rows) / len(rows)
    return max(opts, key=lambda o: o["ev"])
