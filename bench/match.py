"""Pool8 benchmark match runner: agent A vs agent B on fixed seeds, both seats.

  python -B match.py A B --seeds LO HI [--workers W] [--noise ours|icga2008]
                     [--rules ours|icga] [--out games.jsonl]

Each seed s in [LO, HI) is played twice: A breaks (seat "A") and B breaks
(seat "B"), with the same rack and the same table-noise stream, so the luck of
the rack and the dice is shared. Prints A's score (win = 1, draw = 0.5) with a
95% normal half-width, both sides' make rate (called pots made / attempted),
foul rate per shot, and how games ended.

Agent specs (reference_agents.py): ev | lite | greedy | cc[:mult] | poolspike (alias rla_late) | oneply | n5:<weights.npz>
| py:<path/to/file.py>:<factory>. A factory is called as factory(seed=int,
seat=str) and returns either a callable turn -> action or an object with
.act(turn). See example_agent.py for the turn/action format.
"""
from __future__ import annotations

import argparse
import collections as C
import json
import multiprocessing as mp
import os
import statistics as S
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("POOL_COWORLD", os.path.join(HERE, "pool8"))
os.environ["POOL_WORKERS"] = "1"          # parallelism is one game per process
for p in (os.path.join(HERE, "ref"), os.environ["POOL_COWORLD"], HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def play(job):
    a, b, seed, seat, noise, rules = job
    os.environ["POOL_NOISE"], os.environ["POOL_RULESET"] = noise, rules
    import noise_models
    noise_models.apply(noise, rules)
    import reference_agents
    from rl import env as ENV
    import baseline
    from rl import encode as E
    try:
        A = reference_agents.make(a, seed, seat, role="a")
        B = reference_agents.make(b, seed, seat, role="b")
        env = ENV.PoolEnv(opponent=B, agent_seat=seat)
        errors = 0
        t = time.process_time()
        obs = env.reset(seed)
        n = 0
        while obs is not None:
            try:
                act = A(obs)
            except Exception:             # agent A crashed: the server's fallback shot
                errors += 1
                act = baseline.action(E.to_state(obs), obs["phase"], E.GROUPS[obs["my_group"]])
            obs, _, _, _ = env.step(act)
            n += 1
    except Exception as e:
        print("game failed", seed, seat, repr(e)[:300], traceback.format_exc()[-400:], flush=True)
        return None
    side = {}
    for who, st in (("a", env.seat), ("b", "B" if env.seat == "A" else "A")):
        hs = [h for h in env.history if h["seat"] == st]
        pots = [h for h in hs if h["kind"] == "pot"]
        side[who] = {"shots": len(hs), "pots": len(pots), "made": sum(bool(h["made"]) for h in pots),
                     "fouls": sum(h["result"] == "foul" for h in hs),
                     "scratch": sum("scratch" in str(h["reason"]) for h in hs),
                     "invalid": sum(h["invalid"] is not None for h in hs),
                     "fallback": sum(str(h["source"]).startswith("opponent error") for h in hs)}
    side["a"]["fallback"] = errors
    judge = {w: getattr(f, "judge", None) for w, f in (("a", A), ("b", B))}
    return {"judge": {w: j for w, j in judge.items() if j is not None},
            "seed": seed, "seat": seat, "a": a, "b": b, "noise": noise, "rules": rules,
            "score": (env.outcome() + 1) / 2, "end": env.m.end_reason, "sides": side,
            "a_decisions": n, "cpu": time.process_time() - t}


def summarize(rows, a="A", b="B"):
    sc = [r["score"] for r in rows]
    hw = 1.96 * S.pstdev(sc) / len(sc) ** .5
    out = [f"{a} vs {b}: {len(rows)} games, score {S.mean(sc):.3f} +- {hw:.3f} (95%)"]
    for w, name in (("a", a), ("b", b)):
        x = {k: sum(r["sides"][w][k] for r in rows) for k in rows[0]["sides"][w]}
        out.append(f"  {name}: make {x['made'] / max(1, x['pots']):.1%} ({x['made']}/{x['pots']}), fouls "
                   f"{x['fouls'] / max(1, x['shots']):.1%} of {x['shots']} shots, scratches {x['scratch']}, "
                   f"invalid {x['invalid']}, agent errors {x['fallback']}")
    out.append("  ends: " + str(dict(C.Counter(("A wins: " if r["score"] > .5 else "A loses: " if r["score"] < .5
                                                else "draw: ") + str(r["end"]) for r in rows).most_common(8))))
    return "\n".join(out), S.mean(sc), hw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--seeds", nargs=2, type=int, required=True, metavar=("LO", "HI"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--noise", default="ours", choices=("ours", "icga2008"))
    ap.add_argument("--rules", default="ours", choices=("ours", "icga"))
    ap.add_argument("--out")
    ap.add_argument("--summarize", action="store_true", help="only summarize an existing --out file")
    args = ap.parse_args()
    if args.summarize:
        rows = [json.loads(x) for x in open(args.out)]
        print(summarize(rows, args.a, args.b)[0])
        return
    jobs = [(args.a, args.b, s, seat, args.noise, args.rules)
            for s in range(args.seeds[0], args.seeds[1]) for seat in "AB"]
    t = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [r for r in pool.imap(play, jobs, chunksize=1) if r]
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(summarize(rows, args.a, args.b)[0])
    print(f"  noise {args.noise}, rules {args.rules}, seeds [{args.seeds[0]}, {args.seeds[1]}), "
          f"{len(jobs) - len(rows)} failed games, {time.time() - t:.0f}s wall")


if __name__ == "__main__":
    main()
