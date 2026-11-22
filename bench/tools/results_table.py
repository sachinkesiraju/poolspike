"""Results table from match logs.

  python -B tools/results_table.py results/*.jsonl > results/table.md
"""
import collections as C
import json
import os
import statistics as S
import sys


def row(path):
    rs = [json.loads(x) for x in open(path)]
    if not rs:
        return None
    sc = [r["score"] for r in rs]
    hw = 1.96 * S.pstdev(sc) / len(sc) ** .5
    seeds = sorted({r["seed"] for r in rs})
    out = {"file": os.path.basename(path), "a": rs[0]["a"], "b": rs[0]["b"], "noise": rs[0]["noise"],
           "rules": rs[0]["rules"], "games": len(rs), "score": S.mean(sc), "hw": hw,
           "seeds": f"{seeds[0]}-{seeds[-1]}"}
    for w in ("a", "b"):
        x = {k: sum(r["sides"][w][k] for r in rs) for k in rs[0]["sides"][w]}
        out[w + "_make"] = x["made"] / max(1, x["pots"])
        out[w + "_fouls"] = x["fouls"] / max(1, x["shots"])
    ends = C.Counter()
    for r in rs:
        who = "A" if r["score"] > .5 else "B" if r["score"] < .5 else "draw"
        ends[f"{who}: {r['end']}"] += 1
    out["ends"] = ", ".join(f"{k} {v}" for k, v in ends.most_common(5))
    return out


def main(*paths):
    rows = [r for r in map(row, paths) if r]
    print("| A | B | noise / rules | games | A score +- 95% | A make | B make | A fouls | B fouls | seeds |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['a']} | {r['b']} | {r['noise']} / {r['rules']} | {r['games']} | {r['score']:.3f} +- {r['hw']:.3f} "
              f"| {r['a_make']:.1%} | {r['b_make']:.1%} | {r['a_fouls']:.1%} | {r['b_fouls']:.1%} | {r['seeds']} |")
    print()
    for r in rows:
        print(f"- {r['a']} vs {r['b']} ({r['noise']}/{r['rules']}) game ends: {r['ends']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
