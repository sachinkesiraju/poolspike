"""Calibration of r19's judge from match logs: the judge's win chance for each
position r19 faced vs how that game actually ended.

  python -B tools/judge_calibration.py games.jsonl [more.jsonl ...]
"""
import json
import math
import sys


def main(*paths):
    for path in paths:
        pts = []
        for line in open(path):
            r = json.loads(line)
            for w, js in (r.get("judge") or {}).items():
                y = r["score"] if w == "a" else 1 - r["score"]
                pts += [(p, y) for _, p in js]
        if not pts:
            print(path, "no judge records")
            continue
        n = len(pts)
        brier = sum((p - y) ** 2 for p, y in pts) / n
        ll = -sum(y * math.log(max(p, 1e-6)) + (1 - y) * math.log(max(1 - p, 1e-6)) for p, y in pts) / n
        mp = sum(p for p, _ in pts) / n
        my = sum(y for _, y in pts) / n
        print(f"{path}: {n} positions, mean predicted {mp:.3f}, mean outcome {my:.3f}, "
              f"Brier {brier:.4f}, log loss {ll:.4f}")
        edges = [0, .2, .4, .6, .8, 1.0001]
        for lo, hi in zip(edges, edges[1:]):
            b = [(p, y) for p, y in pts if lo <= p < hi]
            if b:
                print(f"  predicted {lo:.1f}-{min(hi, 1):.1f}: n {len(b):5d}, mean predicted "
                      f"{sum(p for p, _ in b) / len(b):.3f}, won {sum(y for _, y in b) / len(b):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
