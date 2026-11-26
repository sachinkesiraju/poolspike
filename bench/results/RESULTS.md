# Reference results

All games: pooltool 0.6.0 physics, one game per core on a Mac mini (M-series,
10 cores, shared with other jobs), `match.py` with fixed seeds, every seed
played in both seats. Score = first agent's mean result (win 1, loss 0), with
the 95% normal half-width. Make = called pots made / attempted. Fouls = fouls
per shot. Overlapping intervals are not a proven difference.

| A | B | noise / rules | games | A score +- 95% | A make | B make | A fouls | B fouls | seeds |
|---|---|---|---|---|---|---|---|---|---|
| cc | greedy | ours / ours | 100 | 0.940 +- 0.047 | 88.7% | 69.0% | 4.0% | 16.9% | 601500-601549 |
| cc | ev | ours / ours | 300 | 0.500 +- 0.057 | 88.7% | 90.2% | 3.7% | 4.4% | 601000-601149 |
| poolspike-deep | cc | ours / ours | 1000 | 0.666 +- 0.029 | 92.5% | 88.3% | 2.3% | 3.9% | 600000-600499 |
| poolspike-deep | cc | icga2008 / icga | 400 | 0.688 +- 0.045 | 95.1% | 90.3% | 1.8% | 4.2% | 602000-602199 |
| poolspike-deep | ev | ours / ours | 1000 | 0.662 +- 0.029 | 92.1% | 89.4% | 2.1% | 3.8% | 110000-110499 |
| poolspike-deep | cc:5 | ours / ours | 400 | 0.590 +- 0.048 | 91.9% | 89.1% | 2.8% | 2.9% | 600000-600199 |
| poolspike | cc | ours / ours | 1000 | 0.583 +- 0.031 | 90.8% | 88.6% | 2.5% | 3.6% | 600000-600499 |
| poolspike | cc | icga2008 / icga | 400 | 0.590 +- 0.048 | 93.4% | 90.6% | 2.6% | 3.7% | 602000-602199 |
| poolspike | ev | icga2008 / icga | 300 | 0.600 +- 0.055 | 92.7% | 91.1% | 2.4% | 3.0% | 602000-602149 |
| poolspike | cc:4 | ours / ours | 200 | 0.550 +- 0.069 | 90.0% | 89.4% | 3.2% | 4.1% | 604000-604099 |
| poolspike | cc:4 | ours / ours | 1000 | 0.550 +- 0.031 | 91.5% | 89.9% | 2.9% | 3.7% | 604100-604599 |

`cc:4` = 4x level-1/level-2 budget, 24 fully sampled variants, 12 shots to
level 2, up to 6 representative states (CueCard's CPU per decision: 11.0 s vs
2.8 s with `cc`; table below). The 1,000-game
confirmation on fresh seeds gives the same score with a tighter interval:
0.550 +- 0.031 vs 0.583 +- 0.031 against `cc`. The extra search narrows the
gap a little (the intervals just overlap) but Poolspike still wins. A first `cc:4` run that only
raised the budget (structure capped, ~same CPU) scored 0.520 +- 0.069
(`poolspike_vs_cc4_budgetonly.jsonl`).

Earlier reference (development repo, same harness): poolspike vs ev, our noise,
1000 games (seeds 110000-110499): 0.595 +- 0.030, make 91.0% vs 90.1%
(`results/vs_ev/` in the repository root).

CPU per decision, measured in the same games (20 games each, seeds
606000-606009, `cpu_*.jsonl`): 

| opponent | Poolspike s/decision | opponent s/decision | ratio |
|---|---|---|---|
| cc | 4.2 | 2.8 | 1.46 |
| cc:4 | 4.3 | 11.0 | 0.39 |
| ev | 3.1 | 10.9 | 0.28 |


How games ended (A's view), main match poolspike vs cc: poolspike legally pocketed the 8
in 556 games, cc in 402; cc lost 21 games by pocketing the 8 early, poolspike 8;
the rest were 8 in the wrong pocket / scratch on the 8 (13 games).

CueCard mechanisms in the main match: 6.0% of its shots were last-resort
shots (paper: under 1-2%).

## Calibration of Poolspike's judge (out of distribution under ICGA noise)

`tools/judge_calibration.py`: the judge's win chance for each position Poolspike faced,
binned, against how the game ended.

| setting | positions | mean predicted | won | Brier |
|---|---|---|---|---|
| ours vs cc | 5850 | 0.696 | 0.728 | 0.176 |
| icga2008 vs cc | 2264 | 0.715 | 0.766 | 0.163 |

In both settings the judge is slightly underconfident (by 3 and 5 points) and
its ranking holds bin by bin (ICGA: predicted 0.53 -> won 0.59, 0.70 -> 0.77,
0.90 -> 0.92). No sign that the change of noise broke it.

## Reproduce

```
python -B match.py cc greedy --seeds 601500 601550 --out cc_vs_greedy.jsonl
python -B match.py cc ev     --seeds 601000 601150 --out cc_vs_ev.jsonl
python -B match.py poolspike-deep cc   --seeds 600000 600500 --out deep_vs_cc.jsonl
python -B match.py poolspike-deep cc   --seeds 602000 602200 --noise icga2008 --rules icga --out icga_deep_vs_cc.jsonl
python -B match.py poolspike-deep ev   --seeds 110000 110500 --out deep_vs_ev.jsonl
python -B match.py poolspike-deep cc:5 --seeds 600000 600200 --out deep_vs_cc5.jsonl
python -B match.py poolspike cc    --seeds 600000 600500 --out poolspike_vs_cc.jsonl
python -B match.py poolspike cc    --seeds 602000 602200 --noise icga2008 --rules icga --out icga_poolspike_vs_cc.jsonl
python -B match.py poolspike ev    --seeds 602000 602150 --noise icga2008 --rules icga --out icga_poolspike_vs_ev.jsonl
python -B match.py poolspike cc:4  --seeds 604000 604100 --out poolspike_vs_cc4.jsonl
python -B match.py poolspike cc:4  --seeds 604100 604600 --out poolspike_vs_cc4_1k.jsonl
```

`poolspike-deep` is Poolspike plus the finalist search (`ref/poolspike_search.py`;
`rla_late` in its logs). Its logs come from the development harness (`run.py`),
which seeds the agents exactly as `match.py` does and also records each side's
CPU time (`cpu_a`, `dec_a`, ...); `match.py poolspike-deep ...` replays them
move for move. CPU per decision in its main match: 12.9 s vs cc's 2.45 s (one
core of a 10-core Mac mini running 7-9 games at once); against `cc:5`,
14.6 s vs 12.5 s. How games ended, poolspike-deep vs cc: it legally pocketed the 8 in
645 games, cc in 320; it lost 9 by pocketing the 8 early, cc 16.

Per-game logs are in this folder. CueCard's build, parameters, sources and
deviations: `CUECARD.md`.
