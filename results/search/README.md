# How Poolspike-deep's finalist search was chosen

Every candidate change was played against Poolspike (`poolspike`). Each game is
one line in a `.jsonl` file. Fields: seed, seat, agents `a` (the challenger)
and `b`, score for `a` (win 1, loss 0), how the game ended, per-side shot
counts, and each side's CPU time and decisions (`cpu_a`, `dec_a`, `cpu_b`,
`dec_b`). The `.log` files show the running score.

Every seed is played in both seats on the same rack (paired). Scores are for
`a`, with 95% paired intervals. CPU is `a`'s CPU per decision divided by `b`'s,
measured in the same games.

## Protocol

1. **Run 1:** a sequential test (SPRT, H0 0.50 vs H1 0.55, 5% error rates, cap
   600 pairs) on a shared block of seeds.
2. **Run 2:** any pass is tested again, the same way, on fresh seeds.
3. **Fixed runs:** the winner is checked with fixed-length runs, which are the
   numbers the README reports. A sequential test stops when the score looks
   clear, so its final score is biased upward; the run-1 and run-2 scores below
   are for selection only.

## Runs

| file | a | b | games | score | CPU | seeds |
|---|---|---|---|---|---|---|
| ens_vs_poolspike_run1 | ens | poolspike | 255 | 0.488 ± 0.043 | 1.00 | 810000–810128 |
| big_vs_poolspike_run1 | big | poolspike | 264 | 0.477 ± 0.054 | 0.98 | 810000–810132 |
| bih_vs_poolspike_run1 | bih | poolspike | 381 | 0.505 ± 0.018 | 1.06 | 810000–810190 |
| safe_vs_poolspike_run1 | safe | poolspike | 218 | 0.495 ± 0.030 | 1.05 | 810000–810108 |
| refine_vs_poolspike_run1 | refine | poolspike | 372 | 0.554 ± 0.044 | 1.29 | 810000–810186 |
| refine_vs_poolspike_run2 | refine | poolspike | 410 | 0.549 ± 0.041 | 1.30 | 851000–851206 |
| endgame_vs_poolspike_run1 | endgame | poolspike | 134 | 0.591 ± 0.051 | 2.18 | 810000–810067 |
| endgame_vs_poolspike_run2 | endgame | poolspike | 490 | 0.539 ± 0.032 | 2.26 | 852000–852245 |
| ceiling_vs_poolspike_run1 | ceiling | poolspike | 140 | 0.614 ± 0.066 | 8.60 | 810000–810069 |
| ceiling_vs_poolspike_run2 | ceiling | poolspike | 94 | 0.685 ± 0.087 | 7.90 | 853000–853049 |
| la_vs_poolspike_run1 | la | poolspike | 683 | 0.534 ± 0.035 | 6.15 | 820000–820344 |
| rla_vs_poolspike_run1 | rla | poolspike | 181 | 0.573 ± 0.055 | 6.87 | 820000–820092 |
| rla_vs_poolspike_run2 | rla | poolspike | 169 | 0.613 ± 0.073 | 7.14 | 861000–861084 |
| rla_late_vs_poolspike_run1 | rla_late | poolspike | 310 | 0.565 ± 0.049 | 3.95 | 820000–820156 |
| rla_late_vs_poolspike_run2 | rla_late | poolspike | 134 | 0.629 ± 0.077 | 4.01 | 862000–862069 |
| deep_vs_poolspike (fixed, stopped at 513 of 1,000) | rla_late | poolspike | 513 | 0.567 ± 0.039 | 3.87 | 870000–870259 |
| **deep_vs_poolspike_equal_cpu** (fixed) | rla_late | poolspike_x | 1000 | **0.559 ± 0.029** | 1.03 | 880000–880499 |
| fast_vs_cc (fixed) | fast | cc | 400 | 0.545 ± 0.049 | 1.30 | 600000–600199 |
| opp_lookahead_vs_deep | c3_opp | rla_late | 279 | 0.489 ± 0.046 | 1.24 | 830000–830139 |
| refine2_vs_deep | c3_hill | rla_late | 419 | 0.519 ± 0.041 | 1.17 | 830000–830210 |

`la` and `c3_hill` were stopped before the test decided.

## The challengers

All of them use Poolspike's judge and candidate generation unless stated.

- **ens:** the mean of the judges promoted in rounds 12, 13, 16, 18 and 19.
- **big:** a judge 4× wider (32-unit ball encoder, 64 hidden units), trained
  from scratch on every stored self-play position (about 120,000).
- **bih:** more ball-in-hand placements (12 spots instead of 5, the full
  stroke grid).
- **safe:** deliberate safety shots added to the candidates.
- **refine:** stroke refinement of the chosen shot (see the README).
- **endgame:** with 2 or fewer balls left, a 2.3× larger search plus refinement
  and lookahead on the top 3.
- **ceiling:** everything at once: 2.3× larger search, then refinement and
  lookahead on the top 3, on every shot.
- **la:** lookahead on the top 3 on every shot, without refinement.
- **rla:** refinement and lookahead on the top 3 on every shot.
- **rla_late:** refinement of the best shot; with 4 or fewer balls left,
  refinement and lookahead on the top 3. **This is Poolspike-deep**
  (`poolspike/poolspike_search.py`).
- **poolspike_x:** Poolspike with a larger search at the same CPU as `rla_late`: 10 aim
  lanes, shortlist 16, 8 + 120 noisy plays (Poolspike: 5, 6, 4 + 20).
- **c3_opp:** Poolspike-deep, plus a lookahead on the opponent's best reply for the
  tables where its ball misses.
- **fast:** Poolspike with a cut-down search (4 aim lanes, shortlist 3, 3 + 6
  noisy plays) plus refinement on 12 draws, sized to CueCard's CPU per shot.
  Played against CueCard, not Poolspike.
- **c3_hill:** Poolspike-deep, plus a second round of refinement at half the step
  size around the best variant.

## The diagnosis

`diagnosis_selfplay.jsonl`: 150 Poolspike self-play games (1,798 shots). For
each shot it records the judge's win chance for the shooter before and after
the shot, the balls left and the outcome. Of the win chance the shooter lost
through its own shots, 43.2% was lost with 1–2 of its balls left and 28.1%
with 3–4. That 71% is why the lookahead runs late in the rack.
