# 8-ball policy: noise-aware EV search

`policy.py` replaces the old `replay.choose_shot` stack (hand-weighted exec
score → top-2 → greedy rollouts → 0.30 confidence floors). One principle
now: **pick the move with the highest expected win-probability under the
player's own execution noise, using the same rules function the game uses.**

## Files

| file | role |
|---|---|
| `policy.py` | `Config` (one strategy), `choose` / `choose_bih` / `choose_break`, `adjudicate` (rules), `value` (leave evaluator) |
| `game.py` | headless `play_game({"A": cfg, "B": cfg}, seed)`; executes every stroke with the shooter's noise |
| `replay.py` | narrated example game → `replay_out/replay_log.json`, `video_items.json` |
| `fast_video.py` | game video in seconds: pooltool 3D render (persistent Panda3D per worker, vsync off, parallel, raw frames → ffmpeg); `--flat` = 2D OpenCV fallback |
| `viewer.py` | `replay_out/index.html`: video + clickable shot list |
| `audit_policy.py` | independent brute-force check of every turn of a logged game |
| `evals.py` | fixed position suite; each decision executed N times under noise |
| `tournament.py` | round-robin between `Config` entrants |

## Search presets

The same strategy and value model at three compute budgets (`policy.PRESETS`;
`pattern` is the `ev` search plus a second ply, see below):

| preset | search | time per shot |
|---|---|---|
| `ev` (`DEFAULT`) | every lane, 43-stroke position sweep, 12 basins, 6 finalists × 8 samples | ~5–45 s on 1 core, 1–7 s on 10 |
| `draft` | 6 basins, 3 finalists × 6 samples | ~half of `ev` |
| `lite` | top 8 lanes by geometric odds, 13-stroke sweep, 4 basins, 2 finalists × 4 samples, thinned break and ball-in-hand search | ~2–6 s on 1 core |

The bundled Coworld player is anytime. It runs `lite` first, then `ev` in
whatever time is left on the shot clock, and plays the deepest answer that
finished. The budget knobs (`max_lanes`, `exec_grid`, `break_samples`)
default to the full search, so `DEFAULT` and `DRAFT` are unchanged.
`tests/test_parity.py` checks that against the original demo.

**Is the full search worth it?** Over 30 games, `ev` vs `lite` finished
15–15: single games are short and swingy. On a per-shot basis over 20 of
those games (seeds 600–609), `ev` made 89% of its called pots (predicted
84%) and `lite` made 84% (predicted 80%); `lite` lost 2 games by sinking the
8 early and `ev` none. Both make slightly more pots than they predict.
(`tools/head_to_head.py ev lite --games=20 --seed=600`)

## Pattern player (`pattern.py`, `--strategy pattern`)

`ev` values a made pot's leave with the analytic `value`: the best
next-shot odds plus a race model. It never plays the next shot. The pattern
player adds a second ply where it matters:

1. Run the full `ev` search (`choose(..., keep=6)` also returns the top
   candidates with the noisy samples it already drew).
2. Take up to 4 pots within 0.15 EV of the leader and at least 25% to make.
3. For each, replay its first 3 *made* samples (same noisy strokes, so no
   new randomness) and run a real search for the next shot from every leave.
   The search is the lite budget, with direct pots only unless nothing is
   pottable. Its EV, which itself prices the leave after next, replaces
   V(leave): `EV2 = p_make·mean(next-shot EV) + (1 − p_make)·V(miss)`.
4. Pick argmax `EV2`, with the same near-tie rule (more makeable shot).
   Breaks, ball in hand and safeties play the `ev` search unchanged.

It is anytime on the clock: the worker yields the `ev` answer first, and
the second ply stops when the next pot would overrun the shot budget. The
note records the cut ("looked N pots ahead") so `tools/audit_replays.py`
replays the decision exactly. The second ply costs about 0–25 s per shot on
10 cores, most of it on positions with several close options.

## Speed (`fastsim.py`, `pot_check`)

Two changes make the search 2.75× faster (212 s → 77 s for 4 full
decisions on one core). Neither changes a decision: the 4 test decisions
matched to 9 decimal places, and `tests/test_fastsim.py` checks both.

- **`fastsim.py` patches pooltool's event loop**:
  - cache invalidation was a generator scan of every cached time per event;
  - the collision search re-probed every (ball, cushion) key per event;
  - the cushion model built numpy debug strings on every hit.

  Simulations come out bit-identical, 1.4–2× faster.
- **`pot_check`**: a pot test stops simulating as soon as the answer is a
  sure no: wrong ball first, cue ball down, or the called ball in another
  pocket. Blocked-lane probes, the bulk of the search, now cost a fraction
  of a full simulation.

## Strategy: how the pieces fit

The goal is an environment where every player plays the best it can, and
strategies can be compared and improved by tournament or RL. Each layer has
one job, and each has a known next step:

| layer | now | next step |
|---|---|---|
| **physics** | pooltool (validated, deterministic) | keep as the oracle; a batched JAX sim for RL throughput, checked against it |
| **real-table uncertainty** | stroke noise grows with pace and side spin, plus per-collision risk discounts (kiss, bank, kick, combo) | replace the discounts with physics randomization per sample (cushion/ball friction and restitution drawn per execution) |
| **search** | analytic candidates, an aim solver (3–5 sims per lane), position sweep with per-stroke re-aimed side spin, sampled EV, refine the leader | successive halving over candidates under a fixed sim budget per decision |
| **strategy (value of a leave)** | analytic: next-shot odds (straight + bank), openness, race-to-finish model | learn it from self-play (fitted value iteration) — this is the RL target |
| **evaluation** | brute-force audit per turn, position suite (realized vs predicted EV), tournaments, replay check | same harness, more seeds; tournaments between learned and analytic values |
| **iteration** | draft preset ~30 s/game; full ~2–4 min; 3D video ~25 s | render each shot as soon as it is decided |

## Decision pipeline (`choose`)

1. **Pots.** Every legal ball × pocket is seeded by an **aim solver**: it
   shoots, reads the object ball's departure angle just after contact, and
   corrects the aim. It needs 3–5 short sims; a blocked lane is known after
   one. Each verified lane then gets a **position sweep**: 5 speeds × 5
   draw/follow, plus left/right english × draw/stun/follow at 3 paces. Each
   stroke's aim is re-solved, because side spin shifts the line. Every
   stroke is scored by the value of the position it *actually* leaves. The
   chosen call records its plan (`plan`: where the cue stops and the next
   shot it sets up). The top strokes get a measured aim
   basin (bisection to 0.012°); aim is re-centred in the basin;
   `p_aim = erf(half_basin / (σ_phi·√2))`. The finalists are executed 10× with
   full noise (aim, speed, spin) plus two forced edge misses:
   `p_make = p_aim × (fraction of in-basin samples that still make it)`,
   `EV = p·V(make states) + (1−p)·V(miss states)`.
2. **Deep routes.** One-rail banks, one-rail cue kicks and combos join
   whenever no direct pot reaches `cfg.deep_below` (default 80%). They
   are prefiltered by geometry (open legs) before any sim. Every candidate's
   route is classified from its own simulated path: direct / bank (rails
   touched) / kick / combo / kiss. That label is what viewers see.
   Real-table risk applies per collision: `risk_kiss` 0.35, `risk_rail`
   0.20, `risk_kick` 0.20, `risk_combo` 0.20.
3. **Safeties.** Contact fraction (13) × speed (5) × spin (3) on every legal
   ball. If fewer than 6 legal hits exist (snookered), a full 360° aim sweep
   at 3 speeds finds kick escapes. The 8 best leaves are executed 8× under noise.
4. **Sample until the leader is confirmed.** Whichever option leads on EV
   gets 40 more full-noise executions, and its make rate becomes the plain
   sampled fraction (unbiased; the first-pass `p_aim × keep` estimate reads
   3–11 points high because english error shifts the basin). This repeats
   until the leader is already refined, so every chosen move rests on
   about 50 noisy executions.
5. **Pick** argmax EV across pots and safeties. No floors, no hand weights on
   the decision itself. The full ranked `contest` is returned with the call.

`choose_bih`: cue placements behind every open pot lane (3 distances ×
5 angles for shape), pre-ranked by make-odds × best leave, then the full
`choose` from the top 3 spots. `choose_break`: 30 (spot, speed, spin)
options × 16 noisy samples each, valued by the same function (scratch risk
is priced in).

## Value of a leave (`value`)

Win probability for the player to move:

* `p1` = their best next-shot probability, counting straight pots AND
  risk-discounted one-rail banks. (Counting only straight pots made
  "no straight shot" leaves look like locks, so the policy over-played
  safeties.) Geometric cut/distance model
  calibrated to sim pot basins (effective pocket half-width 33 mm corner /
  34 mm side, side tapering to 0 at ~65° approach; fit on 122 random shots).
* each side's group *openness* = share of their balls with a clear, feasible
  pocket lane → continuation make-rate `q = q_lo + (q_hi − q_lo)·openness`.
* a two-player race model over shots-left (`_race`): `shoot = p1·W(n−1, m) +
  (1−p1)·(1 − W_opp(m, n))`; `V = max(shoot, play-safe)`.

So every option is judged on **my position vs the opponent's**: their best
look, how open their group is, and how many balls each side has left.

## Rules (`adjudicate`, shared by game and policy)

Open table until a called ball is legally made. You can't hit the 8 first on
an open table. Fouls (scratch, wrong ball first, no contact, no rail after
contact) give ball in hand anywhere. A scratch on the break gives ball in hand
in the kitchen. The 8 on the break is re-spotted. A called safety ends the
turn. The 8 legally made in the called pocket wins; the 8 early, in the wrong
pocket, or with a scratch loses. Not modelled: the three-foul rule, jump/massé
shots.

## Eval results (`python evals.py`, 100 noisy executions per decision)

| position | ev policy | realized | greedy | realized |
|---|---|---|---|---|
| dead_safety | 2→rc combo, 72% | 0.74 | same lane, worse stroke, 34% | 0.54 |
| blocked_kick | 3→rt bank | 0.76 | 3→rt direct | 0.60 |
| combo_lane | safety off 5 | 0.38 | 3→lt (17% make, 16% fouls) | 0.32 |
| endgame_8 | 8→rb | 0.87 | 8→rb (no stroke choice) | 0.84 |
| hanger_free | 6→rt (100%) | 0.81 | 3→lt (95%) | 0.78 |
| two_hangers | 5→lt + shape | 0.90 | 5→lt stun | 0.81 |
| **mean** | | **0.744**, fouls 0% | | **0.680**, fouls 2% |

Predicted EV matched realized value within sampling noise on every position.

## Strategy research surface

Everything tunable is a `Config` field: skill noise (`sigma_phi`, `sigma_v`,
`sigma_ab`), value-model constants (`q_lo`, `q_hi`, `q_safe`), toggles
(`safeties`, `position`, `deep_below`) and search breadth (`shortlist`,
`finalists`, `n_pot_mc`, `n_safe_mc`). `tournament.py` already enters
ablations (`greedy`, `no_safety`, `no_position`, `defensive`, `ev_amateur`). An
autoresearch/RL loop mutates Configs (or swaps `value` for a learned
function) and keeps whatever wins head-to-head; `tournament.jsonl` records
every game's shot sequence for analysis.

## Commands

```
python replay.py 7                 # example game (pro noise, σ_aim 0.10°)
python replay.py 11 --sigma=0.18   # noisier execution: more safeties/BIH
python fast_video.py               # → replay_out/game.mp4 (3D, ~15 s/game) + index.html
python audit_policy.py             # brute-force audit of the logged game
python evals.py --entrants=ev,greedy
python tournament.py --games=2 --entrants=ev,greedy,no_safety
```

Cost: about 3–15 s per decision on 10 cores (ball in hand is roughly 3×). A
full game takes 2–6 min. The 3D video takes ~15 s per game on 8 workers
(~8 ms/frame/worker; the old JPEG + post-pass pipeline took minutes).

## Known limits

* `value` is a calibrated heuristic, not learned; its continuation rates
  (`q_lo`, `q_hi`, `q_safe`) are the main remaining hand-set constants. They
  are exactly what a tournament or self-play loop should fit.
* The opponent model in `value` assumes the opponent has the same skill as
  the deciding player.
* The sim is noiseless apart from the injected stroke noise: cloth and
  cushion variation are not modelled.

## Is pooltool good enough? (battery run 2026-10-20, pooltool-billiards 0.x on py3.12)

| check | result |
|---|---|
| robustness: 1500 random strokes (0.5–9 m/s, random spin) from real game states | 0 exceptions, 0 balls off the table, 0 overlapping at rest |
| determinism (same stroke ×3, hash of final state) | identical |
| sim cost | p50 18 ms, p99 400 ms, max 1.6 s (3 of 1500 above 1 s) |
| straight stun shot | cue keeps 2.5% of its speed (textbook stop shot) |
| 30° stun cut separation | 86.5° (≈90° rule; residual is throw and roll) |
| cut-induced throw | 1.7–1.8° at slow speed, 0.6° at 5 m/s (matches published throw curves) |
| cushion | normal restitution 0.78 |
| cloth | 0.59 m/s rolls 1.76 m = v²/2μg with μ_roll 0.01 (normal cloth) |

Bugs found are all **outside the physics core**: (1) `get_rack` picks ball
ids from a set, so racks aren't reproducible across processes even with a
seed (worked around in `game.rack_state`); (2) `pt.Cue` defaults to a
V0=2 stroke, so simulating a "rest" system fires a phantom shot (worked
around in fast_video/video_worker); (3) the Panda3D renderer leaks scenes
between renders, needs a monkeypatch to run offscreen, and `image_stack`
returns RGB, which the old pipeline treated as BGR. That swap is why the old
video's balls were blue and its cloth needed a recolor hack. All handled in
`fast_video.py`; (4) the PyPI name trap
(`pooltool` ≠ `pooltool-billiards`).

**Verdict: don't rewrite the simulator.** The event-based core (ball-ball
friction and throw, cushion model, pocket jaws, spin transitions) is the
hard, validated part, and it behaved correctly on every check above. The
one real reason to build our own is **RL throughput**: ~15 ms/sim on one core
caps us at ~600 sims/s on 10 cores. If self-play needs millions of games,
build a batched/approximate sim (e.g. JAX) for training and keep pooltool as
the oracle it's validated against. Compare the two head-to-head with this
battery plus the eval suite.
