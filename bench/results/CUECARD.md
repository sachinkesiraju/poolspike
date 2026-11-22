# CueCard rebuild: what was built, from which source, and what differs

Sources: [IJCAI] Archibald, Altman & Shoham, "Analysis of a Winning
Computational Billiards Player", IJCAI 2009. [AIM] Archibald, Altman,
Greenspan & Shoham, "Computational Pool: A New Challenge for Game Theory
Pragmatics", AI Magazine 2010. [PP] M. Smith, "PickPocket: An Artificial
Intelligence for Computer Billiards", MSc thesis, U. Alberta 2006 (CueCard
reuses PickPocket's evaluation function [IJCAI 3.1]). All three PDFs loaded.

Code: `ref/cuecard.py` (tables: `ref/cc_data/<noise>_tables.npz`, break:
`ref/cc_data/<noise>_break.json`, rebuilt per noise model).

## Algorithm as built

1. Straight-in shots for every legal ball x pocket: direction aimed at the
   pocket centre, minimum speed from the velocity table + 0.2 m/s [PP 3.1.1,
   3.1.6]. Impossible shots (blocked paths, cut >= 85 deg, ghost ball off the
   table) are discarded [PP 3.1.1].
2. Speeds v0, v0 + 1, ... up to 4.5 m/s [PP 3.1.1]; a (phi, v) pair is kept
   if the noiseless shot pots the called ball [IJCAI step 2].
3. Variants: the plain stroke, then random english (a, b uniform in a disk of
   radius 0.5 R) for each kept pair, each re-checked noiselessly [IJCAI step 3,
   AIM]; every variant is screened with 6 noisy samples and the best 6 are
   sampled up to 25 with the table noise (screening: PickPocket's root
   sampling [PP 3.1.9]; only fully sampled variants can be played) and every result
   scored by PickPocket's evaluation 1 p1 + 0.33 p2 + 0.15 p3 over the
   mover's best straight-in table probabilities [AIM, PP 3.2]; a failed sample
   scores 0 [PP Fig. 3.10]. Generation is round-robin over pairs until the
   level-1 budget is spent (CueCard ran until a time limit [AIM]).
4. Banks, kicks and two-ball combos are generated when no straight-in variant
   exists or the best level-1 score is below 0.8 [PP 3.1.8 activation].
5. Top two variants per pair; the top K go to level 2 [IJCAI steps 3c, 4].
   Their successful samples are clustered into representative states (same
   balls pocketed, every ball within 5 cm, evaluation within 0.15) [IJCAI 3.4].
6. Level 2 from each representative state [IJCAI step 5]: early game, the
   search of steps 1-3 with smaller constants, value = best shot's mean
   evaluation; late game (own balls + 8 <= 3), a noiseless search to the end of
   the game with each shot weighted by its table probability and misses valued
   by a Markov-chain race model of the opponent [IJCAI 3.4]. Variants are
   re-scored with the level-2 values; the best is played [IJCAI step 6].
7. If the best score is below a threshold, or nothing is feasible: a
   last-resort sweep of 72 directions x 2 speeds, non-fouling strokes kept and
   valued vCC - vOPP over noisy samples [IJCAI 3.1].
8. Break: one precomputed shot from an offline search over cue spot, speed,
   aim offset and draw, scored keep-the-turn + 0.2 x spread [IJCAI 3.3, AIM].
9. Ball in hand: a 10 x 20 grid of cue spots valued by the best straight-in
   table probability, the 4 best local maxima from 24 random hill climbs, a
   shortened single-level search from each [AIM, PP 3.4.2].
10. Tables [PP 3.1.5-3.1.6]: success probability over (cut 0-80 deg x 9,
   object-pocket distance x 8, cue-ghost distance x 8, approach angle x 4
   corner / x 5 side), 48 noisy samples per entry, and the minimum-speed table
   from noiseless bisection; separate corner and side tables.

## Parameters

| parameter | value | source |
|---|---|---|
| evaluation weights | 1.0, 0.33, 0.15 (n = 3), max 1.48 | [PP 3.2], [AIM] |
| failed sample value | 0 | [PP Fig. 3.10] |
| speed boost over table minimum | 0.2 m/s | [PP 3.1.6] |
| speed step, maximum | 1 m/s, 4.5 m/s | [PP 3.1.1] |
| samples per variant (level 1) | 6 screening, 25 for the best 6 | [IJCAI]: 25-100 ("20 to 100" in 3.4); screening is dev |
| top variants per (phi, v) pair | 2 | [IJCAI step 3c] |
| variants to level 2 | 3 | [IJCAI]: 20 (dev, budget) |
| representative states per variant | <= 3 | [IJCAI]: 50 samples -> 5-10 (dev, budget) |
| level-2 constants | 3 lanes x 2 speeds, 2-4 samples | [IJCAI]: "smaller constants" (dev) |
| bank/kick/combo activation | best < 0.8 | [PP 3.1.8] (PickPocket's b0) |
| last-resort threshold | 0.10 on max(level-1, level-2 value) | [IJCAI]: "a certain threshold" (dev) |
| late game | own balls + 8 <= 3 | dev (paper: "late game") |
| Markov chain make rate | 0.80 | dev; [IJCAI]: chain from thousands of self-play games |
| random english radius | 0.5 R | dev; [IJCAI]: "feasible values" |
| table grid / samples | 9x8x8x(4 or 5), 48 | dev; [PP]: 30^4, s = 200 |
| break score | keep + 0.2 spread | dev; [AIM]: keep weighted over spread |
| ball-in-hand grid, k, c | 10 x 20, 4, 24 | dev; [PP 3.4.2] gives the method, not values |
| level-1 / level-2 budget | ~330 / ~250 simulations | dev (about r19's CPU per decision) |

## Deviations from the papers

* **Compute.** CueCard used 600 s per game on one CPU (tournament version on
  a 20-machine cluster) with physics 5.9x faster than POOLFIZ [IJCAI 4.2];
  this rebuild gets ~3 s per decision on one core in pooltool (about 4 ms per
  full-table simulation), roughly two orders of magnitude fewer simulations.
  K = 3 instead of 20 shots reach level 2. `cc:N` multiplies the budget, the
  fully sampled variants (6N), the shots sent to level 2 (3N, at most 20)
  and the representative states (3 sqrt N, at most 10).
* **Physics.** pooltool 0.6.0 instead of POOLFIZ: throw and squirt differ, so
  directions are found with an aim solver on the simulated object-ball
  departure (CueCard computed them geometrically), and english variants
  re-solve the aim (CueCard kept phi and re-checked).
* **Stroke.** No cue elevation (theta = 0 always); POOLFIZ's V is the cue
  speed, ours is the cue-ball speed (max 8 m/s; CueCard's 4.5 m/s cap is kept).
* **Shots not built.** "Special shots designed to disperse clusters"
  [IJCAI step 1]; CueCard's own 5.9x physics re-implementation; the
  distributed architecture; time-based (instead of count-based) budgets;
  varying the sample count by game stage (the paper found no gain from it).
* **Tables.** Coarser grid, multilinear interpolation instead of nearest
  entry; entries count the object ball dropping (scratch or not); each entry
  is the best success rate over three speeds (vmin + 0.3, 2.0, 3.2 m/s) at the
  centre of the potting window.
* **Aim.** Directions come from an aim solver toward points across the pocket
  mouth, centred in the noiseless potting window (+-0.36 deg scan), found once
  per lane and re-solved per speed and per english variant.
* **Feasibility.** "Pots the called ball without noise" ignores scratches
  (the sampled variants pay for them), as the papers word it.
* **Markov chain.** A two-player race model with one per-shot make rate
  instead of a chain fitted on thousands of self-play games.
* **Break.** Our break search found a 77% keep rate under our noise (87%
  under ICGA noise); CueCard's keeps 92% in POOLFIZ.

## Build history (honest record)

* **v1 (discarded).** Aimed with a single solver run at the pocket centre and
  required "no scratch" for feasibility. Its tables were badly wrong in
  places (cut-shot entries with p = 0 and a minimum speed of 4-6 m/s where
  the shot pots at 1 m/s with a centred aim; mean table p 0.56), so it often
  generated no feasible straight-in shot, 16-26 % of shots were last
  resorts, and a level-2 budget bug scored late representatives 0. It made
  ~72 % of called pots, scored 0.57 +- 0.10 vs greedy (100 games, laptop) and
  ~0.15 vs EV in small checks. Those numbers are not CueCard and are not used.
* **v2 (reported).** Robust centred aim, rebuilt tables (mean p 0.86),
  scratch-agnostic feasibility, level-2 budget split equally per
  representative state, screening + top-up at level 1, last-resort test on
  max(level 1, level 2). Checks before the main runs: 0.94 vs greedy
  (50 games, make 90 %, fouls 4.4 %, last resorts 5 %), 0.47 vs EV (30 games,
  make 86 %).
