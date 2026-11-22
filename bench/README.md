# Pool8: an 8-ball benchmark for computational pool agents

A fixed simulator, execution-noise model, rule set, agent interface, reference
agents and a seeded match runner, so two 8-ball agents can be compared on
equal terms and the result reproduced game for game.

Released with Poolspike, November 2026. MIT license (see ../LICENSE).

## Contents

```
match.py              match runner: A vs B on fixed seeds, both seats, 95% CIs
reference_agents.py   agent specs -> agents (reference agents + your own)
example_agent.py      the turn / action format, and an example agent to copy
pool8/                simulator + rules + game engine (vendored, pinned)
  engine.py             the game state machine (one rule function, one noise model)
  policy.py             physics wrapper, rules (adjudicate), noise (noisy), EV agent
  rl/env.py             PoolEnv: single-agent env over engine.Match
  PROTOCOL.md           wire format of turns and actions
ref/                  reference agents (vendored)
  poolspike_player.py   Poolspike player (with shotnet.py, vnet.py support code)
  judge_net.py          Poolspike's learned judge
  nets/poolspike.npz    Poolspike weights
  cuecard.py, cc_data/  CueCard rebuild and its tables
  noise_models.py       the ICGA-2008 noise / rules variants
results/              the reference results table and per-game logs
tools/                results_table.py, judge_calibration.py
MANIFEST.sha256       hashes of every vendored file
```

## Setup

Python 3.12, then `pip install -r pool8/requirements.txt` (the physics is
`pooltool-billiards==0.6.0`; install it with `--no-deps` on Linux and add
`panda3d==1.10.16`, see the comment in requirements.txt). Nothing renders;
the physics is pooltool's event-based simulator, made faster by
`pool8/fastsim.py` with bit-identical results.

## The game

**Table.** pooltool 0.6.0 `seven_foot_showood`: playing surface
0.9906 x 1.9812 m, ball radius R = 28.575 mm, corner pocket radius 62 mm, side
64.5 mm. Coordinates in metres, origin at the bottom-left cushion corner,
pockets `lb lc lt rb rc rt` (left/right x bottom/centre/top). Kitchen:
y < 0.4953 m (head string at a quarter of the length).

**Stroke.** `phi` (deg, 0 = +x, 90 = +y), `speed` = cue-ball speed in m/s
(0.3-8; break up to 12), tip offsets `side` = a and `vertical` = b in units of
R (|a|, |b| <= 0.6, hypot <= 0.7). Level cue (no elevation, no jumps or
masse).

**Noise (`--noise ours`, default).** Every executed stroke gets independent
Gaussian noise:

| parameter | sd |
|---|---|
| phi | 0.10 deg x (1 + 0.2 max(0, V - 2.5) + 1.5 abs(a)) |
| V | 2.5 % of V (result at least 0.3 m/s) |
| a, b | 0.025 R (0.71 mm), clipped to +-0.6 |

The noise stream is seeded per game (`seed * 7919 + 1`), so both seats of a
seed see the same dice. Because the stream follows from the seed, and the rack
reveals the seed, an agent could in principle reconstruct it. The reference
agents never do; for an open competition, pass a secret `noise_seed` to
`engine.Match`.

**Noise (`--noise icga2008`).** The 2008 ICGA computational-pool tournament
model (Archibald et al., AI Magazine 2010; identical to the 2006 Olympiad's
E_Italy in Smith's PickPocket thesis): sd phi 0.125 deg, V 0.075 m/s, a = b =
0.5 mm (0.0175 R), theta 0.1 deg. Differences from the original: V is applied
to our cue-ball speed (POOLFIZ perturbed the cue-stick speed), and theta is not
modelled because the cue is always level.

**Rules (`--rules ours`, default): call-shot 8-ball.**
* Break from the kitchen. Breaker keeps the table if any object ball drops
  (and takes the group of the first ball down; `--rules icga` keeps the table
  open instead, the BCA/ICGA rule). 8 on the break is re-spotted. Scratch on
  the break: opponent has ball in hand behind the head string.
* Every shot calls a ball and pocket, or a safety (`call: null`). Only the
  called ball in the called pocket keeps the turn; other balls that drop stay
  down. Open table: any ball but the 8 may be hit first; the first legally
  called pot assigns groups.
* Legal shot: first contact with a legal ball, then a ball pocketed or any
  ball reaching a rail, and no scratch. A foul gives the opponent ball in hand
  anywhere.
* The 8: legally calling and pocketing it after clearing your group wins;
  pocketing it early, in the wrong pocket, or on a foul loses.
* A game stops after 81 shots (draw, scored 0.5). No clocks.

Not implemented from BCA/ICGA: the 4-balls-to-a-rail break requirement and
the incoming player's re-rack option, the 8 as legal first contact on an open
table, the ICGA 10-minute-per-game clock.

## Writing an agent

An agent is `factory(seed=int, seat=str)` returning a callable
`act(turn) -> action` (or an object with `.act`). The `turn` and `action`
dicts are documented in `example_agent.py` and `pool8/PROTOCOL.md`; `example_agent.py` has a
working example (`Example`, a geometric potter). Agents may plan with the
simulator: `policy.simulate`, `policy.summarize`, `policy.adjudicate`,
`policy.noisy(stroke, rng, policy.DEFAULT)`; under `--noise icga2008` these
already use the tournament noise. Run it:

```
python -B match.py py:example_agent.py:Example greedy --seeds 600000 600050
```

Invalid actions are replaced by the engine's fallback shot and counted; an
agent that raises gets the same fallback, counted as an agent error.

## Running a match

```
python -B match.py A B --seeds LO HI --workers 9 --out games.jsonl \
       [--noise ours|icga2008] [--rules ours|icga]
python -B match.py A B --out games.jsonl --summarize
```

Each seed in `[LO, HI)` is played twice, once with each agent breaking, so
`HI - LO` seeds give `2 (HI - LO)` games. Output: A's score (win 1, draw 0.5)
+- the 95% normal half-width (1.96 sd / sqrt(n)), both sides' make rate (called
pots made / called pots attempted), fouls per shot, scratches, invalid actions
and how games ended. Agent seeds are derived from the game seed and seat
exactly as in the original harness, so the published rows below re-run game
for game. Use fresh seed ranges when you test a new agent; two results whose
intervals overlap are not a proven difference. One game takes one core;
`--workers` games run in parallel.

Agent specs: `greedy`, `lite`, `ev`, `cc` / `cc:<budget multiplier>`,
`poolspike` (alias `r19`), `n5:<weights.npz>`, `py:<file.py>:<factory>`.

## Reference agents

* **greedy**: easiest geometric pot, no search (the server's fallback shot).
* **ev** (`pool8/policy.py`, `pool8/POLICY.md`): noise-aware expected-value
  search. Direct/bank/kick/combo lanes, a speed x draw/follow x english sweep,
  per-stroke aim solving, sampled outcomes under the table noise, an analytic
  leave evaluator and race model. `lite` is its budget preset.
* **poolspike** (alias `r19`, its name in the logs; `ref/poolspike_player.py` +
  `ref/nets/poolspike.npz`): hand-written candidate generation and fine aim, 24 noisy simulations per shortlisted shot, choosing
  the best mean win chance under a learned position judge (VNet2, trained by
  19 rounds of self-play). It is a hybrid, not a pure net: the judge's inputs
  include EV's analytic evaluator and it still searches with simulations.
  About 2 s per decision on one core.
* **cc** (`ref/cuecard.py`): a rebuild of CueCard, the 2008 Computer Olympiad
  champion (Archibald et al., IJCAI 2009), on this simulator: shot
  generation from PickPocket-style tables, Monte Carlo variants with 25 noisy
  samples, state clustering, a second search level (repeat search early, a
  noiseless search to the end late), last-resort shots, a precomputed break,
  grid-and-hill-climb ball in hand. Its search budget is scaled down to about
  Poolspike's compute; every parameter and deviation is documented at the top
  of the file. `cc:4` gives it four times the budget and breadth. Tables are
  rebuilt per noise model with `python -B ref/cuecard.py build` (under `POOL_NOISE`).

## Results

See `results/RESULTS.md` (games, score +- 95% CI, make %, fouls, seeds,
commands) and the per-game logs next to it.
