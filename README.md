# Poolspike

**A self-play 8-ball player that beats the strongest hand-tuned search.**

Poolspike plays every candidate shot many times in a physics simulator with
execution noise. It scores each resulting table with a 2,354-parameter
win-probability network trained by self-play. Every tournament champion since
2005 searched shots under noise and scored the results with a hand-written
formula. Poolspike keeps the search and learns the evaluation, and sets a new
state of the art over hand-tuned search: it beats our rebuild of CueCard, the
2008 tournament champion, and our strongest hand-written agent, EV.

![Poolspike against EV: before each shot it shows its shortlist, with each option's make rate over 24 noisy simulations and its learned win chance, then plays the shot](media/poolspike.gif)

*Poolspike (stripes) runs out against EV. Before each shot, the right panel
shows its shortlist: make rate over 24 noisy simulations and the judge's win
chance for each option. Full game, with sound:
[media/poolspike_vs_ev.mp4](media/poolspike_vs_ev.mp4). Render your own with
[`viz/`](viz/README.md). In the logs, Poolspike's agent name is `r19` (round
19 of training).*

| Match (95% intervals; a win counts 1) | Games | Poolspike score |
|---|---|---|
| vs CueCard rebuild | 1,000 | **0.583 ± 0.031** |
| vs CueCard rebuild, ICGA 2008 tournament noise | 400 | **0.590 ± 0.048** |
| vs EV (our strongest hand-written agent) | 1,000 | **0.595 ± 0.030** |

About 2 CPU-seconds per shot, or 0.5 s on 10 cores with identical moves.

## Background

In computer pool, a shot has five continuous parameters: aim, elevation, speed
and two cue-tip offsets for spin. Gaussian noise perturbs each one before the
physics runs, so a strong player must pick shots that still work when hit
imperfectly.

The ICGA computational pool tournaments (2005–2008) posed this problem in
8-ball:

- **PickPocket** (University of Alberta; [Smith, AAAI 2006](https://cdn.aaai.org/AAAI/2006/AAAI06-156.pdf)) won in 2005 and 2006.
- **CueCard** (Stanford) won in 2008, taking 64 of 78 games. It later beat
  PickPocket 492–144 ([Archibald et al., IJCAI 2009](https://www.ijcai.org/Proceedings/09/Papers/231.pdf)).

Both used search with a hand-written evaluation. CueCard's code was never
released, so we rebuilt it from its papers in our simulator. The rebuild
scores 0.940 against a greedy player and ties our EV agent (0.500 ± 0.057).

## How it works

Three ideas carry the method.

1. **Noise-aware sampling.** Poolspike builds candidates from 5 aim lanes ×
   14 strokes per pot, plus banks, kicks, combinations and ball-in-hand
   placements. It plays each one 4 times with noise, keeps the best 6, and
   plays those 20 more times. A shot is judged by how it lands, not how it
   was aimed.
2. **A learned judge.** Each landing position is scored as follows. The rules
   settle any win or loss. If the shooter keeps the table, the judge gives the
   shooter's win chance. If the turn passes, Poolspike uses 1 − the judge's
   score for the opponent. Poolspike plays the option with the highest mean
   win chance.
3. **Self-play with gating.** The judge is trained only on games Poolspike
   plays, mostly against itself. A refit judge must beat the current champion
   head to head before it replaces it, as in AlphaZero's loop. There is no
   policy network and no deep tree search, only one ply of noisy simulation.

### The judge (VNet2)

The judge is a deep-sets network, so it reads any number of balls in any
order:

![VNet2 judge architecture: per-ball features through a shared encoder, pooled into own, opponent and 8-ball sums, joined with table features, one hidden layer, and an analytic-logit skip to the win-probability output](media/judge.svg)

- Each ball is described by 12 features: position, offset from the cue ball,
  best pot chance, pockets potted at better than even, open lanes, nearest
  ball and rail distance.
- A shared tanh encoder (12 → 16) maps each ball. The encodings are summed into
  three pools: own group, opponent's group and the 8-ball.
- The pools are joined with 17 standardized table features (65 inputs), then
  pass through a 32-unit tanh layer to a single logit.
- EV's analytic win estimate g₀ also feeds the output directly:

```
logit = 0.28 · g₀ + 0.59 + oᵀ tanh(W [ĝ; P_own; P_opp; P_8] + d)
```

The network learns only a correction to the hand-written estimate. That is why
2,354 parameters and about 7,600 games were enough.

### Training

Each round of `poolspike/train_selfplay.py`:

1. **Play.** The champion plays 300 self-play games at temperature 0.3, plus
   100 games against EV.
2. **Refit.** A new judge is trained on every position so far, starting from
   the champion's weights. Training uses logistic loss, Adam at 0.002,
   λ-return targets with λ = 0.7 and mirror augmentation.
3. **Gate.** The new judge plays the champion over 300 games. It is promoted
   only if it scores above 0.5.
4. **Check.** Every third round, the champion plays EV over 300 games.

The final judge was promoted in round 19 of 20. The 20 rounds took about 25
hours on a 10-core Mac mini. Only that final champion was then tested on
1,000 fresh games.

### Why the judge is the lever

| Same pipeline, scored by | vs EV |
|---|---|
| EV's hand-written evaluation | 0.535 ± 0.069 (200 games) |
| the learned judge | 0.595 ± 0.030 (1,000 games) |

Giving Poolspike 2.3× or 7× more search gave no measurable gain. Giving the
CueCard rebuild 4× more search did not close the gap (0.550 ± 0.069, 200
games).

## Repository

```
bench/                  pool8-bench: self-contained benchmark. Start here (bench/README.md).
  match.py                seeded match runner, both seats per seed, 95% intervals
  example_agent.py        turn/action interface with an example agent to copy
  reference_agents.py     reference agents: greedy, ev, cc (CueCard rebuild), poolspike
  pool8/                  simulator wrapper, rules, noise, game engine (pooltool 0.6.0)
  ref/                    vendored copies of the agent code, weights and CueCard tables
  results/                RESULTS.md, CUECARD.md and per-game logs
  tools/                  results_table.py, judge_calibration.py
poolspike/              the player and its training
  poolspike_player.py     the player: candidates, noisy sampling, shortlist
  judge_net.py            the judge, plus data generation, fitting and evaluation
  train_selfplay.py       self-play training loop
  cuecard.py              CueCard rebuild (cc_data/: its precomputed tables)
  nets/v0.npz             starting judge
  nets/poolspike.npz      final Poolspike network (round 19)
sim/                    game engine, rules, physics wrapper and EV policy
results/                logs behind the reported numbers
  training/               20 rounds: loop.log, gate and EV-check games
  vs_cuecard/             Poolspike and CueCard matches, both noise models
  vs_ev/                  final 1,000-game Poolspike vs EV confirmation
  runout/                 run-out rate from 200 start positions
viz/                    render any game as a video with table sounds (viz/README.md)
media/                  the game above (GIF, and full video with sound), judge diagram
```

The logs in `results/` were written before this layout, so the paths inside
them use the old folder and file names (`wsp/`, `nets6/`, `runs/v6`,
`vnet2.py`, `shotnet5.py`).

## Quick start

Requires Python 3.12. On Linux, see the note in `bench/pool8/requirements.txt`
about installing pooltool.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r bench/pool8/requirements.txt
cd bench
../.venv/bin/python -B match.py poolspike cc --seeds 600000 600010 --workers 4
```

That plays 20 games of Poolspike against the CueCard rebuild (`cc`). The
headline match is `--seeds 600000 600500 --workers 9`, about 80 minutes on
a 10-core Mac mini. Add `--noise icga2008 --rules icga` for tournament
conditions. To score your own agent, pass `py:path/to/file.py:factory`; see
`example_agent.py`.

To retrain (about 25 hours), first set `R` in `poolspike/train_selfplay.py`
to a new directory so the logs in `results/training` are not appended to,
then run from `poolspike/`:

```bash
../.venv/bin/python -B train_selfplay.py nets/v0.npz 20
```

## Evaluation protocol

- **Paired games.** Every seed is played twice, so each player breaks once.
  The two games share the rack and the noise stream, which cancels much of
  the luck.
- **Held-out seeds.** Reported matches use seeds 600000–604099. Poolspike
  trained on seeds below 50000, and CueCard was debugged on 603000–603214. No
  reported game was seen during development.
- **No access to the real noise.** The game draws its noise from its own
  random generator. Poolspike and the reference agents never read it; they
  plan with their own random numbers.

## Limitations

- **CueCard is a reimplementation.** Its code and the POOLFIZ simulator were
  never released. Our rebuild runs at about 1% of the original's simulation
  budget, and some parameters fill gaps the papers leave open.
- **One simulator.** All results are in
  [pooltool](https://github.com/ekiefl/pooltool). Transfer to other physics
  engines or real tables is untested.
- **Not learned from scratch.** The judge starts from EV's hand-written
  evaluation and learns a correction.
- **Some tournament rules are missing:** the four-balls-to-a-rail break, the
  8-ball as a legal first contact on an open table, and the game clock.
- **Small ablations.** The hand-written-evaluation comparison used only 200
  games.

## References

- Archibald, Altman, Greenspan, Shoham. [Analysis of a Winning Computational Billiards Player](https://www.ijcai.org/Proceedings/09/Papers/231.pdf). IJCAI 2009.
- Archibald, Altman, Greenspan, Shoham. [Computational Pool: A New Challenge for Game Theory Pragmatics](https://aaai.org/ojs/index.php/aimagazine/article/view/2312). AI Magazine, 2010.
- Smith. [Running the Table: An AI for Computer Billiards](https://cdn.aaai.org/AAAI/2006/AAAI06-156.pdf). AAAI 2006.
- Kiefl. [pooltool: a billiards simulator](https://theoj.org/joss-papers/joss.07301/10.21105.joss.07301.pdf). JOSS.

## License

MIT, see [LICENSE](LICENSE).
