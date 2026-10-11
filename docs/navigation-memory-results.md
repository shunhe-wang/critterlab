# Random-garden memory: fewer loops, lower success

**The GRU escapes the reported seed-67 wilder loop in 10 moves, but is not a
better default brain overall.** It has fewer loop timeouts and more hazard
entries than the current and newly retrained reactive brains. This negative
tradeoff is preserved; there was no post-test checkpoint selection.

[Play the comparison](https://shunhe-wang.github.io/critterlab/#navigation-lab).
“Try seed 67 · wilder” runs the current BC → PPO, recurrent GRU, and that
GRU reset every move on the same patch. The GRU is also in the normal garden's
brain selector, including on edited maps. The current brain remains the default.

## Protocol and measured scores

[Protocol r1](navigation-memory-protocol.md) and [exact-world splits](../results/navigation-r1-splits.json)
were fixed in commit [`657d04a`](https://github.com/shunhe-wang/critterlab/commit/657d04a)
before training or viewing new test outcomes. Training code and boundary checks
were committed before the full run. No budgets, hyperparameters or release
seeds were adjusted after results.

Success, mean ± sample standard deviation across initialization seeds 0/1/2:

| Actor | Standard, unseen exact worlds | Wilder, unseen exact worlds |
|---|---:|---:|
| Current frozen BC → PPO | **87.0 ± 1.6%** | **74.5 ± 1.2%** |
| New recurrent GRU | 77.9 ± 2.3% | 68.5 ± 1.6% |
| New reactive control | 88.7 ± 0.0% | 77.0 ± 1.8% |
| GRU reset every observation | 26.7 ± 2.2% | 19.8 ± 1.8% |

256 unique worlds per split per seed. Test worlds exclude all 23,084 unique
legacy training worlds, all new training and development worlds, and each
other. World identity includes cells, start and goal. These fresh-split scores
are not directly comparable to the older v1 table's different test cases.
The exclusions cover v1 and r1 actors evaluated here; the separately published
350-run imitation-under-shift study is outside this comparison.
Standard deviations describe model-seed variation, not confidence intervals.

Individual success percentages (seeds 0, 1, 2):

| Actor | Standard | Wilder |
|---|---|---|
| Current | 85.5, 88.7, 86.7 | 73.4, 74.2, 75.8 |
| GRU | 77.0, 76.2, 80.5 | 66.8, 68.8, 69.9 |
| Retrained reactive | 88.7, 88.7, 88.7 | 75.4, 78.9, 76.6 |
| GRU reset | 24.2, 27.3, 28.5 | 18.8, 21.9, 18.8 |

Equal reactive success counts are actual outcomes of distinct checkpoints;
inspect the committed raw episodes and checkpoint hashes.

## Did it fix loops, or just fail differently?

A short cycle means six consecutive positions alternating between at most two
tiles; stationary collisions count. **Loop timeout** also requires the 48-action
timeout. Temporary cycles are recorded separately: memory can return to a tile
and later choose a different action.

| Actor | Standard loop timeout | Wilder loop timeout | Standard hazard | Wilder hazard |
|---|---:|---:|---:|---:|
| Current | 13.0 ± 1.6% | 25.5 ± 1.2% | 0.0% | 0.0% |
| GRU | **7.9 ± 0.5%** | **16.4 ± 0.7%** | 12.1 ± 3.0% | 14.1 ± 2.1% |
| Retrained reactive | 11.3 ± 0.0% | 23.0 ± 1.8% | 0.0% | 0.0% |
| GRU reset | 68.0 ± 2.1% | 77.1 ± 2.9% | 4.8 ± 1.2% | 3.1 ± 1.2% |

The GRU recovers some old failures, enters hazards on others, and regresses on
previously successful worlds. Overall success is **9.1 percentage points lower**
than the current brain on standard terrain and **6.0 points lower** on wilder
terrain (means of paired seed differences). Fewer loops alone is not a win.

For every test world where the corresponding current seed ended in a loop:

| Split / seed | Current loop cases | GRU: success / hazard / timeout | Reactive: success / hazard / timeout |
|---|---:|---|---|
| Standard / 0 | 37 | 19 / 9 / 9 | 14 / 0 / 23 |
| Standard / 1 | 29 | 12 / 11 / 6 | 9 / 0 / 20 |
| Standard / 2 | 34 | 14 / 8 / 12 | 16 / 0 / 18 |
| Wilder / 0 | 68 | 27 / 19 / 22 | 20 / 0 / 48 |
| Wilder / 1 | 66 | 26 / 12 / 28 | 17 / 0 / 49 |
| Wilder / 2 | 62 | 23 / 11 / 28 | 14 / 0 / 48 |

These conditional counts are not aggregate success rates. World/seed cases
overlap across initialization seeds; pooled counts are not independent trials.
Resetting state also disrupts learned navigation dynamics, so it is not a
complete causal isolation of every architectural difference.

## Preselected diagnostics

The reported seed-67 wilder world was excluded from new training, development
and aggregate tests. All three GRUs succeed in 10 moves; all three reset-state
GRUs time out in 48. Current and retrained reactive actors time out for seed 0,
while their seeds 1/2 succeed in four moves. Demo seed 0 was fixed before
training, not chosen to make this example look good.

All normally carried actors succeed on standard seed 67. On standard seed
1,100,001, all three GRUs and reactive controls still time out. Known diagnostics
are illustrative, not evidence of held-out generalization.

## What was trained

Unchanged v1 world, rewards, horizon and **20 observation features**. No visit
counts, previous-action input, global map, pathfinder output, inference-time
reward or anti-loop rule reaches the actor.

The 5,316-parameter GRU and 5,764-parameter reactive control each receive the
original 256 BFS sequences (1,290 transitions): 30 masked BC epochs, 180 Adam
updates. Each then receives **65,536 PPO interactions**, 1,024 optimizer updates,
from the same frozen pool of 1,024 unique standard gardens. Actions and episode
resets change their actual experience. Equal budgets are not equal on-policy
data or compute. The frozen current model used v1's different pool and batching;
the newly retrained reactive actor is the architecture control.

GRU PPO uses 16-tick sequence minibatches, detached chunk-start states,
termination resets and updated-weight reconstruction of ongoing episode prefixes
between rollouts. Both new critics are reactive. The [protocol](navigation-memory-protocol.md)
specifies all optimization, discount, GAE, clipping and gradient budgets.

| Local training seconds, including development checks | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| GRU | 11.228 | 8.767 | 8.270 |
| Reactive | 4.747 | 3.759 | 3.925 |

The CPU run, including evaluation/exports, took 48.03 seconds. Timing is
descriptive, not a hardware benchmark. All six new final checkpoints, losses,
development curves, actual training map seeds, actions and metadata are in
[the raw result](../results/navigation-r1.json).

## Reproduce and audit

```sh
uv sync --locked
uv run python -m critterlab.navigation_train --publish
uv run python -m critterlab.navigation_audit
uv run pytest -q
npm ci
npm run check
```

`--publish` enforces the fixed budgets/seeds. Both demo actors are final **seed 0**.
`uv run python -m critterlab.navigation_splits` regenerates the precommitted split
without querying model scores. The independent audit checks **6,192 episodes**,
recomputing every action from full PyTorch observation prefixes independently
of the trainer's carried-step evaluation, plus outcomes, loops, conditional
recoveries, controls and hashes. Browser tests reproduce all seed-0 actions and
PyTorch state/probability fixtures. Save/replay tests reconstruct state at every
frame. Numeric differences across platforms remain possible.

This supports learned history-dependent navigation that resolves some cycles.
It does not establish a superior garden actor, a novel algorithm, production-scale
training or general memory. Original v1 and m1 artifacts are intact.
