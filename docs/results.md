# Field notes: experiment v1

## Question and result

Does demonstration initialization improve garden navigation at the same
subsequent PPO environment-interaction budget? Under this small setup,
BC → PPO has higher average goal success than PPO alone. It also starts with
more supervision and optimization. The experiment does not isolate total
compute, data quantity, expert information, or a new algorithmic contribution.

| Condition | Seed 0 standard | Seed 1 standard | Seed 2 standard | Mean ± SD |
|---|---:|---:|---:|---:|
| BC | 79.69% | 79.30% | 74.61% | 77.86 ± 2.83% |
| PPO | 77.73% | 77.73% | 74.22% | 76.56 ± 2.03% |
| BC → PPO | 87.89% | 85.55% | 86.33% | 86.59 ± 1.19% |

Standard and wilder splits each contain 256 seed-held-out episodes. The
untrained greedy network succeeds in none. A seeded random-action reference
succeeds in 28.91% standard and 20.31% wilder episodes. A BFS expert with the
full map succeeds in all episodes and takes about 4.95/5.21 steps on average.
These references have different information and action-selection mechanisms.

On denser wilder terrain, mean success drops to 62.76% (BC), 64.32% (PPO), and
76.43% (BC → PPO). Failure to transfer perfectly is part of the result.

## Seed separation was insufficient for exact-world separation

The protocol fixed disjoint **random seeds** before the first run. A later audit
compared exact terrain, start, and goal tuples. Small gardens can recur under
different seeds. PPO had 3–7 exact overlaps per standard test split; BC → PPO
had 6–8. One BC → PPO seed had one overlap on the wilder split. BC-only had no
evaluation overlap; its 256 demonstrations contained 255 unique worlds.

We retained the protocol's original results and constructed one **post-hoc
sensitivity analysis**: remove from every condition's evaluation any test world
identical to any training world across all conditions and seeds. That excludes
13 standard seeds and 1 wilder seed, leaving the same 243/255 worlds for every
method. No retraining, hyperparameter changes, or checkpoint selection is
involved. This audit was chosen after seeing the original results and must not
be presented as a predeclared confirmatory result.

| Condition | World-disjoint standard | World-disjoint wilder |
|---|---:|---:|
| BC | 77.23 ± 3.09% | 62.61 ± 2.16% |
| PPO | 75.31 ± 2.14% | 64.18 ± 0.82% |
| BC → PPO | 85.87 ± 1.26% | 76.34 ± 2.40% |

The ordering survives this audit. It still does not establish generalization to
arbitrary tasks, unseen local observation patterns, or equivalent-budget
learning efficiency. The common subset excludes exact whole-world copies;
local observation/action patterns can overlap across distinct worlds.

Adding training-map tracing changed no training behavior: every per-seed
success summary matched the original run exactly. The published run includes
this instrumentation so the overlap audit is independently reproducible.

## Evidence and accounting

- [Full run records](../results/experiment-v1.json): all final evaluation actions,
  development curves, training map seeds, optimization counts, versions, timing,
  and source SHA-256 hashes.
- [Exact overlaps](../results/world-overlap.json): per-condition counts and seeds.
- [Common-subset audit](../results/disjoint-audit.json): excluded seeds and all scores.
- [Checkpoints](../checkpoints/): seed 0, 1, 2 actors for each of four conditions.
- [Browser policies](../public/models/): final seed 0 actors, fixed before results.
- [Protocol](protocol.md): specification written before the first experiment;
  not registered with an external preregistration service.

256 expert episodes contain 1,290 demonstrated transitions. BC uses 180
minibatch updates per seed/condition. PPO uses 1,024 updates and 65,536
environment interactions per seed/condition. Across both PPO conditions and
all three seeds, that is 393,216 environment interactions. The same
demonstrations are reused across paired seeds. The actor has 5,764 parameters;
the PPO critic is separate and is not used in the playable demo.

Training is intentionally small enough for CPU reproduction. Timings in the
run file describe this machine and are not a cross-machine speed guarantee.

## Limitations worth testing next

1. **Memory:** local observations can alias different situations. A recurrent
   agent may escape oscillations that a deterministic reactive policy cannot.
2. **Reward shaping:** goal-direction shaping makes exploration easier. A
   sparse-reward experiment would test whether demonstrations help without it.
3. **Demonstration source:** BFS demonstrations are optimal and know the whole
   map. Human routes could be noisier, longer, and inconsistent.
4. **Evaluation size:** three seeds and one small task family give descriptive
   evidence. The standard deviations are not confidence intervals.
5. **Selection:** release demos use final seed 0. No best-seed or best-checkpoint
   selection. Development curves are visible; final evaluations were not used
   to tune the first experiment.
