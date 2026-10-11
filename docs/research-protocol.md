# When does imitation help or hurt adaptation?

Protocol `imitation-shift-v1`, specified before its first run. This is a local,
versioned protocol, not an externally registered preregistration. The original
experiment and browser policies remain separate. Code and investigation are
developed with AI assistance; this artifact does not establish unaided research
independence. Numerical results must come from executed runs.

## Questions and possible falsifiers

Does source-distribution BC improve PPO adaptation on a new map family when
environment interactions and actor optimizer-update slots are both fixed?
Does that benefit change with topology, and can an entropy intervention explain
any negative transfer? A positive or negative result is useful. No surprising
result, method advantage, or novelty is assumed.

Primary contrast: final greedy success of BC→PPO minus PPO, separately on each
training family's held-out worlds. Secondary contrasts: BC→PPO minus PPO-short
(same subsequent PPO updates); entropy intervention difference-in-differences
`(BC→PPO-entropy − BC→PPO) − (PPO-entropy − PPO)`; adaptive minus fixed auxiliary
imitation. Also publish stochastic-policy success, hazards, timeouts, and
cross-family evaluation. The latter measures zero-shot transfer; training on
shifted terrain measures adaptation. Do not conflate them.

Predictions to test, not statements of fact:

1. Source BC can help standard terrain but lose its benefit on barrier terrain.
2. If reduced exploration contributes, BC starts with lower entropy on common
   probe observations and an entropy intervention improves adaptation. A generic
   PPO entropy benefit alone does not isolate a BC-specific mechanism.
3. Gating imitation by source support and policy entropy can reduce negative
   transfer relative to a fixed imitation penalty. This is a heuristic using
   established ideas, not a claimed new algorithm.

If all intervals remain compatible with zero, report uncertainty. If BC keeps
helping, report that. An entropy association alone is not causal evidence;
interventions also change the optimization objective and visited states.

## Environment and split construction

Retain the v1 7×7 garden, 20 local inputs, four actions, 48-step horizon, rewards,
and feedforward tanh actor. Standard and dense families use v1 generators.
Barrier maps add a vertical interior wall with one random gap, force the start
and goal onto opposite sides, and retain random rocks/hazards. This is a new
topology, not a pure scalar complexity variable. Solvability filtering changes
path distributions; publish shortest-path and obstacle descriptors.

Freeze 256 source demonstration worlds, 256 PPO training worlds per family, and
128 test worlds per family. Reject exact duplicate terrain/start/goal tuples
globally while building the manifest, before optimization. Train and test
worlds are therefore disjoint by identity, not just seed. All methods use the
same pools and paired initial actor/critic weights. Reset timing can change
with policy behavior, so realized visited maps and interactions can differ.
These are finite procedural pools, not an open-ended benchmark. Local patterns
may recur across disjoint maps. The full-map BFS expert has information the
reactive actor lacks; aliasing, tie-breaking and representation capacity can
explain errors independently of entropy.

## Budgets and methods

The executable specification is [research/protocol.json](../research/protocol.json).
Use ten independent paired model/training seeds, 65,536 environment steps,
32 rollouts of 2,048 transitions, minibatches of 256, and final checkpoints only.
PPO learning rate .0003, clip .2, discount .99, GAE .95, value coefficient .5,
gradient cap .5; BC Adam learning rate .001. Default actor width 64.

- PPO: 1,024 actor/critic updates, entropy coefficient .01.
- BC→PPO: 128 source CE updates, then 896 PPO updates (28 per rollout).
- PPO-short: 896 PPO updates, to isolate initialization at the same subsequent
  PPO budget. It is not the equal-total-update primary baseline.
- BC: 1,024 source CE updates, no environment interaction. It is a supervision
  reference, not an equal-interaction learner.
- PPO-entropy and BC→PPO-entropy: corresponding budgets, entropy coefficient .03.
- Fixed imitation: BC→PPO plus source CE weighted .1 inside each PPO update.
- Adaptive imitation: identical to fixed except weight `.1 × support × H/log(4)`
  per rollout. Support is the fraction of visited local rock/hazard bit patterns
  present in demonstrations; H is rollout mean categorical entropy. Goal
  displacement is deliberately excluded from this coarse support proxy. This
  is not a calibrated distribution-shift or uncertainty estimator. Publish the
  weights; fixed versus adaptive isolates the gate, not total supervised loss.
- DAgger-style baseline: 128 BC updates, then 896 CE updates on an aggregate
  dataset, with stochastic learner rollouts on the target family. Query BFS
  labels for sampled visited nonterminal states, up to the initial source
  transition count in additional queries. States with no safe route are
  skipped and counted. This simplified variant has extra expert supervision;
  it is a stronger-supervision reference, not a label-budget-matched comparison.

All RL/DAgger methods get exactly the same environment-interaction ceiling.
Actor optimizer slots are matched for primary PPO versus BC→PPO, entropy, fixed,
adaptive and DAgger comparisons. Slots are not FLOPs: CE and actor+critic PPO
have different compute. Source-label presentations, new expert queries,
critic updates, map exposures and wall time are reported separately. There is
no claim of equal total information, compute, or human annotation cost.

## Ablations and measurement

On barrier adaptation only, rerun PPO and BC→PPO under four one-factor changes:
64 demonstration worlds; 20% independently corrupted source labels (uniformly
choose one of the other actions); no Manhattan shaping (retain step/collision
costs and terminal rewards); width 32 for actor and critic. Corrupted labels
are an action-noise intervention, not a realistic model of human expertise.
BC update count is fixed when dataset volume changes, so label reuse changes.
These exploratory ablations can suggest follow-up work; they are not a full
factorial design or a test of every interaction.

Log entropy on source demonstrations and a fixed probe set of target training
world start observations as well as on-policy rollout entropy, unique visited
world/position pairs, and collisions. Fixed probes separate changing policy
distributions from changing state occupancy, but do not cover all states.
Evaluate greedy and sampled policies on all frozen test families. Sampling
uses a separate evaluation RNG and does not perturb training randomness.
Store actions, outcomes, maps, final actor weights and source digests so outcomes
and inference can be replayed. No checkpoint or seed is selected by test score.

## Analysis and limits

For each primary/secondary contrast, bootstrap the **paired training-seed
differences** (10,000 resamples, fixed seed), reporting mean and 95% percentile
interval. The unit is the independently trained seed; do not treat episodes
as independent models. Intervals condition on the frozen evaluation worlds and
do not quantify uncertainty over arbitrary map populations. No multiplicity
correction: label secondary comparisons and ablations exploratory. Ten seeds
give limited tail information; these are modest CPU navigation experiments.

Predeclared figures: per-family paired effects with uncertainty; fixed-probe
entropy trajectories; greedy versus stochastic evaluation. Preserve failures
and null findings. Defer any claim of a BC-specific exploration mechanism until
the entropy intervention, PPO control and occupancy diagnostics align; even
then describe evidence as suggestive. Method originality requires broader
baselines, independent reproduction and evaluation beyond one toy environment.
