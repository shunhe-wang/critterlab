# Experiment protocol v1

Question: does behavior-cloning initialization improve garden navigation at a
fixed subsequent reinforcement-learning interaction budget?

This protocol is written before the experiment. It is a small methods/demo
experiment, not a novel algorithm or a claim about general intelligence.

## World

7 × 7 garden. Grass, impassable rock, terminal hazard, one strawberry goal.
Actions in order: north/east/south/west. Limit: 48 actions. Observation: goal
dx/dy divided by 7, then 9 rock indicators and 9 hazard indicators in the 3 × 3
neighborhood, row-major. The goal location is known. There is no BFS distance,
expert action, or global map in the observation. This is a reactive policy;
ambiguous local observations can require memory it does not have.

Reward: successful arrival +1, hazard −1, ordinary step −0.02, rock collision
−0.08. Successful ordinary moves also receive 0.1 × reduction in Manhattan
distance. This dense shaping provides directional information, not a safe path.
Evaluation uses actual goal success, not the shaped training reward.

## Conditions and accounting

- Same 20 → 64 → 64 → 4 tanh actor and matched initial weights for each seed.
- Random initial network (untrained reference).
- Behavioral cloning (BC) from 256 BFS-generated demonstration episodes.
- PPO from random initialization.
- BC initialization followed by PPO (BC + PPO).
- 65,536 PPO environment interactions per condition per training seed.
- 3 training seeds: 0, 1, 2. CPU execution, no paid APIs.
- PPO: 16 environments × 128 steps per rollout, 4 update epochs, minibatch 256,
  learning rate 0.0003, discount 0.99, GAE lambda 0.95, clip 0.2, entropy
  coefficient 0.01, value coefficient 0.5, gradient norm cap 0.5.
- BC: 30 epochs, batch 256, Adam learning rate 0.001.
- BC + PPO has **extra demonstrations, supervision and optimization**. A fixed
  PPO budget is not a fixed total compute or data budget. Report demonstration
  transitions, elapsed time and update counts separately. No equal-total-budget
  or causal sample-efficiency claim is supported by this comparison.

## Splits and evaluation

Demonstration map seeds: 10,000 through 10,255 (training distribution).
PPO map seeds: 500,000 through 999,999, sampled using seed-specific RNG.
Development evaluation seeds: 1,000,000 through 1,000,063, every 8,192 steps.
Final held-out evaluation: 1,100,000 through 1,100,255.
Shifted final evaluation: 2,100,000 through 2,100,255, denser rocks/hazards.
Final sets do not inform checkpoint selection or hyperparameters.

Report greedy success rate, hazards, timeouts, successful path length and full
episode records. Show mean and standard deviation across seeds (descriptive,
not a confidence interval). Include a deterministic BFS expert upper reference
and a seeded random-action reference. Demo always uses **seed 0 final** models,
not the best seed/checkpoint. Do not modify the protocol after viewing results
without marking a new exploratory experiment.

## Behavioral contracts under test

Test world rules using hand-worked maps and exact outcomes; deterministic map
generation and disjoint splits; Python/browser observation and transition
agreement; exported policy/browser probability agreement within 1e-6. Training
smoke checks must show weights change and finite outputs, not assert an arbitrary
performance floor. Published metrics must be recomputable from episode records.
