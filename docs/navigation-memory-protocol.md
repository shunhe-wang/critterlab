# Random-garden memory — protocol r1

Fixed before training or viewing new test outcomes. Question: does a recurrent
actor reduce greedy navigation loops in the original random gardens?

## Environment and actors

Unchanged protocol-v1 7×7 gardens, four actions, 48-action horizon, rewards,
and **20 inputs**: known goal displacement plus local 3×3 rocks and hazards.
No visit counts, previous-action input, full map, solver fallback, or loop
escape rule enters either learned actor.

- GRU: 20 → GRU(32) → 4 logits, 5,316 actor parameters. Zero initial hidden
  state, carried across observations, reset at episode boundaries.
- Retrained reactive control: the original 20 → 64 tanh → 64 tanh → 4
  architecture, 5,764 parameters. Same new training regimen as the GRU.
- Current brain reference: frozen v1 BC → PPO checkpoints for seeds 0/1/2.
  Historical training differs; it is not the controlled architecture comparison.

## Data and optimization

The original 256 standard-garden BFS demonstration episodes (seeds 10,000–10,255)
are reconstructed as sequences. Both new actors receive 30 epochs, shuffled
episode batches of 48, Adam 0.001, masked mean cross-entropy on valid steps,
180 updates. GRU backpropagation spans complete expert sequences. Padding has
zero loss weight. Reactive gradients use the same sequence batches and mask.

Then both new actors get **65,536 PPO environment interactions**, 16 parallel
environments × 128 ticks per rollout, four optimization epochs, minibatches of
16 sequences × 16 ticks (256 transitions), 1,024 optimizer updates. Adam
0.0003, epsilon 1e-5, gamma 0.99, GAE lambda 0.95, clip 0.2, entropy 0.01,
value coefficient 0.5, gradient norm cap 0.5. Both critics use the original
feed-forward 20 → 64 → 64 → 1 architecture. PPO training samples actions;
evaluation and browser inference are greedy.

GRU PPO uses truncated sequence backpropagation over 16 ticks, with detached
behavior-policy states at chunk starts and explicit resets after terminal
actions. No shuffled individual timesteps for the recurrent actor. At the next
rollout, carried states are reconstructed from each ongoing episode's actual
observation prefix using updated weights, preventing stale cross-update state.
Bootstrap/GAE never cross an episode termination. Horizon termination follows
v1's terminal objective, not an infinite-horizon truncation bootstrap.

PPO samples from a precommitted pool of **1,024 unique standard gardens**,
chosen by increasing seeds starting at 30,000. Original v1 sampled a much larger
seed range. New actors share the pool and interaction/update budgets, but
actions change their experienced trajectories and reset frequency. Neither
equal compute nor identical on-policy data is claimed. Report actual map seeds,
update counts, timing, BC transitions, curves and all final checkpoints.

## Frozen splits

The split builder first gathers exact worlds (cells, start, goal) from every
published v1 training condition/seed and all demonstrations. Exclude seed 67's
standard and wilder worlds from the new PPO pool. Development: 64 unique
standard worlds, increasing seeds from 50,000, excluding all old/new training.
Standard test: 256 unique worlds from seed 3,100,000 upward. Wilder test: 256
unique worlds from seed 4,100,000 upward. Exclude all old/new training,
development, and the explicitly diagnostic seed-67 worlds, and prevent exact
duplicates across either test split. Freeze selected seeds and world hashes
in `results/navigation-r1-splits.json` **before model training**.

Check that the original BC demonstrations do not contain either seed-67 world.
If that assertion fails, stop and declare a revised protocol before training.
The exclusion concerns exact worlds with their start/goal, not every possible
task sharing the same terrain. Split selection never queries model outcomes.

## Evaluation and diagnostics

Three seeds 0/1/2, CPU, one Torch thread, deterministic algorithms. Development
is measured after BC and every 8,192 PPO steps; never used to tune budgets,
hyperparameters, or select checkpoints. Publish **final seed 0** for both new
demo brains, even if another seed performs better.

On both fixed test splits evaluate GRU, retrained reactive, frozen current brain,
and GRU with hidden state reset before every observation. For each episode
record all actions, outcome, steps and repeated-position behavior. A short
cycle is six consecutive positions alternating between at most two tiles
(including a stationary collision). Report success/hazard/timeout rates,
successful path length, any-short-cycle rate, and **loop-timeout rate** (timeout
with that cycle in the final six positions). A temporary cycle is not automatically
a failure: memory may escape it later.

Report paired success-rate changes and recovery of the current brain's
loop-timeout cases on exactly the same worlds. Show denominators and outcomes,
not only the successful recoveries. Mean ± sample standard deviation across
three model seeds, with each seed's values; no significance claim.

Seed **67**, both terrain settings, is a preselected qualitative diagnostic,
not part of held-out aggregate scores. Also publish original known failures
1100000 and 1100001 on standard terrain as diagnostics. They never guide
training, curriculum, or checkpoint choice. Preserve negative findings. The
reset-state control can disrupt navigation as well as memory and is not a
perfect causal separation of every architectural effect.

## Release checks

Commit all six new actor checkpoints, raw actions, source/checkpoint/export
hashes and metadata. Independently recompute every final neural action and
outcome; verify disjointness, controls, paired summaries and hashes. Generate
full Python/browser recurrent-state and probability fixtures from actual
trajectories. Tests use world contracts, analytic GAE examples and PyTorch
sequence equivalence; no required score floor. Browser state reconstruction
must agree after replay/restore. Existing v1 and m1 artifacts remain intact.
