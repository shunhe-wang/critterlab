# Vanishing-clue experiment — protocol m1

Fixed before training or inspecting test results. This is a small supervised
sequence-learning experiment, independent of navigation protocol v1.

## Question and environment

Can a recurrent actor remember a one-tick cue, when a reactive actor receives
the same current observation at the later fork?

A 7×7 T-shaped garden has a vertical stem from row 5 to the junction and two
identical berry destinations at columns 1 and 5 of the junction row. A blue cue
means left; a gold cue means right. It appears only at observation tick 0.
For `delay` actions the gate holds the creature at its start, irrespective of
the chosen action. The gate then opens. This prevents encoding the initial cue
in the creature's position. Arrival at either berry ends the episode: correct
side is success, the other is wrong-berry. The horizon is 96 actions.

Observation: the original 20 navigation features, pointing toward the **public
junction**, followed by two cue indicators. No target displacement, desired
side, action history, elapsed-time feature or remaining-delay feature reaches
the actor. After tick 0, the cue indicators are both zero. Hidden state starts
at zero for every episode. Every layout/delay is evaluated with both cues.
A deterministic reactive policy follows identical trajectories for those paired
episodes, so its balanced success ceiling is 50% (possibly lower if it times out).

## Data fixed by geometry

- Training: junction columns 2, 3, 4; junction rows 2, 3; delays 2, 4, 8;
  both cues. Exactly 36 unique expert episodes; no random duplicated map seeds.
- Development: the six training layouts, delay 6, both cues. Used only for
  plotted progress; no checkpoint selection or hyperparameter adjustment.
- Test: junction row 1, columns 2, 3, 4; delays 2, 4, 8; both cues (18 episodes).
- Longer-delay test: the same disjoint test layouts, delays 16, 32; both cues
  (12 episodes). Test geometry never appears in training or development.
- Expert: North while waiting and on the stem, then West for blue or East for
  gold until reaching the corresponding berry. The expert knows the cue;
  the learned actor gets only the observation above.

## Models and optimization

Three training seeds: 0, 1, 2. CPU, one Torch thread, deterministic algorithms.

- Reactive: 22 → 64 tanh → 64 tanh → 4 logits (5,892 parameters). Same hidden
  widths as the current brain, retrained on this task with the cue inputs.
- Recurrent: 22 → GRU(32) → 4 logits (5,508 parameters). All recurrent weights
  are trained; the browser implements PyTorch's reset/update/new gate equations.
- Current brain: frozen BC→PPO actors from navigation v1 for each of its three
  original seeds. Drops the two unsupported cue inputs. An out-of-task transfer
  reference, not the controlled architecture comparison.

The two newly trained models get exactly the same 36 sequences, 200 full-batch
Adam updates (learning rate 0.003), and teacher-forced state observations.
Cross-entropy uses valid sequence steps; the first fork decision has weight 4,
all other valid steps weight 1. Padding has weight 0. GRU gradients propagate
through the full sequence. Gradient norm clips at 1. No reward fine-tuning,
extra data, scheduled curriculum, best-seed selection or post-test tuning.
Equal updates/data do not imply equal compute; report measured training times.

## Evaluation, controls and release

Greedy action selection, closed-loop environment evaluation (not teacher-forced
accuracy). Evaluate both test splits for all seeds and report mean ± sample
standard deviation over model seeds, with the individual values. These are
exhaustive finite task cases, not independent samples from a broad population.

On the 18-episode standard test also evaluate recurrent controls:

1. Erase cue inputs at tick 0.
2. Swap the cue indicators while keeping the true destination unchanged.
3. Reset recurrent hidden state before every observation.

Commit all six newly trained checkpoints, frozen raw per-episode action records,
per-seed losses/development curves, source/checkpoint hashes, timing and package
versions. Browser uses final **seed 0**, irrespective of its score. Generate
Python/browser world and recurrent-state/probability parity fixtures.
An independent audit replays episodes, recomputes neural actions and summaries,
checks exact geometry separation and verifies artifact hashes. Preserve negative
and long-delay results. A memory-task success does not establish that this
actor solves the original random gardens or has autobiographical understanding.
