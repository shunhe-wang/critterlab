# A clue fades. Does the memory stay?

Experiment **m1** trains a recurrent actor to remember a one-tick blue or gold
clue. At a later fork, the two berries look identical. The network's cue inputs
are zero after the first observation. A locked starting gate prevents storing
the clue by moving left or right immediately.

The [protocol](memory-protocol.md) was committed as
[`2b6d1e0`](https://github.com/shunhe-wang/critterlab/commit/2b6d1e0)
before training and inspecting test outcomes. Hyperparameters and the final
seed-0 demo export were fixed in advance. No test-driven retraining or
best-checkpoint selection was used.

## Closed-loop results

Success percentages, mean ± **sample standard deviation across three model
seeds**. Parentheses list seeds 0, 1 and 2 individually.

| Actor | Unseen geometry, delays 2/4/8 | Unseen geometry, delays 16/32 |
|---|---:|---:|
| Recurrent GRU, 5,508 parameters | **88.9 ± 19.2%** (100, 66.7, 100) | **66.7 ± 28.9%** (100, 50, 50) |
| Retrained reactive, 5,892 parameters | 50.0 ± 0.0% (50, 50, 50) | 50.0 ± 0.0% (50, 50, 50) |
| Frozen original BC → PPO, 5,764 parameters | 16.7 ± 0.0% (16.7, 16.7, 16.7) | 16.7 ± 0.0% (16.7, 16.7, 16.7) |

Each seed gets 18 standard and 12 long-delay cases: three unseen layouts,
each delay, and **both clues**. These are exhaustive cases in a small finite
task family, not independent samples supporting a broad population estimate.
The ± values are not confidence intervals. Test junction row 1 never appears
in training or development.

The meaningful architecture comparison is GRU versus the reactive actor
**retrained on the same task and data**. The frozen original brain has no cue
inputs and was trained in a different garden; its score is a transfer reference.
Deterministic reactive actors receive identical observations for paired clues
after the first held action, so their balanced success ceiling here is 50%.
A timeout can make their score lower.

## Does the actor actually use the clue?

Standard-test interventions on the GRU, across seeds 0/1/2:

| Intervention | Success | Individual seeds |
|---|---:|---|
| Normal input and carried state | 88.9 ± 19.2% | 100, 66.7, 100 |
| Erase the initial cue | 50.0 ± 0.0% | 50, 50, 50 |
| Swap blue/gold network inputs, keep the real target | 0.0 ± 0.0% | 0, 0, 0 |
| Reset hidden state before every observation | 11.1 ± 19.2% | 0, 33.3, 0 |

Erase and swap establish that the original cue causally changes choices.
Resetting disrupts both remembering and recurrent navigation; its score is
not a pure measurement of one stored bit. Hidden-state colors in the demo
show actual activations, not thoughts or human-readable explanations.

## Training actually performed

- 36 unique expert sequences, **330 transitions**: six layouts × three delays
  × two clues. Development uses these layouts with delay 6.
- Both new architectures: 200 full-batch Adam updates at learning rate 0.003,
  supervised cross-entropy, first fork decision weighted 4, other valid steps
  weighted 1, padding weighted 0, gradient norm clipped at 1.
- Full sequence backpropagation for the GRU. No reward fine-tuning, language
  model, external API, or hand-written cue-to-action branch in neural play.
- Three paired initialization seeds, CPU, one Torch thread, deterministic
  algorithms. Same data and update count **do not mean equal compute**.

| Training seconds | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Reactive | 0.420 | 0.278 | 0.255 |
| GRU | 0.749 | 0.818 | 0.650 |

Times include development evaluations during training; these local wall times
are not a performance benchmark. The complete run, including test evaluation
and exports, took 4.19 seconds. Package/platform metadata, losses, development
curves, all actions, hashes and all six final checkpoints are committed in
[the raw result](../results/memory-m1.json).

## What failed and what this does not establish

Seed 1 achieves only 12/18 standard successes. Seeds 1 and 2 each succeed on
only 6/12 long-delay cases. Longer waits were absent from training, and
performance is sensitive to initialization. The demo uses seed 0 because the
protocol declared it beforehand; its 100% long-delay result does **not**
represent the aggregate result.

This is learned short-term cue memory in a deliberately constrained synthetic
world. It is not autobiographical memory, conversational understanding, or a
solution to navigation loops in the original random gardens. Pet diaries and
care are separately persisted game rules. Trials start with zero recurrent
state; completed trials can leave an ordinary entry in your pet's diary.

## Play and reproduce

[Open the memory grove](https://shunhe-wang.github.io/critterlab/#memory-lab).
Watch each actual seed-0 checkpoint, play yourself, change the waiting delay,
erase the cue, reset state every tick, and download the trial's action record.
The browser carries the same GRU state and gate equations as PyTorch.

```sh
uv sync --locked
uv run python -m critterlab.memory_train --publish
uv run python -m critterlab.memory_audit
uv run pytest -q
npm ci
npm run check
```

Publishing enforces the frozen budget and seeds and replaces the m1 artifacts.
For a smoke run that preserves release files:

```sh
uv run python -m critterlab.memory_train --epochs 2 --seeds 0 --output /tmp/critterlab-memory-smoke.json
```

The independent audit replays **432 episodes**, recomputes each neural action,
all controls and summaries, checks geometry separation, and verifies source,
checkpoint and export hashes. Browser tests compare carried recurrent states
and probabilities against PyTorch fixtures and reproduce published seed-0
outcomes. Numeric differences across platforms remain possible.

See the [memory model card](memory-model-card.md) for distribution and boundaries.
