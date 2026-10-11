# Memory-grove model card

## Models and observations

Recurrent actor: 22 inputs → GRU with 32 hidden units → four action logits,
**5,508 trainable parameters**. Reactive comparison: 22 → 64 tanh → 64 tanh →
four logits, **5,892 parameters**. Both are trained from initialization. The
frozen original garden actor is a separate out-of-task transfer reference.

Inputs: public junction dx/dy, nine local rock flags, nine local hazard flags,
and two cue indicators. The desired berry, previous action, elapsed time and
gate timer are absent. Cue indicators are zero after tick 0. Hidden state
starts at zero and carries across actions within one trial. Action selection
is greedy, matching evaluation. See [protocol m1](memory-protocol.md).

The browser implements PyTorch's reset/update/new GRU gates, including its
placement of the recurrent new-gate bias. Exported weights and carried state
trajectories are checked against PyTorch to a 2e-6 numeric tolerance. Activation
colors are raw hidden-state values, not explanations or confidence scores.

## Training, evaluation and distribution

36 synthetic expert sequences, 330 transitions; supervised weighted
cross-entropy over 200 Adam updates, all recurrent weights trained with full
sequence backpropagation. No personal data, API model, or reward fine-tuning.
The [results](memory-results.md) include three seeds, geometry-disjoint tests,
long-delay tests and erase/swap/reset controls. Long-delay performance varies
substantially across seeds. Same data/updates are not equal compute.

All six final PyTorch state dictionaries and both seed-0 JSON policies are
distributed with this MIT-licensed project. Demo seed 0 was declared before
training, not selected by score. Load checkpoints with `weights_only=True`
into `critterlab.memory_model.Recurrent` or `Reactive`; optimizer state is not
included. Artifacts and raw actions are independently audited.

## Intended use and boundaries

Educational experiments and play in this small T-shaped task family. This
actor does not converse, remember personal experiences, or solve arbitrary
random gardens. It has no external actions or access to your saved diary. Pet
care, dialogue and diary persistence are authored game rules; browser teaching
updates the original garden actor separately, not these memory checkpoints.
Trials reset recurrent state and are not resumed across page refreshes.
