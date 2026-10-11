# Pip model card

## Architecture and use

Actor: 20 inputs → 64 tanh → 64 tanh → 4 action logits, 5,764 learned
parameters. Four actions: north, east, south, west. The browser greedily selects
the largest probability, matching offline evaluation. PPO training samples
actions from a categorical distribution.

Inputs are the known goal's dx/dy divided by 7, followed by 3 × 3 rock indicators
and 3 × 3 hazard indicators. The actor gets no expert action, shortest path,
full terrain map, previous action, or recurrent state. Boundary cells are
impassable rocks. Goal success and hazard entry terminate the episode; the
48-action horizon also terminates the learning objective.

The JSON export preserves all actor dense weights and biases. Browser inference
uses double-precision JavaScript arithmetic; PyTorch uses float32. Committed
parity vectors compare probabilities within 1e-6. Checkpoints contain actor
state dictionaries, not resumable optimizer/critic state. Load them with
`torch.load(path, weights_only=True)` into `critterlab.model.Policy`.

## Training data and objective

256 deterministic BFS expert routes provide 1,290 observation/action pairs.
BC minimizes cross-entropy. PPO maximizes a clipped advantage objective with
GAE, a learned value function and entropy regularization. Exact hyperparameters
and map splits are in [the protocol](protocol.md).

Map geometry and reward definitions are synthetic. No personal, legal, or
production data is used. The full map is available to the expert, so identical
local observations can sometimes require different expert actions. BC cannot
perfectly resolve that missing-information problem.

## Intended use and boundaries

An educational, interactive study of policy learning. The learned behavior is
limited to this garden family. Pip does not understand language, explain its
reasoning, or learn persistent personality traits. The probabilities panel
shows the output distribution, not a verbal explanation or calibrated success
confidence. No external system actions are available.

Human teaching performs real supervised gradient updates on a browser-local
copy. It can overfit a route or harm previous behavior. It is not part of the
published offline controlled comparison. Refreshing clears the personal model;
exported replay JSON includes adapted weights for inspection.

## Distribution and reproducibility

MIT-licensed project checkpoints and JSON weights. Seed 0 final actors power
the demo; every training seed's final actor is committed. See [results](results.md)
for both seed-held-out and post-hoc exact-world-disjoint evaluation. No novel
algorithm, new foundation model, or general-intelligence capability is claimed.
