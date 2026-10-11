# Three actual policy failures

These are recorded behavioral failures, not fabricated software bugs. Recreate
them in the demo using the garden seed and selected brain. The committed demo
policies are final seed 0. Inspect the action probabilities and episode journal.

| Brain | Garden seed | Observed behavior | Outcome |
|---|---:|---|---|
| Untrained | 1100000 | Chooses west 48 times, including collisions | Timeout |
| Demonstrations only | 1100001 | Chooses north 48 times, including collisions | Timeout |
| Demonstrations + rewards | 1100001 | Alternates south/north for all 48 actions | Timeout |

The raw action sequences live in `results/experiment-v1.json`, under seed 0's
`evaluation.held_out.episodes`. The audit replays them through the environment
and verifies the outcomes.

## What these suggest

An untrained network's greedy bias is not a random-action baseline. Imitation
can reproduce useful expert actions yet get stuck in states where local
information is insufficient. Reward learning reduces many failures but still
allows deterministic two-state cycles. The last explanation is a hypothesis
about the mechanism, not an established causal result. A recurrent-policy
comparison is a useful next experiment; it has not been run here.

The current release exposes these failures instead of adding a hidden
pathfinding fallback. The separate solver only supplies offline demonstrations
and checks whether edited worlds have a safe route.
