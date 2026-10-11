# Actual policy failures

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
about the mechanism, not an established causal result. A separate
[recurrent cue-memory comparison](../docs/memory-results.md) has now been run.
It does not train on these random gardens or fix these cycles.

## Reported seed 67, wilder terrain

The untrained seed-0 actor alternates East/West between tiles 18 and 19 until
the 48-action timeout. Its repeated probabilities prefer East at tile 18
(about 26.4%) and West at tile 19 (about 26.6%). Greedy reactive inference has
no previous-action or visit memory: returning to the same state repeats the
choice. The trained seed-0 actors also time out here, in a North/South cycle.
The UI flags repeated tiles and leaves the network's behavior intact.

The r1 random-garden GRU now reaches the berry in **10 moves** on this wilder
patch for all three model seeds. Resetting its recurrent state every move brings
back the 48-action timeout. This is a known diagnostic, excluded from the held-out
aggregate. The new reactive seed-0 control also times out. Seed **1100001** still
times out for all three trained architectures and every model seed.

Across unseen gardens the GRU has fewer repeating-tile timeouts but more hazard
failures and lower overall success. See the
[complete recurrent navigation report](../docs/navigation-memory-results.md)
for per-seed results and all outcomes on the current actor's loop cases.

The current release exposes these failures instead of adding a hidden
pathfinding fallback. The separate solver only supplies offline demonstrations
and checks whether edited worlds have a safe route.
