# Research artifact contract

[The study](../docs/research-results.md) extends CritterLab with controlled
source-imitation/target-RL comparisons. The original v1 results and browser
actors are not replaced.

- `protocol.json`: machine-readable specification, committed before full training.
- `results.json.gz`: gzip-compressed UTF-8 JSON. Includes exact frozen maps,
  every final actor state, all greedy/sampled evaluation actions, per-seed
  curves, budget accounting, package versions and source digests.
- `analysis.json`: paired-seed effects and bootstrap intervals, computed from
  raw records without best-seed or checkpoint selection.

```sh
uv sync --locked
uv run python -m critterlab.research --output research/reproduced.json.gz
uv run python -m critterlab.research_audit research/reproduced.json.gz
uv run python -m critterlab.research_report research/reproduced.json.gz --output /tmp/critterlab-report
```

The report uses Matplotlib (a development dependency). CPU-only training is
sufficient. Durations are machine-specific. No paid service is involved.
An interrupted run saves complete method/seed blocks and is not marked
`complete`; the audit rejects missing or duplicate runs. The runner does not
resume a partial artifact: rerunning restarts the declared experiment.

CI replays the committed results and re-runs inference; it does not retrain the
full study. Local `uv run pytest -q` checks hand-calculated mathematical cases,
map identity separation, exact update accounting, and checkpoint round-trips.
Recorded outcomes are exact. Inference checks recorded states with a 1e-6
probability tolerance for greedy ties or sampled CDF boundaries across CPUs;
it does not replace recorded actions or outcomes with another rollout.
Regeneration can vary numerically across Torch/platform versions. Compare raw
outcomes and effects, not timing or compressed-file byte identity.

The source dataset and test panels are shared across model seeds. Intervals
condition on those panels. DAgger gets additional target expert labels; update
slots are not FLOPs. See the protocol and report before interpreting scores.
