"""Recompute published outcomes, metrics and seed summaries from action records."""
import hashlib
import json
import statistics
from pathlib import Path

from .world import World, generate

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT / "results/experiment-v1.json").read_text())
    checked = 0
    groups = [(r["evaluation"], r["method"]) for r in result["runs"]]
    groups += [({split: b[name]}, name) for split, b in result["baselines"].items() for name in b]
    for evaluations, name in groups:
        for split, evaluation in evaluations.items():
            records = evaluation["episodes"]
            for record in records:
                world = World(generate(record["map_seed"], split == "shifted"))
                for action in record["actions"]:
                    world.step(action)
                assert world.outcome == record["outcome"], (name, record["map_seed"])
                assert world.steps == record["steps"]
                checked += 1
            metric = evaluation["metrics"]
            for outcome, key in [("success", "success_rate"), ("hazard", "hazard_rate"), ("timeout", "timeout_rate")]:
                observed = sum(r["outcome"] == outcome for r in records) / len(records)
                assert abs(observed - metric[key]) < 1e-12
            lengths = [r["steps"] for r in records if r["outcome"] == "success"]
            assert metric["episodes"] == len(records)
            if lengths:
                assert abs(statistics.mean(lengths) - metric["mean_success_steps"]) < 1e-12
    for name, splits in result["summary"].items():
        for split, summary in splits.items():
            values = [r["evaluation"][split]["metrics"]["success_rate"] for r in result["runs"] if r["method"] == name]
            assert summary["values"] == values
            assert abs(summary["mean"] - statistics.mean(values)) < 1e-12
            assert abs(summary["std"] - statistics.stdev(values)) < 1e-12
    for path, digest in result["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    print(f"Audited {checked} replayed episodes, metric summaries and experiment source hashes.")


if __name__ == "__main__":
    main()
