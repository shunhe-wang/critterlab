"""Report exact world overlap despite disjoint RNG seeds (no scores excluded)."""
import json
import statistics
from functools import lru_cache
from pathlib import Path
from critterlab.world import generate

@lru_cache(maxsize=None)
def key(seed, shifted=False):
    g = generate(seed, shifted)
    return tuple(g.cells) + (g.start, g.goal)

result = json.loads(Path("results/experiment-v1.json").read_text())
report = []
all_training = set()
for run in result["runs"]:
    training = set(key(s) for s in run["training_map_seeds"])
    if run["method"] in ("bc", "bc-ppo"):
        training.update(key(s) for s in range(10000, 10256))
    all_training.update(training)
    overlap = {}
    for split, evaluation in run["evaluation"].items():
        overlap[split] = [r["map_seed"] for r in evaluation["episodes"] if key(r["map_seed"], split == "shifted") in training]
    report.append({"seed": run["seed"], "method": run["method"], "unique_training_worlds": len(training), "exact_overlap": overlap})
Path("results/world-overlap.json").write_text(json.dumps(report, indent=2) + "\n")
audit = {"description": "Post-hoc sensitivity analysis. Common test subset excluding worlds identical to any training world across all conditions/seeds. No retraining or checkpoint selection.", "splits": {}}
for split in ["held_out", "shifted"]:
    example = result["runs"][0]["evaluation"][split]["episodes"]
    excluded = [r["map_seed"] for r in example if key(r["map_seed"], split == "shifted") in all_training]
    scores = {}
    for method in ["untrained", "bc", "ppo", "bc-ppo"]:
        values = []
        for run in result["runs"]:
            if run["method"] == method:
                records = [r for r in run["evaluation"][split]["episodes"] if r["map_seed"] not in excluded]
                values.append(sum(r["outcome"] == "success" for r in records) / len(records))
        scores[method] = {"mean": statistics.mean(values), "std": statistics.stdev(values), "values": values}
    audit["splits"][split] = {"episodes": len(example) - len(excluded), "excluded_seeds": excluded, "summary": scores}
Path("results/disjoint-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
demo_path = Path("public/data/experiment.json")
demo = json.loads(demo_path.read_text())
demo["disjoint_audit"] = audit
demo_path.write_text(json.dumps(demo, separators=(",", ":")) + "\n")
print(json.dumps(audit, indent=2))
