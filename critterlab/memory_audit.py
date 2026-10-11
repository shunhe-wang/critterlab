"""Recompute every m1 neural action, environment outcome, control and summary."""
from dataclasses import asdict
import hashlib
import json
import statistics
from pathlib import Path
import torch
from .memory_world import TRAIN, DEV, TEST, LONG, Task, MemoryWorld
from .memory_model import Reactive, Recurrent
from .model import Policy

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    result = json.loads((ROOT / "results/memory-m1.json").read_text())
    assert result["protocol"] == "m1" and result["epochs"] == 200 and result["training_seeds"] == [0, 1, 2]
    assert len(result["runs"]) == 9
    assert {(r["seed"], r["method"]) for r in result["runs"]} == {(s, m) for s in (0, 1, 2) for m in ("reactive", "gru", "current")}
    public = json.loads((ROOT / "public/data/memory.json").read_text())
    assert public == {key: result[key] for key in ["protocol", "epochs", "summary", "controls", "total_seconds", "training_transitions"]}
    for key in ("source_sha256", "export_sha256"):
        for path, digest in result[key].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    geometries = lambda cases: {tuple(MemoryWorld(t).cells) for t in cases}
    assert not geometries(TRAIN + DEV) & geometries(TEST + LONG)
    assert len(result["training_episodes"]) == len(TRAIN) == 36
    for record, task in zip(result["training_episodes"], TRAIN):
        assert record["task"] == asdict(task)
        world = MemoryWorld(task)
        for obs, action, weight in zip(record["observations"], record["actions"], record["weights"], strict=True):
            assert obs == world.obs().tolist() and action == world.expert()
            assert weight == (4 if world.pos == world.junction and world.steps >= task.delay else 1)
            world.step(action)
        assert world.outcome == "success"
    assert result["training_transitions"] == sum(len(e["actions"]) for e in result["training_episodes"])
    checked = 0
    for run in result["runs"]:
        constructor = {"reactive": Reactive, "gru": Recurrent, "current": Policy}[run["method"]]
        model = constructor()
        path = ROOT / run["checkpoint"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == run["checkpoint_sha256"]
        model.load_state_dict(torch.load(path, weights_only=True))
        groups = [(name, evaluation, "normal", cases) for name, evaluation, cases in
                  [("standard", run["evaluation"]["standard"], TEST), ("long", run["evaluation"]["long"], LONG)]]
        groups += [(name, evaluation, name, TEST) for name, evaluation in run.get("controls", {}).items()]
        with torch.no_grad():
            for name, evaluation, control, cases in groups:
                assert [r["task"] for r in evaluation["episodes"]] == [asdict(t) for t in cases]
                for record in evaluation["episodes"]:
                    world, hidden = MemoryWorld(Task(**record["task"])), None
                    for action in record["actions"]:
                        obs = torch.tensor(world.obs())[None, :]
                        if run["method"] == "current":
                            logits = model(obs[:, :20])
                        else:
                            if control == "no-cue": obs[:, -2:] = 0
                            if control == "flip-cue": obs[:, -2:] = obs[:, -2:].flip(-1)
                            if control == "reset": hidden = None
                            logits, hidden = model.step(obs, hidden)
                        assert int(logits.argmax(-1)) == action, (run["seed"], run["method"], name, record["task"])
                        world.step(action)
                    assert world.done and world.outcome == record["outcome"] and world.steps == record["steps"]
                    checked += 1
                assert evaluation["metrics"]["episodes"] == len(cases)
                for outcome in ("success", "wrong-berry", "timeout"):
                    value = sum(r["outcome"] == outcome for r in evaluation["episodes"]) / len(cases)
                    assert evaluation["metrics"][outcome.replace('-', '_') + "_rate"] == value
    def summary(values, stored):
        assert values == stored["values"]
        assert abs(statistics.mean(values) - stored["mean"]) < 1e-12
        assert abs(statistics.stdev(values) - stored["std"]) < 1e-12
    for method, splits in result["summary"].items():
        for split, stored in splits.items():
            summary([r["evaluation"][split]["metrics"]["success_rate"] for r in result["runs"] if r["method"] == method], stored)
    for condition, stored in result["controls"].items():
        summary([r["controls"][condition]["metrics"]["success_rate"] for r in result["runs"] if r["method"] == "gru"], stored)
    for method in ("reactive", "gru"):
        model = {"reactive": Reactive, "gru": Recurrent}[method]()
        model.load_state_dict(torch.load(ROOT / f"checkpoints/m1-{method}-seed-0.pt", weights_only=True))
        assert model.export(0) == json.loads((ROOT / f"public/models/m1-{method}.json").read_text())
    print(f"Audited {checked} memory episodes: neural actions, outcomes, controls, summaries, geometry separation and hashes.")


if __name__ == "__main__":
    main()
