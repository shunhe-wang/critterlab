"""Replay every research episode and verify inference, splits and accounting."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .research import (ROOT, PROTOCOL, Actor, build_manifest, configurations, garden_from, identity,
                       read_result, update_budget)
from .train import metrics
from .world import World, route


def verify_actions(actor, pool, episodes, sampled):
    """Check recorded actions at recorded states, with portable numeric tolerance.

    A near-tie on another CPU must not branch into a different trajectory and
    invalidate its later states. Outcomes are still checked exactly by replay.
    """
    worlds = [World(garden_from(r)) for r in pool]
    uniforms = np.random.default_rng(19123).random((48, len(pool)))
    borderline = 0
    for t in range(48):
        active = [i for i, e in enumerate(episodes) if t < len(e["actions"])]
        if not active:
            break
        with torch.no_grad():
            probabilities = torch.softmax(actor(torch.tensor(np.array([worlds[i].obs() for i in active]))), -1).numpy().astype(np.float64)
        probabilities /= probabilities.sum(-1, keepdims=True)
        actions = np.array([episodes[i]["actions"][t] for i in active])
        rows = np.arange(len(active))
        if sampled:
            cdf = probabilities.cumsum(-1)
            upper = cdf[rows, actions]
            lower = upper - probabilities[rows, actions]
            u = uniforms[t, active]
            assert np.all(u >= lower - 1e-6) and np.all(u <= upper + 1e-6), "Sample incompatible with actor probabilities"
            borderline += int(((u < lower) | (u > upper)).sum())
        else:
            gap = probabilities.max(-1) - probabilities[rows, actions]
            assert np.all(gap <= 1e-6), "Greedy action incompatible with actor probabilities"
            borderline += int((actions != probabilities.argmax(-1)).sum())
        for i, action in zip(active, actions.tolist()):
            worlds[i].step(action)
    return borderline


def audit(result, verify_inference=True):
    def finite(value):
        if isinstance(value, float):
            assert math.isfinite(value), "Nonfinite artifact value"
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
    finite(result)
    assert result.get("complete"), "Incomplete study"
    spec = result["spec"]
    if result["protocol_run"]:
        assert spec == json.loads(PROTOCOL.read_text()), "Specification differs from frozen protocol"
    assert result["manifest"] == build_manifest(spec), "Manifest differs from deterministic frozen construction"
    expected = {(s, f, m, seed) for s, f, m, _ in configurations(spec) for seed in spec["training_seeds"]}
    observed = [(r["setting"], r["family"], r["method"], r["seed"]) for r in result["runs"]]
    assert len(observed) == len(set(observed)) and set(observed) == expected
    world_index = {}
    for pool in result["manifest"].values():
        for record in pool:
            assert record["id"] == identity(garden_from(record))
            assert record["id"] not in world_index, "Duplicate world across frozen groups"
            assert len(route(record["cells"], record["start"], record["goal"])) >= 4
            world_index[record["id"]] = record
    checked, borderline = 0, 0
    for run in result["runs"]:
        before, after = update_budget(run["method"], spec)
        account = run["accounting"]
        assert account["bc_updates"] == before
        assert account["actor_updates"] == before + after
        assert account["critic_updates"] == (after if run["method"] not in ("bc", "dagger") else 0)
        assert account["environment_steps"] == (spec["environment_steps"] if after else 0)
        assert 0 <= account["new_expert_queries"] <= run["demo_transitions"]
        assert account["source_expert_queries"] == run["demo_transitions"]
        assert run["curve"][0]["steps"] == 0
        assert run["curve"][-1]["steps"] == account["environment_steps"]
        assert all(0 <= index < spec["training_worlds_per_family"] for index in account["training_map_indices"])
        actor = Actor.restore(run["actor"])
        for family, modes in run["evaluation"].items():
            pool = result["manifest"][f"test/{family}"]
            for mode, evaluation in modes.items():
                assert [r["world_id"] for r in evaluation["episodes"]] == [r["id"] for r in pool]
                for episode in evaluation["episodes"]:
                    world = World(garden_from(world_index[episode["world_id"]]))
                    for action in episode["actions"]:
                        world.step(action)
                    assert world.done and world.outcome == episode["outcome"]
                    assert world.steps == episode["steps"] == len(episode["actions"])
                    checked += 1
                assert metrics(evaluation["episodes"]) == evaluation["metrics"]
                if verify_inference:
                    borderline += verify_actions(actor, pool, evaluation["episodes"], mode == "sampled")
    for path, digest in result["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    if borderline:
        print(f"Accepted {borderline} numerical boundary choices within 1e-6 probability tolerance; exact recorded outcomes retained.")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=ROOT / "research/results.json.gz")
    parser.add_argument("--replay-only", action="store_true", help="Skip actor inference recomputation")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = read_result(args.path)
    checked = audit(result, not args.replay_only)
    print(f"Audited {len(result['runs'])} trained actors, {checked} episodes, exact splits, budgets and source hashes.")


if __name__ == "__main__":
    main()
