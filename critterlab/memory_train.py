"""Execute the frozen m1 protocol. No test-driven tuning or checkpoint selection."""
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .memory_world import TRAIN, DEV, TEST, LONG, MemoryWorld, Task
from .memory_model import Reactive, Recurrent
from .model import Policy

ROOT = Path(__file__).resolve().parents[1]
EPOCHS = 200


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def demonstrations():
    episodes = []
    for task in TRAIN:
        world = MemoryWorld(task)
        observations, actions, weights = [], [], []
        while not world.done:
            observations.append(world.obs().tolist())
            actions.append(world.expert())
            weights.append(4 if world.steps >= task.delay and world.pos == world.junction else 1)
            world.step(actions[-1])
        episodes.append({"task": asdict(task), "observations": observations, "actions": actions, "weights": weights})
    length = max(len(e["actions"]) for e in episodes)
    obs = torch.zeros(len(episodes), length, 22)
    labels = torch.zeros(len(episodes), length, dtype=torch.long)
    weights = torch.zeros(len(episodes), length)
    for i, episode in enumerate(episodes):
        n = len(episode["actions"])
        obs[i, :n] = torch.tensor(episode["observations"])
        labels[i, :n] = torch.tensor(episode["actions"])
        weights[i, :n] = torch.tensor(episode["weights"])
    return obs, labels, weights, episodes


def metrics(records):
    return {"episodes": len(records), **{f"{name.replace('-', '_')}_rate": sum(r["outcome"] == name for r in records) / len(records)
                                     for name in ("success", "wrong-berry", "timeout")}}


def neural_step(model, world, hidden, condition="normal", original=False):
    observation = torch.tensor(world.obs())[None, :]
    if original:
        return model(observation[:, :20]), None
    if condition == "no-cue":
        observation[:, -2:] = 0
    elif condition == "flip-cue":
        observation[:, -2:] = observation[:, -2:].flip(-1)
    if condition == "reset":
        hidden = None
    return model.step(observation, hidden)


def evaluate(model, cases, condition="normal", original=False):
    records = []
    with torch.no_grad():
        for task in cases:
            world, hidden, actions = MemoryWorld(task), None, []
            while not world.done:
                logits, hidden = neural_step(model, world, hidden, condition, original)
                action = int(logits.argmax(-1).item())
                world.step(action); actions.append(action)
            records.append({"task": asdict(task), "actions": actions, "outcome": world.outcome, "steps": world.steps})
    return {"metrics": metrics(records), "episodes": records}


def train(model, obs, labels, weights, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    curve = [{"epoch": 0, "success_rate": evaluate(model, DEV)["metrics"]["success_rate"]}]
    losses = []
    for epoch in range(1, epochs + 1):
        logits = model(obs)
        errors = nn.functional.cross_entropy(logits.reshape(-1, 4), labels.flatten(), reduction="none").reshape_as(weights)
        loss = (errors * weights).sum() / weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        losses.append(float(loss.detach()))
        if epoch % 20 == 0:
            score = evaluate(model, DEV)["metrics"]["success_rate"]
            curve.append({"epoch": epoch, "success_rate": score})
            print(f"{type(model).__name__} epoch={epoch} loss={losses[-1]:.4f} dev={score:.3f}", flush=True)
    return losses, curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=ROOT / "results/memory-m1.json")
    args = parser.parse_args()
    if args.epochs < 1 or args.publish and (args.epochs != EPOCHS or args.seeds != [0, 1, 2]):
        parser.error("Publishing requires m1's 200 epochs and seeds 0, 1, 2")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    obs, labels, weights, episodes = demonstrations()
    result = {"protocol": "m1", "protocol_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "epochs": args.epochs, "training_seeds": args.seeds, "device": "cpu", "torch": torch.__version__,
              "numpy": np.__version__, "platform": platform.platform(), "runs": [],
              "training_episodes": episodes, "training_transitions": sum(len(e["actions"]) for e in episodes),
              "splits": {name: [asdict(t) for t in cases] for name, cases in [("development", DEV), ("standard", TEST), ("long", LONG)]}}
    export_hashes = {}
    for seed in args.seeds:
        for method, constructor in [("reactive", Reactive), ("gru", Recurrent), ("current", Policy)]:
            torch.manual_seed(seed); model = constructor()
            run = {"seed": seed, "method": method, "parameters": sum(p.numel() for p in model.parameters()), "losses": [], "curve": []}
            before = time.perf_counter()
            if method == "current":
                path = ROOT / f"checkpoints/bc-ppo-seed-{seed}.pt"
                model.load_state_dict(torch.load(path, weights_only=True))
            else:
                run["losses"], run["curve"] = train(model, obs, labels, weights, args.epochs)
                path = ROOT / f"checkpoints/m1-{method}-seed-{seed}.pt"
                if args.publish:
                    torch.save(model.state_dict(), path)
            run["training_seconds"] = time.perf_counter() - before
            if args.publish or method == "current":
                run["checkpoint"] = str(path.relative_to(ROOT))
                run["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            run["evaluation"] = {name: evaluate(model, cases, original=method == "current") for name, cases in [("standard", TEST), ("long", LONG)]}
            if method == "gru":
                run["controls"] = {condition: evaluate(model, TEST, condition) for condition in ("no-cue", "flip-cue", "reset")}
            result["runs"].append(run)
            if args.publish and seed == 0 and method != "current":
                exported = ROOT / f"public/models/m1-{method}.json"
                dump(exported, model.export(seed))
                export_hashes[str(exported.relative_to(ROOT))] = hashlib.sha256(exported.read_bytes()).hexdigest()
                if method == "gru":
                    vectors = []
                    for task in [Task(3, 1, 8, 0), Task(4, 1, 32, 1), Task(2, 2, 2, 1)]:
                        world, hidden, frames = MemoryWorld(task), None, []
                        with torch.no_grad():
                            while not world.done:
                                logits, hidden = neural_step(model, world, hidden)
                                action = world.expert()
                                frames.append({"observation": world.obs().tolist(), "hidden": hidden[0].tolist(),
                                               "probabilities": logits.softmax(-1)[0].tolist(), "action": action, "pos": world.pos})
                                world.step(action)
                        vectors.append({"task": asdict(task), "cells": world.cells, "frames": frames, "outcome": world.outcome})
                    dump(ROOT / "tests/fixtures/memory-parity.json", vectors)
    result["summary"] = {}
    for method in ("reactive", "gru", "current"):
        result["summary"][method] = {}
        for split in ("standard", "long"):
            values = [r["evaluation"][split]["metrics"]["success_rate"] for r in result["runs"] if r["method"] == method]
            result["summary"][method][split] = {"values": values, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.}
    result["controls"] = {}
    for condition in ("no-cue", "flip-cue", "reset"):
        values = [r["controls"][condition]["metrics"]["success_rate"] for r in result["runs"] if r["method"] == "gru"]
        result["controls"][condition] = {"values": values, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.}
    sources = ["docs/memory-protocol.md", "critterlab/memory_world.py", "critterlab/memory_model.py", "critterlab/memory_train.py"]
    result["source_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}
    result["export_sha256"] = export_hashes
    result["total_seconds"] = time.perf_counter() - started
    dump(args.output, result)
    if args.publish:
        dump(ROOT / "public/data/memory.json", {key: result[key] for key in ["protocol", "epochs", "summary", "controls", "total_seconds", "training_transitions"]})
    print(json.dumps({"summary": result["summary"], "controls": result["controls"], "seconds": result["total_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
