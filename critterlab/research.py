"""CPU research runner; deliberately independent of immutable v1 artifacts.

Run: uv run python -m critterlab.research --output research/results.json.gz
The JSON protocol is the specification. Smoke runs are marked non-protocol.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .train import metrics
from .world import Garden, RNG, World, generate, route

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/protocol.json"
FAMILIES = ("standard", "dense", "barrier")


class Actor(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(nn.Linear(20, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 4))

    def forward(self, observation):
        return self.net(observation)

    def export(self):
        return {"width": self.width,
                "state": {k: v.detach().tolist() for k, v in self.state_dict().items()}}

    @classmethod
    def restore(cls, record):
        actor = cls(record["width"])
        actor.load_state_dict({k: torch.tensor(v) for k, v in record["state"].items()})
        return actor


class ResearchWorld(World):
    def __init__(self, garden, shaping=1.0):
        super().__init__(garden)
        self.shaping = shaping

    def step(self, action):
        old = self.pos
        obs, reward, done = super().step(action)
        # v1 only shapes ordinary safe moves. Terminal rewards take precedence.
        if self.pos != old and self.outcome in ("running", "timeout"):
            gx, gy = self.garden.goal % 7, self.garden.goal // 7
            before = abs(gx - old % 7) + abs(gy - old // 7)
            after = abs(gx - self.pos % 7) + abs(gy - self.pos // 7)
            reward -= (1 - self.shaping) * .1 * (before - after)
        return obs, reward, done


def identity(garden):
    return hashlib.sha256(bytes(garden.cells + [garden.start, garden.goal])).hexdigest()


def make_garden(seed, family):
    if family != "barrier":
        return generate(seed, family == "dense")
    rng = RNG(seed)
    for _ in range(1000):
        cells = []
        for y in range(7):
            for x in range(7):
                r = rng.next()
                cells.append(1 if x in (0, 6) or y in (0, 6) else
                             1 if r < .08 else 2 if r < .14 else 0)
        gap = 1 + rng.integer(5)
        for y in range(1, 6):
            cells[y * 7 + 3] = 0 if y == gap else 1
        start = 1 + rng.integer(2) + 7 * (1 + rng.integer(5))
        goal = 4 + rng.integer(2) + 7 * (1 + rng.integer(5))
        # Randomize which side contains the start without changing wall structure.
        if rng.integer(2):
            start, goal = goal, start
        cells[start] = cells[goal] = 0
        path = route(cells, start, goal)
        if path is not None and len(path) >= 4:
            return Garden(cells, start, goal, seed)
    raise ValueError(f"No barrier garden for {seed}")


def garden_from(record):
    return Garden(record["cells"], record["start"], record["goal"], record["seed"])


def build_manifest(spec):
    used = set()
    groups = {}
    plans = [("demos", "standard", spec["demonstration_worlds"], 3_000_000)]
    plans += [(f"train/{f}", f, spec["training_worlds_per_family"], 4_000_000 + i * 100_000)
              for i, f in enumerate(FAMILIES)]
    plans += [(f"test/{f}", f, spec["evaluation_worlds_per_family"], 5_000_000 + i * 100_000)
              for i, f in enumerate(FAMILIES)]
    for group, family, count, seed in plans:
        records = []
        while len(records) < count:
            garden = make_garden(seed, family)
            seed += 1
            key = identity(garden)
            if key in used:
                continue
            used.add(key)
            records.append({"id": key, "seed": garden.seed, "family": family,
                            "cells": garden.cells, "start": garden.start, "goal": garden.goal})
        groups[group] = records
    return groups


def demo_data(manifest, count, noise):
    observations, labels = [], []
    for record in manifest["demos"][:count]:
        garden = garden_from(record)
        world = World(garden)
        for action in route(garden.cells, garden.start, garden.goal):
            observations.append(world.obs())
            labels.append(action)
            world.step(action)
    labels = np.array(labels)
    random = np.random.default_rng(61011)
    corrupted = random.random(len(labels)) < noise
    labels[corrupted] = (labels[corrupted] + random.integers(1, 4, size=corrupted.sum())) % 4
    return torch.tensor(np.array(observations)), torch.tensor(labels), int(corrupted.sum())


def bc_updates(actor, observations, labels, updates):
    optimizer = torch.optim.Adam(actor.parameters(), lr=.001)
    for _ in range(updates):
        batch = torch.randint(len(labels), (256,))
        loss = nn.functional.cross_entropy(actor(observations[batch]), labels[batch])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite BC loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def support_patterns(observations):
    return {bytes(row[2:].to(torch.uint8).tolist()) for row in observations}


def imitation_weight(method, support, entropy):
    if method == "fixed-imitation":
        return .1
    if method == "adaptive-imitation":
        return .1 * support * min(1., max(0., entropy / math.log(4)))
    return 0.


def update_budget(method, spec):
    if method in ("ppo", "ppo-entropy"):
        return 0, spec["actor_update_budget"]
    if method == "bc":
        return spec["actor_update_budget"], 0
    if method == "ppo-short":
        return 0, spec["actor_update_budget"] - spec["pretraining_updates"]
    return spec["pretraining_updates"], spec["actor_update_budget"] - spec["pretraining_updates"]


def probe(actor, observations):
    with torch.no_grad():
        dist = Categorical(logits=actor(observations))
        return {"entropy": float(dist.entropy().mean()),
                "max_probability": float(dist.probs.max(-1).values.mean())}


def generalized_advantages(rewards, dones, values, next_value):
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1])
    for t in reversed(range(len(rewards))):
        following = next_value if t == len(rewards) - 1 else values[t + 1]
        alive = 1 - dones[t]
        delta = rewards[t] + .99 * following * alive - values[t]
        gae = delta + .99 * .95 * alive * gae
        advantages[t] = gae
    return advantages


def clipped_objective(ratio, advantages):
    return torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages)


def learn(actor, method, seed, spec, pool, demo_obs, demo_labels, shaping, curve):
    before_updates, after_updates = update_budget(method, spec)
    torch.manual_seed(seed + 10)
    bc_updates(actor, demo_obs, demo_labels, before_updates)
    probe_obs = torch.tensor(np.array([World(garden_from(r)).obs() for r in pool]))
    patterns = support_patterns(demo_obs)
    def point(steps):
        return {"steps": steps, "source_probe": probe(actor, demo_obs),
                "target_probe": probe(actor, probe_obs)}
    curve.append(point(0))
    account = {"source_label_presentations": before_updates * 256,
               "source_expert_queries": len(demo_labels), "new_expert_queries": 0,
               "unreachable_query_states": 0, "bc_updates": before_updates,
               "actor_updates": before_updates, "critic_updates": 0,
               "environment_steps": 0, "collisions": 0, "training_map_indices": [],
               "unique_world_positions": 0}
    if not after_updates:
        return account
    torch.manual_seed(seed + 200)
    width = actor.width
    critic = nn.Sequential(nn.Linear(20, width), nn.Tanh(), nn.Linear(width, width),
                           nn.Tanh(), nn.Linear(width, 1))
    dagger = method == "dagger"
    optimizer = torch.optim.Adam(actor.parameters() if dagger else
                                 list(actor.parameters()) + list(critic.parameters()),
                                 lr=.001 if dagger else .0003, eps=1e-5)
    reset_rng = np.random.default_rng(seed + 881)
    query_rng = np.random.default_rng(seed + 918)
    visited = set()
    def next_world():
        index = int(reset_rng.integers(len(pool)))
        account["training_map_indices"].append(index)
        return ResearchWorld(garden_from(pool[index]), shaping), index
    pairs = [next_world() for _ in range(16)]
    worlds, indices = [p[0] for p in pairs], [p[1] for p in pairs]
    iterations = spec["environment_steps"] // 2048
    per_rollout = after_updates // iterations
    if after_updates % iterations:
        raise ValueError("Update budget must divide the number of rollouts")
    aggregated_obs, aggregated_labels = demo_obs.clone(), demo_labels.clone()
    entropy_coefficient = .03 if method.endswith("entropy") else .01
    for iteration in range(iterations):
        obs_log, action_log, log_prob_log, value_log, reward_log, done_log = [], [], [], [], [], []
        candidates = []
        entropies = []
        for _ in range(128):
            obs = torch.tensor(np.array([w.obs() for w in worlds]))
            with torch.no_grad():
                distribution = Categorical(logits=actor(obs))
                actions = distribution.sample()
                values = critic(obs).squeeze(-1)
            entropies.append(float(distribution.entropy().mean()))
            rewards, dones = [], []
            for i, action in enumerate(actions.tolist()):
                visited.add((indices[i], worlds[i].pos))
                if dagger:
                    candidates.append((obs[i], worlds[i].garden, worlds[i].pos))
                old_pos = worlds[i].pos
                _, reward, done = worlds[i].step(action)
                account["collisions"] += worlds[i].pos == old_pos
                rewards.append(reward)
                dones.append(done)
                if done:
                    worlds[i], indices[i] = next_world()
            obs_log.append(obs)
            action_log.append(actions)
            log_prob_log.append(distribution.log_prob(actions))
            value_log.append(values)
            reward_log.append(torch.tensor(rewards))
            done_log.append(torch.tensor(dones, dtype=torch.float32))
        observations = torch.stack(obs_log)
        actions = torch.stack(action_log)
        old_log_probs = torch.stack(log_prob_log)
        values = torch.stack(value_log)
        rewards = torch.stack(reward_log)
        dones = torch.stack(done_log)
        with torch.no_grad():
            next_value = critic(torch.tensor(np.array([w.obs() for w in worlds]))).squeeze(-1)
            advantages = generalized_advantages(rewards, dones, values, next_value)
        returns = (advantages + values).flatten()
        advantages = advantages.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        observations = observations.reshape(-1, 20)
        actions = actions.flatten()
        old_log_probs = old_log_probs.flatten()
        support = float(np.mean([bytes(row[2:].to(torch.uint8).tolist()) in patterns
                                 for row in observations]))
        rollout_entropy = float(np.mean(entropies))
        weight = imitation_weight(method, support, rollout_entropy)
        if dagger:
            allowance = min(math.ceil(len(demo_labels) / iterations),
                            len(demo_labels) - account["new_expert_queries"])
            extra_obs, extra_labels = [], []
            for candidate in query_rng.permutation(len(candidates)):
                if len(extra_labels) >= allowance:
                    break
                observation, garden, pos = candidates[candidate]
                path = route(garden.cells, pos, garden.goal)
                if not path:
                    account["unreachable_query_states"] += 1
                    continue
                extra_obs.append(observation)
                extra_labels.append(path[0])
            if extra_labels:
                aggregated_obs = torch.cat((aggregated_obs, torch.stack(extra_obs)))
                aggregated_labels = torch.cat((aggregated_labels, torch.tensor(extra_labels)))
                account["new_expert_queries"] += len(extra_labels)
        batches = []
        while len(batches) < per_rollout:
            batches.extend(torch.randperm(2048).split(256))
        kl_values, clip_fractions, value_errors, gradient_norms = [], [], [], []
        for batch in batches[:per_rollout]:
            if dagger:
                examples = torch.randint(len(aggregated_labels), (256,))
                loss = nn.functional.cross_entropy(actor(aggregated_obs[examples]), aggregated_labels[examples])
                # Count actual source examples drawn from the aggregate.
                account["source_label_presentations"] += int((examples < len(demo_labels)).sum())
            else:
                dist = Categorical(logits=actor(observations[batch]))
                ratio = (dist.log_prob(actions[batch]) - old_log_probs[batch]).exp()
                objective = clipped_objective(ratio, advantages[batch])
                value_loss = ((critic(observations[batch]).squeeze(-1) - returns[batch]) ** 2).mean()
                with torch.no_grad():
                    kl_values.append(float(((ratio - 1) - ratio.log()).mean()))
                    clip_fractions.append(float(((ratio - 1).abs() > .2).float().mean()))
                    value_errors.append(float(value_loss))
                loss = -objective.mean() + .5 * value_loss - entropy_coefficient * dist.entropy().mean()
                if weight:
                    examples = torch.randint(len(demo_labels), (256,))
                    loss += weight * nn.functional.cross_entropy(actor(demo_obs[examples]), demo_labels[examples])
                    account["source_label_presentations"] += 256
                account["critic_updates"] += 1
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite research loss")
            optimizer.zero_grad()
            loss.backward()
            gradient_norms.append(float(nn.utils.clip_grad_norm_(
                list(actor.parameters()) + ([] if dagger else list(critic.parameters())), .5)))
            optimizer.step()
            account["actor_updates"] += 1
        steps = (iteration + 1) * 2048
        curve.append({**point(steps), "rollout_entropy": rollout_entropy,
                      "source_support": support, "imitation_weight": weight,
                      "unique_world_positions": len(visited), "collisions": account["collisions"],
                      "minibatch_approx_kl": float(np.mean(kl_values)) if kl_values else None,
                      "minibatch_clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else None,
                      "value_mse": float(np.mean(value_errors)) if value_errors else None,
                      "unclipped_gradient_norm": float(np.mean(gradient_norms))})
    account["environment_steps"] = iterations * 2048
    account["unique_world_positions"] = len(visited)
    return account


def evaluate(actor, pool, sampled=False):
    worlds = [World(garden_from(r)) for r in pool]
    records = [{"world_id": r["id"], "actions": []} for r in pool]
    uniforms = np.random.default_rng(19123).random((48, len(pool)))
    for t in range(48):
        active = [i for i, w in enumerate(worlds) if not w.done]
        if not active:
            break
        with torch.no_grad():
            logits = actor(torch.tensor(np.array([worlds[i].obs() for i in active])))
            if sampled:
                probabilities = torch.softmax(logits, -1).numpy().astype(np.float64)
                probabilities /= probabilities.sum(-1, keepdims=True)
                actions = (uniforms[t, active, None] > np.cumsum(probabilities, -1)).sum(-1).tolist()
            else:
                actions = logits.argmax(-1).tolist()
        for i, action in zip(active, actions):
            worlds[i].step(action)
            records[i]["actions"].append(action)
    for record, world in zip(records, worlds):
        record.update(outcome=world.outcome, steps=world.steps)
    return {"metrics": metrics(records), "episodes": records}


def write_result(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix == ".gz":
        with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
            zipped.write(json.dumps(value, separators=(",", ":")).encode())
    else:
        temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    temporary.replace(path)


def read_result(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as file:
            return json.load(file)
    return json.loads(path.read_text())


def source_digests():
    paths = ["critterlab/research.py", "critterlab/world.py", "critterlab/train.py",
             "research/protocol.json", "docs/research-protocol.md"]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def configurations(spec):
    settings = [("main", f, m, {}) for f in spec["families"] for m in spec["methods"]]
    for ablation in spec["ablations"]:
        settings += [(ablation["name"], spec["ablation_family"], m, ablation)
                     for m in spec["ablation_methods"]]
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research/results.json.gz")
    parser.add_argument("--smoke", action="store_true", help="Tiny non-protocol harness check")
    args = parser.parse_args()
    spec = json.loads(PROTOCOL.read_text())
    if args.smoke:
        spec.update(training_seeds=[0], environment_steps=2048, actor_update_budget=32,
                    pretraining_updates=4, training_worlds_per_family=8,
                    evaluation_worlds_per_family=8, demonstration_worlds=8, ablations=[])
        if args.output == ROOT / "research/results.json.gz":
            parser.error("Smoke requires a separate --output")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    result = {"protocol": spec["name"], "protocol_run": not args.smoke, "spec": spec,
              "manifest": build_manifest(spec), "source_sha256": source_digests(),
              "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
              "device": "cpu", "runs": []}
    settings = configurations(spec)
    total = len(settings) * len(spec["training_seeds"])
    for setting, family, method, override in settings:
        demo_count = min(override.get("demo_worlds", spec["demonstration_worlds"]), spec["demonstration_worlds"])
        demo_obs, demo_labels, corrupted = demo_data(result["manifest"], demo_count, override.get("label_noise", 0.))
        for seed in spec["training_seeds"]:
            run_started = time.perf_counter()
            torch.manual_seed(seed)
            actor = Actor(override.get("width", 64))
            run = {"setting": setting, "family": family, "method": method, "seed": seed,
                   "demo_worlds": demo_count, "demo_transitions": len(demo_labels),
                   "corrupted_labels": corrupted, "shaping": override.get("shaping", 1.), "curve": []}
            run["accounting"] = learn(actor, method, seed, spec, result["manifest"][f"train/{family}"],
                                      demo_obs, demo_labels, run["shaping"], run["curve"])
            run["training_seconds"] = time.perf_counter() - run_started
            run["evaluation"] = {f: {mode: evaluate(actor, result["manifest"][f"test/{f}"], mode == "sampled")
                                     for mode in ("greedy", "sampled")} for f in FAMILIES}
            run["actor"] = actor.export()
            result["runs"].append(run)
            result["total_seconds"] = time.perf_counter() - started
            print(f"{len(result['runs'])}/{total} {setting}/{family}/{method} seed={seed} "
                  f"greedy={run['evaluation'][family]['greedy']['metrics']['success_rate']:.3f}", flush=True)
        # Save complete paired-seed blocks. Partial results are explicitly detectable.
        write_result(args.output, result)
    result["complete"] = True
    result["total_seconds"] = time.perf_counter() - started
    write_result(args.output, result)
    print(f"Saved {total} runs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
