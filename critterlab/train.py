"""Run the predeclared small BC/PPO experiment and export real browser policies.

PPO/GAE follow the published algorithms, with implementation references in
docs/references.md. All code here is authored for this project.
"""
import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .model import Policy
from .world import RNG, World, generate, route

ROOT = Path(__file__).resolve().parents[1]
DEV_SEEDS = list(range(1_000_000, 1_000_064))
TEST_SEEDS = list(range(1_100_000, 1_100_256))
SHIFT_SEEDS = list(range(2_100_000, 2_100_256))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def demonstrations(count=256):
    observations, actions = [], []
    for seed in range(10_000, 10_000 + count):
        garden = generate(seed)
        world = World(garden)
        for action in route(garden.cells, garden.start, garden.goal):
            observations.append(world.obs())
            actions.append(action)
            world.step(action)
    return torch.tensor(np.array(observations)), torch.tensor(actions)


def behavior_clone(policy, obs, actions, epochs=30):
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    updates = 0
    for _ in range(epochs):
        indices = torch.randperm(len(actions))
        for batch in indices.split(256):
            loss = nn.functional.cross_entropy(policy(obs[batch]), actions[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
    return updates


def evaluate(policy, seeds, shifted=False):
    worlds = [World(generate(seed, shifted)) for seed in seeds]
    records = [{"map_seed": seed, "actions": [], "outcome": "running", "steps": 0} for seed in seeds]
    random = RNG(94123)
    for _ in range(48):
        active = [i for i, w in enumerate(worlds) if not w.done]
        if not active:
            break
        if policy is None:
            actions = [random.integer(4) for _ in active]
        else:
            with torch.no_grad():
                obs = torch.tensor(np.array([worlds[i].obs() for i in active]))
                actions = policy(obs).argmax(-1).tolist()
        for index, action in zip(active, actions):
            worlds[index].step(action)
            records[index]["actions"].append(action)
    for record, world in zip(records, worlds):
        record.update(outcome=world.outcome, steps=world.steps)
    return metrics(records), records


def metrics(records):
    n = len(records)
    success = [r["steps"] for r in records if r["outcome"] == "success"]
    return {"episodes": n, "success_rate": len(success) / n,
            "hazard_rate": sum(r["outcome"] == "hazard" for r in records) / n,
            "timeout_rate": sum(r["outcome"] == "timeout" for r in records) / n,
            "mean_success_steps": float(np.mean(success)) if success else None}


def ppo(policy, seed, total_steps, callback):
    torch.manual_seed(seed + 200)
    critic = nn.Sequential(nn.Linear(20, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=0.0003, eps=1e-5)
    rng = RNG(seed + 881)
    map_seeds = []
    def next_world():
        map_seed = 500_000 + rng.integer(500_000)
        map_seeds.append(map_seed)
        return World(generate(map_seed))
    worlds = [next_world() for _ in range(16)]
    updates = 0
    for iteration in range(total_steps // 2048):
        obs_log, action_log, prob_log, value_log, reward_log, done_log = [], [], [], [], [], []
        for _ in range(128):
            obs = torch.tensor(np.array([w.obs() for w in worlds]))
            with torch.no_grad():
                distribution = Categorical(logits=policy(obs))
                actions = distribution.sample()
                values = critic(obs).squeeze(-1)
            rewards, dones = [], []
            for index, action in enumerate(actions.tolist()):
                _, reward, done = worlds[index].step(action)
                rewards.append(reward)
                dones.append(done)
                if done:
                    worlds[index] = next_world()
            obs_log.append(obs)
            action_log.append(actions)
            prob_log.append(distribution.log_prob(actions))
            value_log.append(values)
            reward_log.append(torch.tensor(rewards))
            done_log.append(torch.tensor(dones, dtype=torch.float32))
        observations = torch.stack(obs_log)
        actions = torch.stack(action_log)
        old_log_probs = torch.stack(prob_log)
        values = torch.stack(value_log)
        rewards = torch.stack(reward_log)
        dones = torch.stack(done_log)
        advantages = torch.zeros_like(rewards)
        with torch.no_grad():
            next_value = critic(torch.tensor(np.array([w.obs() for w in worlds]))).squeeze(-1)
            gae = torch.zeros(16)
            for t in reversed(range(128)):
                following = next_value if t == 127 else values[t + 1]
                alive = 1 - dones[t]
                delta = rewards[t] + 0.99 * following * alive - values[t]
                gae = delta + 0.99 * 0.95 * alive * gae
                advantages[t] = gae
        returns = (advantages + values).flatten()
        advantages = advantages.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        observations = observations.reshape(-1, 20)
        actions = actions.flatten()
        old_log_probs = old_log_probs.flatten()
        for _ in range(4):
            for batch in torch.randperm(2048).split(256):
                distribution = Categorical(logits=policy(observations[batch]))
                ratio = (distribution.log_prob(actions[batch]) - old_log_probs[batch]).exp()
                objective = torch.minimum(ratio * advantages[batch], ratio.clamp(0.8, 1.2) * advantages[batch])
                value_loss = ((critic(observations[batch]).squeeze(-1) - returns[batch]) ** 2).mean()
                loss = -objective.mean() + 0.5 * value_loss - 0.01 * distribution.entropy().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite optimization loss")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic.parameters()), 0.5)
                optimizer.step()
                updates += 1
        step = (iteration + 1) * 2048
        if step % 8192 == 0 or step == total_steps:
            callback(step, policy)
    return updates, map_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=65536)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "experiment-v1.json")
    parser.add_argument("--publish", action="store_true", help="Export final seed-0 policies and demo evidence")
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 2048:
        parser.error("steps must be a positive multiple of 2048")
    if args.publish and (args.steps != 65536 or args.seeds != [0, 1, 2]):
        parser.error("publish requires protocol-v1 budget and seeds")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    demo_obs, demo_actions = demonstrations()
    result = {"protocol": "v1", "steps": args.steps, "training_seeds": args.seeds,
              "demonstration_episodes": 256, "demonstration_transitions": len(demo_actions),
              "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
              "device": "cpu", "runs": [], "baselines": {}}
    for split, seeds, shifted in [("held_out", TEST_SEEDS, False), ("shifted", SHIFT_SEEDS, True)]:
        score, records = evaluate(None, seeds, shifted)
        expert = []
        for map_seed in seeds:
            garden = generate(map_seed, shifted)
            actions = route(garden.cells, garden.start, garden.goal)
            expert.append({"map_seed": map_seed, "actions": actions, "outcome": "success", "steps": len(actions)})
        result["baselines"][split] = {"random": {"metrics": score, "episodes": records},
                                        "expert": {"metrics": metrics(expert), "episodes": expert}}
    exports = {}
    parity = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        initial = Policy()
        for name in ["untrained", "bc", "ppo", "bc-ppo"]:
            torch.manual_seed(seed + 10)
            policy = copy.deepcopy(initial)
            run_started = time.perf_counter()
            run = {"seed": seed, "method": name, "curve": [], "bc_updates": 0, "ppo_updates": 0, "training_map_seeds": []}
            if name in ("bc", "bc-ppo"):
                run["bc_updates"] = behavior_clone(policy, demo_obs, demo_actions)
            def progress(step, current):
                score, _ = evaluate(current, DEV_SEEDS)
                run["curve"].append({"steps": step, **score})
                print(f"seed={seed} method={name} steps={step} dev_success={score['success_rate']:.3f}", flush=True)
            progress(0, policy)
            if name in ("ppo", "bc-ppo"):
                run["ppo_updates"], run["training_map_seeds"] = ppo(policy, seed, args.steps, progress)
            run["training_seconds"] = time.perf_counter() - run_started
            run["evaluation"] = {}
            for split, seeds, shifted in [("held_out", TEST_SEEDS, False), ("shifted", SHIFT_SEEDS, True)]:
                score, records = evaluate(policy, seeds, shifted)
                run["evaluation"][split] = {"metrics": score, "episodes": records}
            result["runs"].append(run)
            if args.publish:
                path = ROOT / "checkpoints" / f"{name}-seed-{seed}.pt"
                path.parent.mkdir(exist_ok=True)
                torch.save(policy.state_dict(), path)
                if seed == 0:
                    steps = args.steps if name in ("ppo", "bc-ppo") else 0
                    exports[name] = policy.export(name, seed, steps)
                    for map_seed in [1_100_000, 1_100_009, 2_100_011]:
                        world = World(generate(map_seed, map_seed >= 2_000_000))
                        with torch.no_grad():
                            probabilities = torch.softmax(policy(torch.tensor(world.obs())), -1).tolist()
                        parity.append({"policy": name, "observation": world.obs().tolist(), "probabilities": probabilities})
    result["total_seconds"] = time.perf_counter() - started
    result["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT / "critterlab/world.py", ROOT / "critterlab/model.py",
                                          ROOT / "docs/protocol.md"]}
    result["summary"] = {}
    for name in ["untrained", "bc", "ppo", "bc-ppo"]:
        result["summary"][name] = {}
        for split in ["held_out", "shifted"]:
            values = [r["evaluation"][split]["metrics"]["success_rate"] for r in result["runs"] if r["method"] == name]
            result["summary"][name][split] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                                               "values": values}
    dump(args.output, result)
    if args.publish:
        for name, value in exports.items():
            dump(ROOT / "public/models" / f"{name}.json", value)
        demo = {k: result[k] for k in ["protocol", "steps", "training_seeds", "demonstration_transitions", "summary", "total_seconds"]}
        demo["curves"] = [{"seed": r["seed"], "method": r["method"], "points": r["curve"]} for r in result["runs"]]
        dump(ROOT / "public/data/experiment.json", demo)
        dump(ROOT / "tests/fixtures/policy-parity.json", parity)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
