"""Refresh cross-language reference vectors; behavioral literals live in tests."""
import json
from pathlib import Path
from critterlab.world import World, generate

vectors = []
for seed, shifted in [(0, False), (42, False), (1_100_009, False), (2_100_011, True)]:
    garden = generate(seed, shifted)
    world = World(garden)
    vector = {"seed": seed, "shifted": shifted,
              "garden": {"cells": garden.cells, "start": garden.start, "goal": garden.goal, "seed": seed},
              "observation": world.obs().tolist(), "transitions": []}
    for action in [0, 1, 2, 3, 1, 1, 0, 0, 2, 2]:
        if world.done:
            break
        _, reward, _ = world.step(action)
        vector["transitions"].append({"action": action, "pos": world.pos, "reward": reward, "outcome": world.outcome})
    vectors.append(vector)
path = Path("tests/fixtures/world-parity.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(vectors, separators=(",", ":")) + "\n")
