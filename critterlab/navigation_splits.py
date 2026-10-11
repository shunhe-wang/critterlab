"""Freeze r1 worlds without observing any model's scores."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from .world import generate

ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=None)
def key(seed, shifted=False):
    garden = generate(seed, shifted)
    return tuple(garden.cells) + (garden.start, garden.goal)

def world_hash(seed, shifted=False):
    return hashlib.sha256(bytes(key(seed, shifted))).hexdigest()

def build():
    previous_path = ROOT / "results/experiment-v1.json"
    previous = json.loads(previous_path.read_text())
    demos = list(range(10000, 10256))
    demonstrations = {key(s) for s in demos}
    old = set(demonstrations)
    for run in previous["runs"]:
        old.update(key(s) for s in run["training_map_seeds"])
    diagnostic = {key(67), key(67, True), key(1100000), key(1100001)}
    assert not demonstrations & {key(67), key(67, True)}, "Revise protocol: seed 67 appears in BC"
    def choose(start, count, excluded, shifted=False):
        selected, seen, seed = [], set(), start
        while len(selected) < count:
            value = key(seed, shifted)
            if value not in excluded and value not in seen:
                selected.append(seed); seen.add(value)
            seed += 1
        return selected, seen
    pool, new_worlds = choose(30000, 1024, diagnostic)
    forbidden = old | new_worlds | diagnostic
    dev, dev_worlds = choose(50000, 64, forbidden)
    standard, test_worlds = choose(3100000, 256, forbidden | dev_worlds)
    wilder, _ = choose(4100000, 256, forbidden | dev_worlds | test_worlds, True)
    return {"protocol":"r1", "legacy_result_sha256":hashlib.sha256(previous_path.read_bytes()).hexdigest(),
            "legacy_unique_training_worlds":len(old), "demonstration_seeds":demos, "pool":pool,
            "development":dev, "standard":standard, "wilder":wilder,
            "world_sha256":{name:[world_hash(s, name == "wilder") for s in seeds]
                            for name,seeds in [("pool",pool),("development",dev),("standard",standard),("wilder",wilder)]}}

def main():
    result = build()
    path = ROOT / "results/navigation-r1-splits.json"
    path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print({name:len(result[name]) for name in ["pool","development","standard","wilder"]})
    print(f"Excluded {result['legacy_unique_training_worlds']} unique legacy training worlds")

if __name__ == "__main__":
    main()
