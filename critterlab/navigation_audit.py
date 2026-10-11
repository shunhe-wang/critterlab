"""Independent full-prefix neural replay and exact-world audit for r1."""
import hashlib
import json
import statistics
from pathlib import Path
import numpy as np
import torch
from .world import World, generate
from .model import Policy
from .navigation_model import GardenGRU
from .navigation_splits import build, key

ROOT = Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(1)
    result = json.loads((ROOT/"results/navigation-r1.json").read_text())
    splits = json.loads((ROOT/"results/navigation-r1-splits.json").read_text())
    assert splits == build(), "Frozen split differs from deterministic pretraining selection"
    assert result["protocol"] == "r1" and result["ppo_steps"] == 65536 and result["bc_epochs"] == 30
    assert result["training_seeds"] == [0,1,2] and len(result["runs"]) == 9
    assert {(r["seed"],r["method"]) for r in result["runs"]} == {(s,m) for s in [0,1,2] for m in ["gru","reactive","current"]}
    for group in ["source_sha256","export_sha256"]:
        for path,digest in result[group].items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest,path
    public = json.loads((ROOT/"public/data/navigation.json").read_text())
    assert public == {k:result[k] for k in ["protocol","summary","paired","ppo_steps","bc_transitions","total_seconds"]}
    checked = 0
    def verify_episode(model,record,reset=False):
        nonlocal checked
        world = World(generate(record["map_seed"],record["shifted"])); prefix,positions = [],[world.pos]
        cycles = []
        for action in record["actions"]:
            observation = world.obs()
            if isinstance(model,GardenGRU):
                prefix.append(observation)
                sequence = torch.tensor(np.array([observation] if reset else prefix))[None]
                features,_ = model.gru(sequence)
                logits = model.head(features[:,-1])
            else: logits = model(torch.tensor(observation)[None])
            assert int(logits.argmax(-1).item()) == action,(record["map_seed"],reset,world.steps)
            world.step(action); positions.append(world.pos)
            tail = positions[-6:]; cycles.append(len(tail)==6 and len(set(tail[::2]))==len(set(tail[1::2]))==1)
        assert world.done and world.outcome == record["outcome"] and world.steps == record["steps"]
        assert record["any_cycle"] == any(cycles)
        assert record["loop_timeout"] == (world.outcome=="timeout" and cycles[-1])
        checked += 1
    def verify_metrics(evaluation):
        episodes = evaluation["episodes"]; score = evaluation["metrics"]; n = len(episodes)
        assert score["episodes"] == n
        for outcome in ["success","hazard","timeout"]:
            assert score[outcome+"_rate"] == sum(r["outcome"]==outcome for r in episodes)/n
        for flag in ["any_cycle","loop_timeout"]: assert score[flag+"_rate"] == sum(r[flag] for r in episodes)/n
        lengths = [r["steps"] for r in episodes if r["outcome"]=="success"]
        assert score["mean_success_steps"] == (statistics.mean(lengths) if lengths else None)
    with torch.no_grad():
        for run in result["runs"]:
            model = GardenGRU() if run["method"]=="gru" else Policy()
            checkpoint = ROOT/run["checkpoint"]
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == run["checkpoint_sha256"]
            model.load_state_dict(torch.load(checkpoint,weights_only=True))
            assert sum(p.numel() for p in model.parameters()) == run["parameters"]
            if run["method"]!="current":
                assert len(run["bc_losses"])==180 and run["ppo_updates"]==1024
                assert set(run["training_map_seeds"]) <= set(splits["pool"])
                assert [p["steps"] for p in run["curve"]] == list(range(0,65537,8192))
            for group in ["evaluation","controls"]:
                for split,evaluation in run.get(group,{}).items():
                    assert [r["map_seed"] for r in evaluation["episodes"]] == splits[split]
                    assert all(r["shifted"] == (split=="wilder") for r in evaluation["episodes"])
                    for record in evaluation["episodes"]: verify_episode(model,record,group=="controls")
                    verify_metrics(evaluation)
            for group in ["diagnostics","reset_diagnostics"]:
                if group not in run: continue
                assert [(r["map_seed"],r["shifted"]) for r in run[group]] == [(67,False),(67,True),(1100000,False),(1100001,False)]
                for record in run[group]: verify_episode(model,record,group=="reset_diagnostics")
            if run["seed"]==0 and run["method"]!="current":
                expected = model.export(0) if run["method"]=="gru" else model.export("Random-garden reactive · r1",0,65536)
                assert expected == json.loads((ROOT/f"public/models/r1-{run['method']}.json").read_text())
    for method,ss in result["summary"].items():
        for split,scores in ss.items():
            evaluations = [(r["controls"] if method=="gru-reset" else r["evaluation"])[split]
                           for r in result["runs"] if r["method"] == ("gru" if method=="gru-reset" else method)]
            for metric,stored in scores.items():
                values = [e["metrics"][metric] for e in evaluations]
                assert values == stored["values"] and abs(statistics.mean(values)-stored["mean"])<1e-12
                assert abs(statistics.stdev(values)-stored["std"])<1e-12
    for split,comparisons in result["paired"].items():
        assert [c["seed"] for c in comparisons] == [0,1,2]
        for comparison in comparisons:
            runs = {r["method"]:r for r in result["runs"] if r["seed"]==comparison["seed"]}
            old = runs["current"]["evaluation"][split]
            subset = {r["map_seed"] for r in old["episodes"] if r["loop_timeout"]}
            assert len(subset) == comparison["current_loop_cases"]
            for method,stored in comparison["methods"].items():
                evaluation = runs[method]["evaluation"][split]
                assert stored["success_delta"] == evaluation["metrics"]["success_rate"]-old["metrics"]["success_rate"]
                cases = [r for r in evaluation["episodes"] if r["map_seed"] in subset]
                assert stored["old_loop_outcomes"] == {o:sum(r["outcome"]==o for r in cases) for o in ["success","hazard","timeout"]}
    assert not {key(s) for s in splits["development"]} & {key(s) for s in splits["standard"]}
    print(f"Audited {checked} r1 episodes: full-prefix neural actions, outcomes, loops, paired recoveries, controls and disjoint splits.")

if __name__ == "__main__": main()
