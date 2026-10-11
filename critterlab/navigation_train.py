"""Run frozen r1: sequence BC then correctly reset, truncated-sequence PPO."""
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .world import RNG, World, generate, route
from .model import Policy
from .navigation_model import GardenGRU

ROOT = Path(__file__).resolve().parents[1]
STEPS = 65536
DIAGNOSTICS = [(67,False),(67,True),(1100000,False),(1100001,False)]

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")

def demonstrations(seeds):
    episodes = []
    for seed in seeds:
        world = World(generate(seed)); obs, actions = [], route(world.garden.cells, world.pos, world.garden.goal)
        for action in actions:
            obs.append(world.obs().tolist()); world.step(action)
        episodes.append({"map_seed":seed,"observations":obs,"actions":actions})
    length = max(len(e["actions"]) for e in episodes)
    observations = torch.zeros(len(episodes), length, 20)
    labels = torch.zeros(len(episodes), length, dtype=torch.long)
    mask = torch.zeros(len(episodes), length)
    for i,e in enumerate(episodes):
        n = len(e["actions"]); observations[i,:n] = torch.tensor(e["observations"])
        labels[i,:n] = torch.tensor(e["actions"]); mask[i,:n] = 1
    return observations, labels, mask, episodes

def sequence(policy, obs, hidden=None, starts=None):
    return policy.sequence(obs, hidden, starts) if isinstance(policy, GardenGRU) else policy(obs)

def step(policy, obs, hidden=None):
    return policy.step(obs, hidden) if isinstance(policy, GardenGRU) else (policy(obs), None)

def clone(policy, data, epochs=30):
    obs, labels, mask, _ = data
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    losses = []
    for _ in range(epochs):
        for batch in torch.randperm(len(labels)).split(48):
            logits = sequence(policy, obs[batch])
            errors = nn.functional.cross_entropy(logits.reshape(-1,4), labels[batch].flatten(), reduction="none").reshape_as(mask[batch])
            loss = (errors * mask[batch]).sum() / mask[batch].sum()
            if not torch.isfinite(loss): raise RuntimeError("Non-finite BC loss")
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
    return losses

def cycle(positions):
    return len(positions) >= 6 and all(positions[-6+i] == positions[-6+i%2] for i in range(6))

def metrics(records):
    n = len(records); lengths = [r["steps"] for r in records if r["outcome"] == "success"]
    return {"episodes":n, **{f"{outcome}_rate":sum(r["outcome"] == outcome for r in records)/n for outcome in ["success","hazard","timeout"]},
            "any_cycle_rate":sum(r["any_cycle"] for r in records)/n,
            "loop_timeout_rate":sum(r["loop_timeout"] for r in records)/n,
            "mean_success_steps":statistics.mean(lengths) if lengths else None}

def rollout(policy, seed, shifted=False, reset=False):
    world, hidden, actions = World(generate(seed, shifted)), None, []
    positions = [world.pos]; any_cycle = False
    with torch.no_grad():
        while not world.done:
            logits, hidden = step(policy, torch.tensor(world.obs())[None], None if reset else hidden)
            action = int(logits.argmax(-1).item()); actions.append(action); world.step(action); positions.append(world.pos)
            any_cycle |= cycle(positions)
    return {"map_seed":seed,"shifted":shifted,"actions":actions,"outcome":world.outcome,"steps":world.steps,
            "any_cycle":any_cycle,"loop_timeout":world.outcome == "timeout" and cycle(positions)}

def evaluate(policy, seeds, shifted=False, reset=False):
    records = [rollout(policy,s,shifted,reset) for s in seeds]
    return {"metrics":metrics(records),"episodes":records}

def gae(rewards, values, dones, bootstrap):
    advantages = torch.zeros_like(rewards); running = torch.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        following = bootstrap if t == len(rewards)-1 else values[t+1]
        alive = 1-dones[t]
        delta = rewards[t] + .99*following*alive - values[t]
        running = delta + .99*.95*alive*running; advantages[t] = running
    return advantages

def chunks(tensor):
    """Time×environment → chunk×environment-major sequence, preserving time."""
    return tensor.reshape(8,16,16,*tensor.shape[2:]).transpose(1,2).reshape(128,16,*tensor.shape[2:])

def prefix_state(policy, prefixes):
    if not isinstance(policy, GardenGRU): return None
    hidden = torch.zeros(len(prefixes),32)
    if not any(prefixes): return hidden
    padded = torch.zeros(len(prefixes),max(map(len,prefixes)),20)
    for i,prefix in enumerate(prefixes):
        if prefix: padded[i,:len(prefix)] = torch.tensor(np.array(prefix))
    with torch.no_grad():
        outputs,_ = policy.gru(padded)
        for i,prefix in enumerate(prefixes):
            if prefix: hidden[i] = outputs[i,len(prefix)-1]
    return hidden

def ppo(policy, seed, budget, pool, callback):
    torch.manual_seed(seed+200)
    critic = nn.Sequential(nn.Linear(20,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
    parameters = list(policy.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(parameters,lr=.0003,eps=1e-5)
    rng, map_seeds = RNG(seed+881), []
    def next_world():
        s = pool[rng.integer(len(pool))]; map_seeds.append(s); return World(generate(s))
    worlds, prefixes = [next_world() for _ in range(16)], [[] for _ in range(16)]
    updates = 0
    for iteration in range(budget//2048):
        hidden = prefix_state(policy,prefixes)
        log = {key:[] for key in ["obs","actions","probs","values","rewards","dones","starts","hidden"]}
        for _ in range(128):
            obs = torch.tensor(np.array([w.obs() for w in worlds]))
            log["hidden"].append(hidden.clone() if hidden is not None else torch.zeros(16,32))
            log["starts"].append(torch.tensor([not prefix for prefix in prefixes]))
            with torch.no_grad():
                logits,hidden = step(policy,obs,hidden)
                distribution = Categorical(logits=logits); actions = distribution.sample(); values = critic(obs).squeeze(-1)
            rewards, dones = [], []
            for i,action in enumerate(actions.tolist()):
                prefixes[i].append(obs[i].numpy().copy())
                _,reward,done = worlds[i].step(action); rewards.append(reward); dones.append(done)
                if done:
                    worlds[i] = next_world(); prefixes[i] = []
                    if hidden is not None: hidden[i] = 0
            for key,value in [("obs",obs),("actions",actions),("probs",distribution.log_prob(actions)),("values",values),
                              ("rewards",torch.tensor(rewards)),("dones",torch.tensor(dones,dtype=torch.float32))]: log[key].append(value)
        log = {key:torch.stack(value) for key,value in log.items()}
        with torch.no_grad(): bootstrap = critic(torch.tensor(np.array([w.obs() for w in worlds]))).squeeze(-1)
        advantages = gae(log["rewards"],log["values"],log["dones"],bootstrap)
        returns = chunks(advantages+log["values"])
        advantages = chunks((advantages-advantages.mean())/(advantages.std()+1e-8))
        obs,actions,old_probs,starts = [chunks(log[k]) for k in ["obs","actions","probs","starts"]]
        initial = log["hidden"][::16].reshape(128,32).detach()
        for _ in range(4):
            for batch in torch.randperm(128).split(16):
                logits = sequence(policy,obs[batch],initial[batch],starts[batch])
                distribution = Categorical(logits=logits)
                ratio = (distribution.log_prob(actions[batch])-old_probs[batch]).exp()
                objective = torch.minimum(ratio*advantages[batch],ratio.clamp(.8,1.2)*advantages[batch])
                value_loss = (critic(obs[batch]).squeeze(-1)-returns[batch]).square().mean()
                loss = -objective.mean()+.5*value_loss-.01*distribution.entropy().mean()
                if not torch.isfinite(loss): raise RuntimeError("Non-finite PPO loss")
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(parameters,.5); optimizer.step(); updates += 1
        count = (iteration+1)*2048
        if count%8192 == 0 or count == budget: callback(count,policy)
    return updates,map_seeds

def aggregate(values):
    return {"values":values,"mean":statistics.mean(values),"std":statistics.stdev(values) if len(values)>1 else 0}

def summarize(result):
    scores,paired = {},{}
    for method in ["gru","reactive","current","gru-reset"]:
        scores[method] = {}
        for split in ["standard","wilder"]:
            evaluations = [(r["controls"] if method == "gru-reset" else r["evaluation"])[split]
                           for r in result["runs"] if r["method"] == ("gru" if method == "gru-reset" else method)]
            scores[method][split] = {key:aggregate([e["metrics"][key] for e in evaluations])
                                    for key in ["success_rate","hazard_rate","timeout_rate","any_cycle_rate","loop_timeout_rate"]}
    for split in ["standard","wilder"]:
        paired[split] = []
        for seed in result["training_seeds"]:
            runs = {r["method"]:r for r in result["runs"] if r["seed"] == seed}
            old = runs["current"]["evaluation"][split]["episodes"]
            subset = {r["map_seed"] for r in old if r["loop_timeout"]}
            comparison = {"seed":seed,"current_loop_cases":len(subset),"methods":{}}
            for method in ["gru","reactive"]:
                evaluation = runs[method]["evaluation"][split]
                cases = [r for r in evaluation["episodes"] if r["map_seed"] in subset]
                comparison["methods"][method] = {"success_delta":evaluation["metrics"]["success_rate"]-runs["current"]["evaluation"][split]["metrics"]["success_rate"],
                                                 "old_loop_outcomes":{outcome:sum(r["outcome"] == outcome for r in cases) for outcome in ["success","hazard","timeout"]}}
            paired[split].append(comparison)
    return scores,paired

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--publish",action="store_true")
    parser.add_argument("--steps",type=int,default=STEPS); parser.add_argument("--bc-epochs",type=int,default=30)
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2]); parser.add_argument("--output",type=Path,default=ROOT/"results/navigation-r1.json")
    args = parser.parse_args()
    if args.steps <= 0 or args.steps%2048 or args.bc_epochs<1: parser.error("positive multiples of 2048 steps and positive BC epochs required")
    if args.publish and (args.steps!=STEPS or args.bc_epochs!=30 or args.seeds!=[0,1,2]): parser.error("Publishing requires frozen r1 budgets/seeds")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    splits = json.loads((ROOT/"results/navigation-r1-splits.json").read_text()); data = demonstrations(splits["demonstration_seeds"])
    started = time.perf_counter()
    result = {"protocol":"r1","protocol_commit":subprocess.check_output(["git","log","-1","--format=%H","--","docs/navigation-memory-protocol.md"],cwd=ROOT,text=True).strip(),
              "training_seeds":args.seeds,"ppo_steps":args.steps,"bc_epochs":args.bc_epochs,"bc_transitions":sum(len(e["actions"]) for e in data[3]),
              "torch":torch.__version__,"numpy":np.__version__,"platform":platform.platform(),"device":"cpu","runs":[],"export_sha256":{}}
    parity = []
    for seed in args.seeds:
        for method,constructor in [("gru",GardenGRU),("reactive",Policy),("current",Policy)]:
            torch.manual_seed(seed); model = constructor(); before = time.perf_counter()
            run = {"seed":seed,"method":method,"parameters":sum(p.numel() for p in model.parameters()),"curve":[],"bc_losses":[],"ppo_updates":0,"training_map_seeds":[]}
            path = ROOT/(f"checkpoints/bc-ppo-seed-{seed}.pt" if method=="current" else f"checkpoints/r1-{method}-seed-{seed}.pt")
            if method=="current": model.load_state_dict(torch.load(path,weights_only=True))
            else:
                torch.manual_seed(seed+10); run["bc_losses"] = clone(model,data,args.bc_epochs)
                def progress(count,actor):
                    score = evaluate(actor,splits["development"])["metrics"]
                    run["curve"].append({"steps":count,**score}); print(f"seed={seed} {method} steps={count} dev={score['success_rate']:.3f}",flush=True)
                progress(0,model)
                run["ppo_updates"],run["training_map_seeds"] = ppo(model,seed,args.steps,splits["pool"],progress)
                if args.publish: torch.save(model.state_dict(),path)
            run["training_seconds"] = time.perf_counter()-before
            if args.publish or method=="current":
                run["checkpoint"] = str(path.relative_to(ROOT)); run["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            run["evaluation"] = {name:evaluate(model,splits[name],name=="wilder") for name in ["standard","wilder"]}
            run["diagnostics"] = [rollout(model,s,shifted) for s,shifted in DIAGNOSTICS]
            if method=="gru":
                run["controls"] = {name:evaluate(model,splits[name],name=="wilder",True) for name in ["standard","wilder"]}
                run["reset_diagnostics"] = [rollout(model,s,shifted,True) for s,shifted in DIAGNOSTICS]
            result["runs"].append(run)
            if args.publish and seed==0 and method!="current":
                exported = ROOT/f"public/models/r1-{method}.json"
                dump(exported,model.export(seed) if method=="gru" else model.export("Random-garden reactive · r1",seed,args.steps))
                result["export_sha256"][str(exported.relative_to(ROOT))] = hashlib.sha256(exported.read_bytes()).hexdigest()
                for map_seed,shifted in [(67,True),(67,False),(splits["standard"][0],False)]:
                    world,hidden,frames = World(generate(map_seed,shifted)),None,[]
                    with torch.no_grad():
                        while not world.done:
                            logits,hidden = step(model,torch.tensor(world.obs())[None],hidden); action = int(logits.argmax(-1))
                            frames.append({"observation":world.obs().tolist(),"hidden":hidden[0].tolist() if hidden is not None else None,
                                           "probabilities":logits.softmax(-1)[0].tolist(),"action":action,"pos":world.pos})
                            world.step(action)
                    parity.append({"method":method,"map_seed":map_seed,"shifted":shifted,"garden":asdict(world.garden),"frames":frames,"outcome":world.outcome})
    result["summary"],result["paired"] = summarize(result)
    sources = ["docs/navigation-memory-protocol.md","results/navigation-r1-splits.json","critterlab/navigation_splits.py","critterlab/navigation_model.py","critterlab/navigation_train.py","critterlab/world.py","critterlab/model.py"]
    result["source_sha256"] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    result["total_seconds"] = time.perf_counter()-started; dump(args.output,result)
    if args.publish:
        dump(ROOT/"tests/fixtures/navigation-parity.json",parity)
        dump(ROOT/"public/data/navigation.json",{key:result[key] for key in ["protocol","summary","paired","ppo_steps","bc_transitions","total_seconds"]})
    print(json.dumps({"summary":result["summary"],"paired":result["paired"],"seconds":result["total_seconds"]},indent=2),flush=True)

if __name__ == "__main__": main()
