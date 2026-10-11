"""Generate the declared paired-seed analysis, figures and readable research note."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .research import ROOT, FAMILIES, garden_from, read_result
from .world import World, route

LABELS = {"bc": "BC", "ppo": "PPO", "ppo-short": "PPO-short", "bc-ppo": "BC→PPO",
          "ppo-entropy": "PPO + entropy", "bc-ppo-entropy": "BC→PPO + entropy",
          "fixed-imitation": "Fixed imitation", "adaptive-imitation": "Adaptive imitation", "dagger": "DAgger-style"}


def interval(values, spec):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(spec["bootstrap_seed"])
    means = values[rng.integers(len(values), size=(spec["bootstrap_resamples"], len(values)))].mean(1)
    lo, hi = np.quantile(means, [.025, .975])
    return {"mean": float(values.mean()), "low": float(lo), "high": float(hi), "paired_values": values.tolist()}


def values(result, method, family, setting="main", mode="greedy", test_family=None):
    runs = [r for r in result["runs"] if (r["method"], r["family"], r["setting"]) == (method, family, setting)]
    runs.sort(key=lambda r: r["seed"])
    assert [r["seed"] for r in runs] == result["spec"]["training_seeds"]
    return np.array([r["evaluation"][test_family or family][mode]["metrics"]["success_rate"] for r in runs])


def format_effect(effect):
    return f"{100 * effect['mean']:+.2f} [{100 * effect['low']:+.2f}, {100 * effect['high']:+.2f}]"


def analyze(result):
    spec = result["spec"]
    analysis = {"protocol": result["protocol"], "primary": {}, "secondary": {}, "ablations": {}, "means": {}}
    for family in spec["families"]:
        analysis["means"][family] = {m: {mode: interval(values(result, m, family, mode=mode), spec)
                                          for mode in ("greedy", "sampled")} for m in spec["methods"]}
        main_delta = values(result, "bc-ppo", family) - values(result, "ppo", family)
        analysis["primary"][family] = interval(main_delta, spec)
        analysis["secondary"][family] = {
            "initialization_at_matched_ppo_updates": interval(values(result, "bc-ppo", family) - values(result, "ppo-short", family), spec),
            "entropy_difference_in_differences": interval(
                values(result, "bc-ppo-entropy", family) - values(result, "bc-ppo", family)
                - values(result, "ppo-entropy", family) + values(result, "ppo", family), spec),
            "adaptive_minus_fixed": interval(values(result, "adaptive-imitation", family) - values(result, "fixed-imitation", family), spec),
            "adaptive_minus_bc_ppo": interval(values(result, "adaptive-imitation", family) - values(result, "bc-ppo", family), spec),
        }
    family = spec["ablation_family"]
    for override in spec["ablations"]:
        setting = override["name"]
        delta = values(result, "bc-ppo", family, setting) - values(result, "ppo", family, setting)
        main_delta = values(result, "bc-ppo", family) - values(result, "ppo", family)
        analysis["ablations"][setting] = {"effect": interval(delta, spec), "change_from_main": interval(delta - main_delta, spec)}
    return analysis


def figures(result, analysis, directory):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "critterlab-imitation-shift-v1"})
    effects = [analysis["primary"][f] for f in FAMILIES]
    fig, ax = plt.subplots(figsize=(7.5, 3.2), constrained_layout=True)
    means = np.array([100 * e["mean"] for e in effects])
    limits = np.array([[100 * e["low"], 100 * e["high"]] for e in effects])
    ax.errorbar(means, np.arange(3), xerr=np.array([means - limits[:, 0], limits[:, 1] - means]),
                fmt="o", color="#32705d", capsize=5, markersize=7)
    for i, effect in enumerate(effects):
        ax.scatter(100 * np.array(effect["paired_values"]), np.full(10, i + .15),
                   s=15, alpha=.4, color="#32705d")
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set(yticks=np.arange(3), yticklabels=FAMILIES, xlabel="BC→PPO − PPO success (percentage points)",
           title="Adaptation at matched interaction and actor-update budgets")
    ax.invert_yaxis()
    fig.savefig(directory / "paired-effects.svg", metadata={"Date": None})
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, constrained_layout=True)
    colors = ["#526b82", "#32705d", "#9f7497", "#c2813e"]
    methods = ["ppo", "bc-ppo", "ppo-entropy", "bc-ppo-entropy"]
    for ax, family in zip(axes, FAMILIES):
        for color, method in zip(colors, methods):
            runs = [r for r in result["runs"] if (r["setting"], r["family"], r["method"]) == ("main", family, method)]
            matrix = np.array([[p["target_probe"]["entropy"] for p in r["curve"]] for r in runs])
            steps = [p["steps"] for p in runs[0]["curve"]]
            ax.plot(steps, matrix.mean(0), label=LABELS[method], color=color)
            ax.fill_between(steps, matrix.mean(0) - matrix.std(0, ddof=1),
                            matrix.mean(0) + matrix.std(0, ddof=1), alpha=.1, color=color)
        ax.set(title=family, xlabel="Environment steps")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].set_ylabel("Entropy on fixed target-start probes (nats)")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Mean ± seed SD; changing occupancy does not change these probe inputs")
    fig.savefig(directory / "probe-entropy.svg", metadata={"Date": None})
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)
    methods = ["bc", "ppo", "bc-ppo", "bc-ppo-entropy", "adaptive-imitation", "dagger"]
    for ax, family in zip(axes, FAMILIES):
        x = np.arange(len(methods))
        for offset, mode, color in [(-.18, "greedy", "#32705d"), (.18, "sampled", "#c2813e")]:
            ax.bar(x + offset, [100 * analysis["means"][family][m][mode]["mean"] for m in methods],
                   width=.36, label=mode, color=color)
        ax.set(title=family, ylim=(0, 100), xticks=x, xticklabels=[LABELS[m] for m in methods])
        ax.tick_params(axis="x", rotation=70)
    axes[0].set_ylabel("Held-out target success (%)")
    axes[-1].legend()
    fig.savefig(directory / "action-selection.svg", metadata={"Date": None})
    plt.close(fig)


def failures(result, path):
    lines = ["# Exploratory failure traces", "", "Selection rule: first failing frozen test world for the final seed-0 BC→PPO",
             "actor trained on that family. This case selection is post-hoc; it is not",
             "used to choose models or estimate effects. Full maps and actions are in the raw artifact.", ""]
    for family in FAMILIES:
        run = next(r for r in result["runs"] if (r["setting"], r["family"], r["method"], r["seed"]) == ("main", family, "bc-ppo", 0))
        record = next((e for e in run["evaluation"][family]["greedy"]["episodes"] if e["outcome"] != "success"), None)
        if record is None:
            lines += [f"## {family}", "", "No failure in this seed's frozen panel.", ""]
            continue
        garden_record = next(g for g in result["manifest"][f"test/{family}"] if g["id"] == record["world_id"])
        garden = garden_from(garden_record)
        world = World(garden)
        positions = [world.pos]
        for action in record["actions"]:
            world.step(action)
            positions.append(world.pos)
        symbols = [".#!"[c] for c in garden.cells]
        symbols[garden.start], symbols[garden.goal] = "S", "G"
        lines += [f"## {family}: {record['outcome']}", "", f"World `{garden_record['id']}` (generator seed {garden.seed}).",
                  "", "```text", *["".join(symbols[y * 7:(y + 1) * 7]) for y in range(7)], "```", "",
                  "`#` rock, `!` hazard, `S` start, `G` goal. Coordinates are flat row-major indices.", "",
                  f"Expert actions: `{route(garden.cells, garden.start, garden.goal)}`.", "",
                  f"Policy actions: `{record['actions']}`.", "", f"Visited positions: `{positions}`.", "",
                  f"Unique positions: {len(set(positions))}; recorded actions: {record['steps']}.", "",
                  "A repeated position under greedy reactive inference can produce a deterministic loop.",
                  "This trace establishes behavior, not whether entropy, aliasing, or optimization caused it.", ""]
    path.write_text("\n".join(lines) + "\n")


def report(result, analysis, path, raw_path):
    spec = result["spec"]
    lines = ["# Imitation under distribution shift: a controlled CritterLab study", "",
             "Executed CPU experiment, 2026-10-11. Developed with AI assistance. This is",
             "an empirical research artifact in a small navigation environment; it is not",
             "a claim of a new algorithm or independently validated discovery.", "",
             f"**{len(result['runs'])} training runs, ten paired seeds, three terrain families.**",
             "The primary comparison matches 65,536 environment interactions and 1,024",
             "actor-update slots. BC→PPO allocates 128 slots to imitation and 896 to PPO;",
             "PPO allocates all 1,024 to PPO. Extra expert information is reported, not erased.", "",
             "[Protocol](research-protocol.md) · [Primary-source background](research-background.md) ·",
             "[Raw maps, all actors and episodes](../research/results.json.gz) ·",
             "[Machine-readable analysis](../research/analysis.json) · [Failure traces](research-failures.md)", "",
             "## Primary finding", "", "Effects are **BC→PPO minus PPO**, in percentage points, on held-out worlds",
             "of the family used for PPO training. Brackets are 95% paired-seed percentile",
             "bootstrap intervals, conditional on the fixed map and demonstration panels.", "",
             "| Adaptation terrain | Mean effect [95% interval] | Paired seed effects |",
             "|---|---:|---|" ]
    for family in FAMILIES:
        effect = analysis["primary"][family]
        lines.append(f"| {family} | {format_effect(effect)} | " + ", ".join(f"{100 * v:+.1f}" for v in effect["paired_values"]) + " |")
    lines += ["", "![Paired effects with seed-level uncertainty](research-figures/paired-effects.svg)", "",
              "## All methods", "", "Mean goal success across ten trained seeds. Each seed gets the same 128 frozen",
              "worlds per evaluation family. Greedy inference matches the browser policy;", "sampled inference tests the training policy's action distribution.", "",
              "| Method | Standard greedy / sampled | Dense greedy / sampled | Barrier greedy / sampled |",
              "|---|---:|---:|---:|"]
    for method in spec["methods"]:
        cells = []
        for family in FAMILIES:
            mean = analysis["means"][family][method]
            cells.append(f"{100 * mean['greedy']['mean']:.2f}% / {100 * mean['sampled']['mean']:.2f}%")
        lines.append(f"| {LABELS[method]} | " + " | ".join(cells) + " |")
    lines += ["", "BC repeats the same source-only model across training-family blocks; these are",
              "different test panels, not extra independent BC training seeds. DAgger queries",
              "additional target-state expert labels and has a different supervision budget.", "",
              "![Greedy and sampled policies](research-figures/action-selection.svg)", "",
              "## Mechanism tests and adaptive method", "",
              "All following intervals are exploratory and uncorrected for multiple comparisons.", "",
              "| Contrast (percentage points) | Standard | Dense | Barrier |",
              "|---|---:|---:|---:|"]
    for key, label in [("initialization_at_matched_ppo_updates", "BC→PPO − PPO-short"),
                       ("entropy_difference_in_differences", "BC entropy gain − PPO entropy gain"),
                       ("adaptive_minus_fixed", "Adaptive − fixed imitation"),
                       ("adaptive_minus_bc_ppo", "Adaptive − BC→PPO")]:
        lines.append(f"| {label} | " + " | ".join(format_effect(analysis["secondary"][f][key]) for f in FAMILIES) + " |")
    lines += ["", "The entropy contrast subtracts the benefit of stronger entropy in PPO itself.",
              "A BC-specific gain, lower initial entropy, and improved visitation would be",
              "consistent with an exploration account. They would not rule out objective,",
              "occupancy, aliasing, or optimization explanations.", "",
              "![Fixed-observation entropy trajectories](research-figures/probe-entropy.svg)", "",
              "| Terrain / method | Initial target-probe entropy | Final target-probe entropy | Unique visited world/positions | Collision fraction |",
              "|---|---:|---:|---:|---:|"]
    for family in FAMILIES:
        for method in ("ppo", "bc-ppo", "ppo-entropy", "bc-ppo-entropy", "adaptive-imitation"):
            runs = [r for r in result["runs"] if (r["setting"], r["family"], r["method"]) == ("main", family, method)]
            mean = lambda get: float(np.mean([get(r) for r in runs]))
            lines.append(f"| {family} / {LABELS[method]} | {mean(lambda r:r['curve'][0]['target_probe']['entropy']):.3f} | "
                         f"{mean(lambda r:r['curve'][-1]['target_probe']['entropy']):.3f} | "
                         f"{mean(lambda r:r['accounting']['unique_world_positions']):.1f} | "
                         f"{mean(lambda r:r['accounting']['collisions']/r['accounting']['environment_steps']):.3f} |")
    lines += ["", "Visitation counts depend on episode termination and the finite pool; more",
              "visited states are not automatically useful exploration. Entropy is in nats.", "",
              "## One-factor ablations", "", "Barrier adaptation, ten seeds per condition. Change from main subtracts the",
              "primary barrier effect using the same seed blocks. This is not a full factorial.", "",
              "| Change | BC→PPO − PPO [95% interval] | Effect change from main [95% interval] |",
              "|---|---:|---:|"]
    for name, ablation in analysis["ablations"].items():
        lines.append(f"| {name} | {format_effect(ablation['effect'])} | {format_effect(ablation['change_from_main'])} |")
    lines += ["", "## Zero-shot transfer is a separate question", "",
              "These models are trained **only on standard terrain** and evaluated without",
              "any dense/barrier adaptation. Do not interpret this table as target learning.", "",
              "| Standard-trained method | Standard | Dense | Barrier |", "|---|---:|---:|---:|"]
    for method in ("ppo", "bc-ppo", "adaptive-imitation"):
        lines.append(f"| {LABELS[method]} | " + " | ".join(f"{100 * values(result, method, 'standard', test_family=f).mean():.2f}%" for f in FAMILIES) + " |")
    lines += ["", "## What changed between terrain families?", "",
              "Descriptors below are for the frozen test panels. Barriers alter topology",
              "and hidden information, not only an abstract complexity level.", "",
              "| Test family | Worlds | Shortest path | Manhattan distance | Detour excess | Interior rock fraction | Interior hazard fraction |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for family in FAMILIES:
        pool = result["manifest"][f"test/{family}"]
        lengths = [len(route(r["cells"], r["start"], r["goal"])) for r in pool]
        manhattan = [abs(r["start"] % 7 - r["goal"] % 7) + abs(r["start"] // 7 - r["goal"] // 7) for r in pool]
        interior = [r["cells"][y * 7 + x] for r in pool for y in range(1, 6) for x in range(1, 6)]
        lines.append(f"| {family} | {len(pool)} | {np.mean(lengths):.2f} | {np.mean(manhattan):.2f} | "
                     f"{np.mean(np.array(lengths)-manhattan):.2f} | {interior.count(1)/len(interior):.3f} | {interior.count(2)/len(interior):.3f} |")
    lines += ["", "## Resource accounting", "",
              "Means across main-study family/seed blocks. Updates and label presentations",
              "are counts, not equal FLOPs. All source demonstrations are reused across seeds.", "",
              "| Method | Env steps | Actor / critic updates | Source label presentations | New target expert queries | Training seconds |",
              "|---|---:|---:|---:|---:|---:|"]
    for method in spec["methods"]:
        runs = [r for r in result["runs"] if r["setting"] == "main" and r["method"] == method]
        mean = lambda key: float(np.mean([r["accounting"][key] for r in runs]))
        lines.append(f"| {LABELS[method]} | {mean('environment_steps'):.0f} | {mean('actor_updates'):.0f} / {mean('critic_updates'):.0f} | "
                     f"{mean('source_label_presentations'):.0f} | {mean('new_expert_queries'):.0f} | {np.mean([r['training_seconds'] for r in runs]):.2f} |")
    lines += ["", f"Total elapsed wall time: {result['total_seconds']:.1f} seconds (including evaluation and serialization).",
              f"Runtime: `{result['platform']}`, PyTorch `{result['torch']}`, NumPy `{result['numpy']}`, CPU.", "",
              "## Reproduce and audit", "", "```sh", "uv sync --locked",
              "uv run python -m critterlab.research --output research/reproduced.json.gz",
              "uv run python -m critterlab.research_audit research/reproduced.json.gz",
              "uv run python -m critterlab.research_report research/reproduced.json.gz --output /tmp/critterlab-report",
              "```", "", "For a short harness run, use `--smoke --output /tmp/critterlab-research-smoke.json.gz`.",
              "Smoke results are explicitly marked non-protocol. The full artifact contains",
              "all final actor weights and 268,800 evaluation action sequences, and the audit",
              "recomputes outcomes, greedy/sampled inference, metrics, split identities, and budgets.", "",
              f"Raw artifact SHA-256: `{hashlib.sha256(raw_path.read_bytes()).hexdigest()}`.", "",
              "## Limits and next falsifying experiments", "",
              "This run conditions on one source dataset and one finite pool per family.",
              "Seed bootstrap intervals measure training variation, not map-population or",
              "demonstration-sampling uncertainty. Secondary results are exploratory; no",
              "multiple-comparison correction or independent replication is claimed.", "",
              "The actor has no memory and sees less than the expert. We have not controlled",
              "aliasing with a full-observation or recurrent model. Reward shaping is not",
              "guaranteed policy-invariant, and label corruption is not a coherent poor expert.",
              "An adaptive heuristic has prior conceptual neighbors; a better score would",
              "not establish novelty. Its gate also lowers the average imitation penalty",
              "relative to the fixed .1 baseline: any advantage could be penalty strength",
              "rather than adaptation. A matched-average constant/time schedule is still needed.",
              "No holdout was used for checkpoint selection or tuning.", "",
              "Useful next tests: independently resample demonstrations and map pools;",
              "control observation aliasing with memory/full-state policies; compare the gate",
              "with a matched time schedule and target demonstrations; independently reproduce",
              "the most informative boundary on a larger environment. A null or failed adaptive",
              "method remains evidence if its limitations are reported."]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=ROOT / "research/results.json.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    result = read_result(args.path)
    if not result.get("complete") or not result["protocol_run"]:
        parser.error("Report requires a complete full-protocol run")
    analysis = analyze(result)
    args.output.mkdir(parents=True, exist_ok=True)
    directory = args.output / "research-figures"
    directory.mkdir(exist_ok=True)
    figures(result, analysis, directory)
    report(result, analysis, args.output / "research-results.md", args.path)
    failures(result, args.output / "research-failures.md")
    # Regeneration outside docs must not overwrite the committed analysis artifact.
    analysis_path = ROOT / "research/analysis.json" if args.output == ROOT / "docs" else args.output / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis["primary"], indent=2))


if __name__ == "__main__":
    main()
