"""Expected values from hand calculations and the written research protocol."""
import json
import math

import numpy as np
import pytest
import torch

from critterlab.research import (
    Actor, PROTOCOL, ResearchWorld, build_manifest, clipped_objective,
    evaluate, garden_from, generalized_advantages, identity, imitation_weight,
    learn, update_budget,
)
from critterlab.world import Garden, route


def test_clipping_handles_advantage_signs():
    # Positive advantages cap profitable increases; negative cap decreases.
    ratio = torch.tensor([1.5, .5, 1.5, .5])
    advantages = torch.tensor([2., 2., -2., -2.])
    torch.testing.assert_close(clipped_objective(ratio, advantages),
                               torch.tensor([2.4, 1., -3., -1.6]))


def test_gae_terminal_cuts_bootstrap_and_cross_episode_trace():
    # Final terminal delta = 1-.4=.6, regardless of bootstrap value 100.
    # Previous delta = .1+.99*.4-.2=.296; A = .296+.99*.95*.6=.8603.
    rewards = torch.tensor([[.1], [1.]])
    values = torch.tensor([[.2], [.4]])
    observed = generalized_advantages(rewards, torch.tensor([[0.], [1.]]), values, torch.tensor([100.]))
    torch.testing.assert_close(observed, torch.tensor([[.8603], [.6]]))
    observed = generalized_advantages(rewards, torch.ones(2, 1), values, torch.tensor([100.]))
    torch.testing.assert_close(observed, torch.tensor([[-.1], [.6]]))


def test_no_shaping_changes_ordinary_reward_only():
    cells = [1 if x in (0, 6) or y in (0, 6) else 0 for y in range(7) for x in range(7)]
    world = ResearchWorld(Garden(cells, 8, 10, 0), shaping=0)
    assert world.step(0)[1] == -.08
    assert world.step(1)[1] == pytest.approx(-.02)
    assert world.step(1)[1] == 1.


def test_budget_and_gate_follow_protocol():
    spec = json.loads(PROTOCOL.read_text())
    assert update_budget("ppo", spec) == (0, 1024)
    assert update_budget("bc-ppo", spec) == (128, 896)
    assert update_budget("ppo-short", spec) == (0, 896)
    assert imitation_weight("adaptive-imitation", .5, math.log(4) / 2) == pytest.approx(.025)
    assert imitation_weight("adaptive-imitation", 0, math.log(4)) == 0
    assert imitation_weight("fixed-imitation", 0, 0) == .1


def test_manifest_is_globally_world_disjoint_and_solvable():
    spec = json.loads(PROTOCOL.read_text())
    manifest = build_manifest(spec)
    records = [r for pool in manifest.values() for r in pool]
    assert len({r["id"] for r in records}) == len(records)
    for record in records:
        garden = garden_from(record)
        assert record["id"] == identity(garden)
        assert len(route(garden.cells, garden.start, garden.goal)) >= 4
    for record in manifest["test/barrier"]:
        assert (record["start"] % 7 < 3) != (record["goal"] % 7 < 3)
        assert sum(record["cells"][y * 7 + 3] == 0 for y in range(1, 6)) == 1


def test_training_accounting_and_export_roundtrip():
    torch.set_num_threads(1)
    spec = json.loads(PROTOCOL.read_text())
    spec.update(environment_steps=2048, actor_update_budget=32, pretraining_updates=4,
                demonstration_worlds=8, training_worlds_per_family=8, evaluation_worlds_per_family=8)
    manifest = build_manifest(spec)
    from critterlab.research import demo_data
    observations, labels, _ = demo_data(manifest, 8, 0)
    for method, expected in [("ppo", (32, 32)), ("bc-ppo", (32, 28)), ("dagger", (32, 0))]:
        torch.manual_seed(3)
        actor = Actor()
        before = [p.detach().clone() for p in actor.parameters()]
        curve = []
        account = learn(actor, method, 3, spec, manifest["train/barrier"], observations, labels, 1., curve)
        assert (account["actor_updates"], account["critic_updates"]) == expected
        assert account["environment_steps"] == 2048
        assert account["new_expert_queries"] <= len(labels)
        assert any(not torch.equal(a, b) for a, b in zip(before, actor.parameters()))
        restored = Actor.restore(actor.export())
        torch.testing.assert_close(actor(observations), restored(observations), rtol=0, atol=0)
        before_rng = torch.get_rng_state().clone()
        score = evaluate(actor, manifest["test/barrier"], sampled=True)
        assert torch.equal(before_rng, torch.get_rng_state())
        assert sum(score["metrics"][k] for k in ("success_rate", "hazard_rate", "timeout_rate")) == 1
        assert np.isfinite([p["target_probe"]["entropy"] for p in curve]).all()
