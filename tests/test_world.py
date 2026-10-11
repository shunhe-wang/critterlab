import numpy as np
import pytest
import torch

from critterlab.model import Policy
from critterlab.train import behavior_clone, demonstrations
from critterlab.world import Garden, World, generate, route


def simple_world():
    # Hand-worked contract: start (1,1), goal (3,1), rock north, hazard south.
    cells = [1 if x in (0, 6) or y in (0, 6) else 0 for y in range(7) for x in range(7)]
    cells[15] = 2
    return World(Garden(cells, 8, 10, 0))


def test_arrival_collision_and_terminal_contract():
    world = simple_world()
    _, reward, done = world.step(0)
    assert world.pos == 8 and reward == -0.08 and not done
    _, reward, done = world.step(1)
    assert world.pos == 9 and reward == pytest.approx(0.08) and not done
    _, reward, done = world.step(1)
    assert world.pos == 10 and reward == 1 and done and world.outcome == "success"
    with pytest.raises(ValueError, match="already over"):
        world.step(0)


def test_hazard_and_timeout_contract():
    world = simple_world()
    _, reward, done = world.step(2)
    assert reward == -1 and done and world.outcome == "hazard"
    world = simple_world()
    for _ in range(48):
        world.step(0)
    assert world.outcome == "timeout"


def test_observation_is_local_and_has_no_expert_path():
    obs = simple_world().obs()
    # dx=2/7, dy=0. Top row and west neighbors are boundary rocks; south is hazard.
    np.testing.assert_allclose(obs, [2/7, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 1, 0])


def test_seeded_gardens_are_reproducible_and_solvable():
    for seed in [0, 42, 1_100_003, 2_100_003]:
        garden = generate(seed, seed >= 2_000_000)
        assert garden == generate(seed, seed >= 2_000_000)
        world = World(garden)
        actions = route(garden.cells, garden.start, garden.goal)
        assert len(actions) >= 4
        for action in actions:
            world.step(action)
        assert world.outcome == "success"


def test_demonstrations_change_real_model_weights():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model = Policy()
    before = [p.detach().clone() for p in model.parameters()]
    obs, actions = demonstrations(8)
    behavior_clone(model, obs, actions, epochs=2)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert torch.isfinite(model(obs)).all()
