"""Hand-worked m1 contract examples and checks for actual learning."""
import numpy as np
import pytest
import torch
from critterlab.memory_world import Task, MemoryWorld, TRAIN, TEST, LONG
from critterlab.memory_model import Recurrent, Reactive
from critterlab.memory_train import demonstrations, train


def test_gate_cue_and_hand_worked_branches():
    left, right = [MemoryWorld(Task(3, 2, 2, cue)) for cue in (0, 1)]
    assert left.pos == right.pos == 38
    assert left.obs()[-2:].tolist() == [1, 0]
    assert right.obs()[-2:].tolist() == [0, 1]
    left.step(3); right.step(1)
    assert left.pos == right.pos == 38
    np.testing.assert_array_equal(left.obs(), right.obs())
    left.step(0); right.step(0)
    assert left.pos == right.pos == 38
    for action in [0, 0, 0, 3, 3]: left.step(action)
    assert left.pos == 15 and left.outcome == "success" and left.steps == 7
    for action in [0, 0, 0, 3, 3]: right.step(action)
    assert right.pos == 15 and right.outcome == "wrong-berry"
    with pytest.raises(ValueError, match="already over"): left.step(0)


def test_target_does_not_leak_into_post_cue_observations():
    for column in [2, 3, 4]:
        a, b = [MemoryWorld(Task(column, 1, 8, cue)) for cue in (0, 1)]
        for action in [1] * 8 + [0] * 4:
            a.step(action); b.step(action)
            np.testing.assert_array_equal(a.obs(), b.obs())
        assert a.pos == b.pos == 7 + column
        assert a.obs()[:2].tolist() == [0, 0]


def test_expert_episodes_are_unique_and_world_disjoint():
    geometry = lambda t: tuple(MemoryWorld(t).cells)
    assert not {geometry(t) for t in TRAIN} & {geometry(t) for t in TEST + LONG}
    assert len(TRAIN) == len(set(TRAIN)) == 36
    for task in TRAIN + TEST + LONG:
        w = MemoryWorld(task)
        while not w.done: w.step(w.expert())
        assert w.outcome == "success"


def test_real_sequence_training_updates_both_models_and_padding_has_no_loss_weight():
    torch.set_num_threads(1); obs, labels, weights, episodes = demonstrations()
    assert sum(len(e["actions"]) for e in episodes) == 330
    assert weights[0, len(episodes[0]["actions"]):].sum() == 0
    for constructor in [Reactive, Recurrent]:
        torch.manual_seed(3); model = constructor()
        before = [p.detach().clone() for p in model.parameters()]
        losses, _ = train(model, obs, labels, weights, 2)
        assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        assert torch.isfinite(model(obs)).all() and len(losses) == 2


def test_recurrent_step_matches_full_sequence_and_episode_reset():
    torch.manual_seed(5); model = Recurrent(); sequence = torch.randn(2, 7, 22)
    hidden = None; logits = []
    for t in range(7):
        output, hidden = model.step(sequence[:, t], hidden); logits.append(output)
    torch.testing.assert_close(torch.stack(logits, 1), model(sequence))
    first, _ = model.step(sequence[:, 0], None)
    torch.testing.assert_close(first, model(sequence)[:, 0])
