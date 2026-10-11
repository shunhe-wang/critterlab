import numpy as np
import pytest
import torch

from critterlab.research import Actor
from critterlab.research_audit import verify_actions


def uniform_actor_and_pool():
    actor = Actor()
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
    # All logits zero: hand-worked categorical probabilities exactly 1/4.
    cells = [1 if x in (0, 6) or y in (0, 6) else 0 for y in range(7) for x in range(7)]
    return actor, [{"cells": cells, "start": 8, "goal": 10, "seed": 0}]


def test_inference_audit_accepts_ties_but_rejects_wrong_greedy_actions():
    actor, pool = uniform_actor_and_pool()
    assert verify_actions(actor, pool, [{"actions": [1, 1]}], False) == 2
    with torch.no_grad():
        actor.net[-1].bias[0] = 10
    with pytest.raises(AssertionError, match="Greedy action"):
        verify_actions(actor, pool, [{"actions": [1]}], False)


def test_sampled_audit_obeys_uniform_categorical_intervals():
    actor, pool = uniform_actor_and_pool()
    uniforms = np.random.default_rng(19123).random((48, 1))
    # Inverse CDF of [1/4,1/4,1/4,1/4]: action j occupies [j/4,(j+1)/4).
    expected = np.floor(4 * uniforms[:2, 0]).astype(int).tolist()
    assert verify_actions(actor, pool, [{"actions": expected}], True) == 0
    with pytest.raises(AssertionError, match="Sample incompatible"):
        verify_actions(actor, pool, [{"actions": [(expected[0] + 1) % 4]}], True)
