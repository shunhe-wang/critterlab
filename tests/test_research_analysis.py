import numpy as np

from critterlab.research_report import interval


def test_paired_seed_bootstrap_matches_exact_two_seed_distribution():
    # With differences -1 and +1, the two-draw bootstrap means are -1, 0, +1
    # with probabilities 1/4, 1/2, 1/4; 2.5/97.5 percentiles are exactly -1/+1.
    result = interval(np.array([-1., 1.]), {"bootstrap_seed": 11, "bootstrap_resamples": 10000})
    assert result["mean"] == 0
    assert result["low"] == -1
    assert result["high"] == 1
    assert result["paired_values"] == [-1., 1.]


def test_bootstrap_of_constant_effect_has_zero_width():
    result = interval(np.full(10, .125), {"bootstrap_seed": 11, "bootstrap_resamples": 10000})
    assert result["mean"] == result["low"] == result["high"] == .125
