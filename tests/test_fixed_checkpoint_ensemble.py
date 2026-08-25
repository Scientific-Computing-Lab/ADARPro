import numpy as np
import pytest
import torch

from adarpro.models.fixed_checkpoint_ensemble import (
    equal_logit_average,
    equal_logit_average_many,
    model_for_mode,
)


def test_equal_logit_average_is_symmetric_and_bounded():
    left = np.array([0.2, 0.5, 0.9])
    right = np.array([0.8, 0.5, 0.4])
    observed = equal_logit_average(left, right)
    reverse = equal_logit_average(right, left)
    assert np.allclose(observed, reverse)
    assert np.all((observed > 0) & (observed < 1))
    assert observed[0] == 0.5
    assert observed[1] == 0.5


def test_equal_logit_average_keeps_identical_probabilities():
    probability = np.array([0.1, 0.4, 0.7, 0.99])
    assert np.allclose(equal_logit_average(probability, probability), probability)


def test_equal_logit_average_many_uses_equal_weights():
    probabilities = [
        np.array([0.2, 0.8]),
        np.array([0.5, 0.5]),
        np.array([0.8, 0.2]),
    ]
    assert np.allclose(equal_logit_average_many(probabilities), [0.5, 0.5])


def test_equal_logit_average_many_rejects_invalid_components():
    with pytest.raises(ValueError, match="at least two"):
        equal_logit_average_many([np.array([0.5])])
    with pytest.raises(ValueError, match="different shapes"):
        equal_logit_average_many([np.array([0.5]), np.array([0.5, 0.6])])


def test_local_component_model_is_constructible():
    model = model_for_mode("binary_pair_balanced_local", torch.device("cpu"))
    assert model.use_local
