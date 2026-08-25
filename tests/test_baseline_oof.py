import numpy as np

from adarpro.models.baseline_oof import choose_threshold


def test_threshold_selection_uses_prespecified_grid() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.7, 0.8])
    threshold, score = choose_threshold(labels, probabilities)
    assert 0.1 <= threshold <= 0.9
    assert score == 1.0
