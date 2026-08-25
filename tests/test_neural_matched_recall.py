import numpy as np
from sklearn.metrics import recall_score

from adarpro.models.residual import matched_recall_threshold


def test_matched_recall_threshold_meets_requested_recall():
    labels = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.4, 0.8, 0.2])
    threshold = matched_recall_threshold(labels, probabilities, 0.5)
    predictions = probabilities >= threshold
    assert predictions[labels == 1].mean() >= 0.5


def test_matched_recall_threshold_matches_exhaustive_search():
    rng = np.random.default_rng(7)
    for _ in range(20):
        labels = np.r_[np.ones(13, dtype=int), np.zeros(17, dtype=int)]
        probabilities = rng.uniform(size=len(labels))
        target = rng.choice([0.0, 3 / 13, 7 / 13, 1.0])
        candidates = np.unique(np.r_[0.0, probabilities, 1.0])
        expected = max(
            value
            for value in candidates
            if recall_score(labels, probabilities >= value, zero_division=0)
            >= target
        )
        assert matched_recall_threshold(labels, probabilities, target) == expected
