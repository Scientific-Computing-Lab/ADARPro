import numpy as np

from adarpro.models.foundation_residual import shuffle_within_folds


def test_shuffle_within_folds_preserves_each_fold_distribution():
    values = np.arange(24).reshape(8, 3)
    folds = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    shuffled = shuffle_within_folds(values, folds, seed=17)
    for fold in np.unique(folds):
        expected = values[folds == fold]
        observed = shuffled[folds == fold]
        assert sorted(map(tuple, expected)) == sorted(map(tuple, observed))


def test_shuffle_within_folds_is_deterministic():
    values = np.arange(30).reshape(10, 3)
    folds = np.repeat(np.arange(5), 2)
    assert np.array_equal(
        shuffle_within_folds(values, folds, seed=3),
        shuffle_within_folds(values, folds, seed=3),
    )
