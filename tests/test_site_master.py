import pandas as pd

from adarpro.site_master import assign_balanced_pair_folds


def test_pair_folds_are_deterministic_and_disjoint() -> None:
    sites = pd.DataFrame(
        {
            "pair_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "context": ["Artery", "Brain"] * 5,
            "label": [0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
        }
    )
    first = assign_balanced_pair_folds(sites, n_splits=2, seed=7)
    second = assign_balanced_pair_folds(sites, n_splits=2, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first["pair_id"].is_unique
    assert set(first["development_fold"]) == {0, 1}
