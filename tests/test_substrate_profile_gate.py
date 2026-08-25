import numpy as np
import pandas as pd

from adarpro.models.substrate_profile_gate import (
    permute_within_pair,
    relative_features,
)


def test_relative_features_match_focal_probability():
    canonical = pd.DataFrame(
        {
            "example_id": ["a", "b"],
            "pair_id": [1, 1],
            "local_position": [2, 5],
            "expanded_probability": [0.2, 0.8],
        }
    )
    metadata = pd.DataFrame(
        {"pair_id": [1, 1, 1], "local_position": [2, 3, 5]}
    )
    result, difference = relative_features(
        canonical, metadata, np.asarray([0.2, 0.5, 0.8]), 0.5
    )
    assert difference == 0
    assert result["relative_percentile"].tolist() == [1 / 6, 5 / 6]
    assert result.loc[0, "relative_zscore"] < 0
    assert result.loc[1, "relative_zscore"] > 0


def test_within_pair_permutation_preserves_pair_multisets():
    table = pd.DataFrame(
        {
            "example_id": ["a", "b", "c", "d"],
            "pair_id": [1, 1, 1, 2],
            "x": [0.1, 0.2, 0.3, 0.9],
            "z": [-1.0, 0.0, 1.0, 2.0],
        }
    )
    result = permute_within_pair(table, ["x", "z"], 7)
    for pair_id in (1, 2):
        before = table[table["pair_id"] == pair_id][["x", "z"]].to_numpy()
        after = result[result["pair_id"] == pair_id][["x", "z"]].to_numpy()
        assert sorted(map(tuple, before)) == sorted(map(tuple, after))
    assert result.loc[result["pair_id"] == 2, "x"].item() == 0.9
