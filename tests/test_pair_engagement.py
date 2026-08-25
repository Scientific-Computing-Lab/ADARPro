import numpy as np
import pandas as pd

from adarpro.models.pair_engagement import fit_cross_fitted, prepare_pair_table


def test_prepare_pair_table_builds_fraction_and_rejects_intermediate_from_denominator():
    raw = pd.DataFrame(
        {
            "context": ["x"] * 6,
            "pair_id": [1, 1, 1, 2, 2, 2],
            "development_fold": [0, 0, 0, 1, 1, 1],
            "editing_index": [0.0, 10.0, 20.0, 0.0, 0.5, 5.0],
        }
    )
    features = pd.DataFrame(
        {
            "context": ["x", "x"],
            "pair_id": [1, 2],
            "geometry__span": [10.0, 20.0],
            "intrinsic__gc": [0.4, 0.5],
        }
    )
    table, intrinsic, full = prepare_pair_table(raw, features, "x")
    assert np.isclose(table.set_index("pair_id").loc[1, "edited_fraction"], 0.5)
    assert table.set_index("pair_id").loc[2, "edited_fraction"] == 0
    assert intrinsic == full == ["geometry__span", "intrinsic__gc"]


def test_pair_engagement_cross_fitting_emits_every_pair_once():
    table = pd.DataFrame(
        {
            "pair_id": np.arange(25),
            "development_fold": np.arange(25) % 5,
            "edited_fraction": np.linspace(0, 1, 25),
            "feature": np.linspace(0, 1, 25),
        }
    )
    predictions, selections = fit_cross_fitted(
        table,
        ["feature"],
        "ridge",
        "ridge",
        (0.1, 1.0),
        seed=1,
    )
    assert predictions["pair_id"].nunique() == len(table)
    assert len(predictions) == len(table)
    assert len(selections) == 5
