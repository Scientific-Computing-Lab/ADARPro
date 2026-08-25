import numpy as np
import pandas as pd

from adarpro.evaluation.target_local_interpretation import (
    add_strata,
    loss_contrasts,
    transition_row,
)


def test_transition_row_counts_rescue_and_damage():
    result = transition_row(
        np.array([0, 0, 1, 1]),
        np.array([1, 0, 1, 0]),
        np.array([0, 1, 0, 1]),
        "pair",
        "local",
    )
    assert result["fp_rescued"] == 1
    assert result["new_fp"] == 1
    assert result["tp_damaged"] == 1
    assert result["fn_recovered"] == 1


def test_strata_and_loss_contrasts():
    table = pd.DataFrame(
        {
            "label": [0, 1],
            "pair_probability": [0.8, 0.6],
            "local_probability": [0.2, 0.8],
            "target_geometry__target_paired": [0.0, 1.0],
            "target_geometry__stem_extent_total": [0.0, 8.0],
            "target_geometry__w10_helix_intact_fraction": [0.0, 0.8],
            "target_geometry__partner_base_A": [0.0, 0.0],
            "target_geometry__partner_base_C": [0.0, 1.0],
            "target_geometry__partner_base_G": [0.0, 0.0],
            "target_geometry__partner_base_T": [0.0, 0.0],
            "target_geometry__partner_base_N": [1.0, 0.0],
        }
    )
    result = loss_contrasts(add_strata(table), ("local",))
    assert result["target_paired"].tolist() == ["unpaired", "paired"]
    assert result["partner_base"].tolist() == ["unpaired", "C"]
    assert result["stem_extent"].tolist() == ["0", "6-10"]
    assert (result["local_log_loss"] < result["pair_log_loss"]).all()
