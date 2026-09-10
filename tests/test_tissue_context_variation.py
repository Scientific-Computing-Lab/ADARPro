import pandas as pd

from adarpro.evaluation.tissue_context_variation import add_variation, switch_matrix


def example_table():
    return pd.DataFrame(
        {
            "site_id": ["a", "b"],
            "pair_id": [1, 2],
            "observed_tissues": [4, 2],
            "switch_category": ["strict_edited_nonedited_switch", "other"],
            "Artery": [20.0, 3.0],
            "Brain": [0.0, 5.0],
            "Liver": [10.0, None],
            "MuscleSkeletal": [2.0, None],
            "Artery_state": ["edited", "intermediate"],
            "Brain_state": ["nonedited", "intermediate"],
            "Liver_state": ["intermediate", None],
            "MuscleSkeletal_state": ["intermediate", None],
        }
    )


def test_add_variation_uses_only_observed_tissues():
    result = add_variation(example_table())
    assert result.loc[0, "editing_index_range"] == 20.0
    assert result.loc[1, "editing_index_range"] == 2.0
    assert bool(result.loc[0, "strict_switch"])


def test_switch_matrix_reports_direction_and_denominator():
    result = switch_matrix(example_table())
    row = result[
        result["edited_tissue"].eq("Artery")
        & result["nonedited_tissue"].eq("Brain")
    ].iloc[0]
    assert row["shared_sites"] == 2
    assert row["strict_switch_sites"] == 1
    assert row["strict_switch_per_10000_shared"] == 5000
