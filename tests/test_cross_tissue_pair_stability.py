import pandas as pd

from adarpro.evaluation.cross_tissue_pair_stability import (
    correlations,
    pair_summaries,
)


def test_pair_summaries_use_identical_sites_and_filter_small_pairs():
    table = pd.DataFrame(
        {
            "site_id": ["a", "b", "c", "d"],
            "pair_id": [1, 1, 2, 2],
            "Artery": [0.0, 20.0, 5.0, 7.0],
            "Brain": [1.0, 30.0, 6.0, None],
        }
    )
    result = pair_summaries(table, "Artery", "Brain", min_sites=2)
    assert result["pair_id"].tolist() == [1]
    assert result.loc[0, "matched_sites"] == 2
    assert result.loc[0, "first_mean_editing"] == 10.0
    assert result.loc[0, "second_high_fraction"] == 0.5


def test_correlations_report_rank_stability():
    table = pd.DataFrame(
        {
            "first_mean_editing": [1.0, 2.0, 3.0],
            "second_mean_editing": [2.0, 4.0, 6.0],
            "first_high_fraction": [0.0, 0.5, 1.0],
            "second_high_fraction": [0.0, 0.5, 1.0],
        }
    )
    result = correlations(table)
    assert result["mean_editing_spearman"] == 1.0
    assert result["high_fraction_spearman"] == 1.0
