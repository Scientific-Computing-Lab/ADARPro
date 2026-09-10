import pandas as pd

from adarpro.evaluation.within_pair_discrimination import (
    bootstrap_summary,
    per_pair_table,
    within_pair_auc,
)


def test_within_pair_auc_counts_ties_as_half():
    labels = pd.Series([0, 0, 1, 1])
    scores = pd.Series([0.1, 0.5, 0.5, 0.9])
    assert within_pair_auc(labels, scores) == 0.875


def test_per_pair_table_excludes_single_class_pairs_and_weights_pairs_equally():
    table = pd.DataFrame(
        {
            "context": ["X"] * 7,
            "pair_id": [1, 1, 2, 2, 2, 3, 3],
            "development_fold": [0, 0, 1, 1, 1, 2, 2],
            "label": [0, 1, 0, 0, 1, 0, 0],
            "development_reference": [0.2, 0.8, 0.1, 0.2, 0.9, 0.2, 0.1],
            "pair_first": [0.3, 0.7, 0.1, 0.3, 0.8, 0.2, 0.1],
        }
    )
    result = per_pair_table(table, ("development_reference", "pair_first"))
    assert set(result["pair_id"]) == {1, 2}
    assert len(result) == 4
    summary, contrast = bootstrap_summary(
        result, (("pair_first", "development_reference"),), 50, 7
    )
    assert set(summary["mixed_pairs"]) == {2}
    assert contrast.loc[0, "mixed_pairs"] == 2
