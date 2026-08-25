import pandas as pd

from adarpro.evaluation.matched_recall_curve import (
    evaluate_at_reference_recall,
    pair_bootstrap,
)


def prediction_table():
    return pd.DataFrame(
        {
            "pair_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "label": [1, 0, 1, 0, 1, 0, 1, 0],
            "probability": [0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.1, 0.6],
            "baseline_oof_probability": [0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.4, 0.3],
            "baseline_oof_prediction": [1, 1, 1, 1, 1, 1, 0, 0],
        }
    )


def test_candidate_is_evaluated_at_at_least_reference_recall():
    threshold, candidate, baseline = evaluate_at_reference_recall(prediction_table())
    assert 0 <= threshold <= 1
    assert candidate["recall"] >= baseline["recall"]
    assert candidate["specificity"] > baseline["specificity"]


def test_pair_bootstrap_is_clustered_and_reproducible():
    first = pair_bootstrap(prediction_table(), replicates=10, seed=3)
    second = pair_bootstrap(prediction_table(), replicates=10, seed=3)
    pd.testing.assert_frame_equal(first, second)
    assert first["replicate"].nunique() == 10
