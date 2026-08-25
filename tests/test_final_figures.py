import json

import pandas as pd
import pytest

from adarpro.evaluation.final_figures import (
    error_transitions,
    evaluate_at_baseline_recall,
    matched_recall_bootstrap,
    run,
)
from adarpro.quarantine import QuarantineViolation


def test_error_transitions_reports_rescue_and_damage_rates():
    table = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "baseline_prediction": [1, 0, 1, 0],
            "prediction": [0, 1, 0, 1],
        }
    )
    result = error_transitions(table, "Liver")
    assert result["fp_rescued"] == 1
    assert result["new_fp"] == 1
    assert result["tp_damaged"] == 1
    assert result["fn_recovered"] == 1
    assert result["fp_rescue_rate"] == 1.0
    assert result["tp_damage_rate"] == 1.0


def test_final_figures_reject_unconsumed_quarantine(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema_version": 1, "state": "development_sealed"}))
    with pytest.raises(QuarantineViolation, match="consumed final evaluation"):
        run(tmp_path, policy, tmp_path / "final", tmp_path / "figures")


def test_matched_recall_is_descriptive_and_pair_bootstrapped():
    table = pd.DataFrame(
        {
            "pair_id": [1, 1, 2, 2, 3, 3],
            "label": [0, 1, 0, 1, 0, 1],
            "baseline_probability": [0.8, 0.9, 0.2, 0.8, 0.1, 0.2],
            "baseline_prediction": [1, 1, 0, 1, 0, 0],
            "probability": [0.3, 0.9, 0.2, 0.8, 0.1, 0.7],
        }
    )
    threshold, candidate, baseline = evaluate_at_baseline_recall(table)
    assert 0 <= threshold <= 1
    assert candidate["recall"] >= baseline["recall"]
    assert candidate["fp"] < baseline["fp"]
    first = matched_recall_bootstrap(table, 5, 9)
    second = matched_recall_bootstrap(table, 5, 9)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["metric"]) == {
        "f1",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "mcc",
        "false_positives",
    }
