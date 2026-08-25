import numpy as np
import pandas as pd

from adarpro.models.residual import (
    best_f1_threshold,
    matched_recall_threshold,
    rescue_metrics,
)


def test_threshold_selection_and_rescue_definitions():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.7, 0.9])
    threshold, score = best_f1_threshold(labels, probabilities)
    assert 0.1 <= threshold <= 0.9
    assert score > 0.7
    matched = matched_recall_threshold(labels, probabilities, 1.0)
    assert matched <= 0.7
    frame = pd.DataFrame(
        {
            "label": labels,
            "baseline_oof_prediction": [0, 1, 1, 1],
            "prediction": [0, 0, 1, 0],
        }
    )
    rates = rescue_metrics(frame, "prediction")
    assert rates["fp_rescue_rate"] == 1.0
    assert rates["tp_damage_rate"] == 0.5
