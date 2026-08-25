import pandas as pd

from adarpro.models.continuous_differences import weighted_metrics


def test_continuous_metrics_are_zero_error_for_exact_prediction():
    frame = pd.DataFrame(
        {
            "editing_difference": [1.0, -2.0, 3.0],
            "predicted_difference": [1.0, -2.0, 3.0],
            "site_weight": [1.0, 0.5, 1.0],
        }
    )
    result = weighted_metrics(frame)
    assert result["rmse"] == 0
    assert result["mae"] == 0
    assert result["r2"] == 1
