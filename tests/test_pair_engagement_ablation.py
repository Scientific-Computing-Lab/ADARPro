import numpy as np
import pandas as pd

from adarpro.models.pair_engagement import score


def test_pair_engagement_score_uses_continuous_and_any_high_targets():
    table = pd.DataFrame(
        {
            "edited_fraction": [0.0, 0.2, 0.8, 1.0],
            "any_edited": [0, 1, 1, 1],
            "prediction": [0.1, 0.3, 0.7, 0.9],
        }
    )
    result = score(table, "prediction")
    assert np.isclose(result["spearman"], 1.0)
    assert np.isclose(result["any_edited_auroc"], 1.0)
