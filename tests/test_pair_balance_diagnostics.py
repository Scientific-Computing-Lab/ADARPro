import numpy as np
import pandas as pd

from adarpro.evaluation.pair_balance_diagnostics import binary_log_loss


def test_binary_log_loss_matches_expected_values():
    labels = pd.Series([0, 1])
    probabilities = pd.Series([0.25, 0.75])
    observed = binary_log_loss(labels, probabilities)
    assert np.allclose(observed, [-np.log(0.75), -np.log(0.75)])


def test_binary_log_loss_clips_endpoint_probabilities():
    observed = binary_log_loss(pd.Series([0, 1]), pd.Series([1.0, 0.0]))
    assert np.isfinite(observed).all()
