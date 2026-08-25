import numpy as np

from adarpro.rbp.rbpnet_predict import entropy, window_summary


def test_rbpnet_profile_summaries_are_relative_signal_features():
    profile = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    result = window_summary(profile, 2, "target_focal")
    assert result["target_focal_value"] == 0.4
    assert np.isclose(result["target_focal_mass_w5"], 1.0)
    assert np.isclose(entropy(profile), -np.sum(profile * np.log(profile)))
