import numpy as np

from adarpro.features.substrate_profile import summarize_profile


def test_summarize_profile_excludes_focal_from_other_distribution():
    values = summarize_profile(np.array([0.1, 0.8, 0.3]), 1, 0.5)
    assert np.isclose(values["substrate_profile__all_a_mean"], 0.4)
    assert np.isclose(values["substrate_profile__other_a_mean"], 0.2)
    assert np.isclose(values["substrate_profile__other_a_max"], 0.3)
    assert values["substrate_profile__other_a_fraction_above_threshold"] == 0
    assert values["substrate_profile__focal_percentile"] > 0.8
