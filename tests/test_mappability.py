import numpy as np

from adarpro.mapping.mappability import summarize


def test_mappability_summary_handles_missing_values():
    result = summarize(np.array([1.0, 0.5, np.nan, 1.0]))
    assert result["mean"] == 2.5 / 3
    assert result["minimum"] == 0.5
    assert result["unique_fraction"] == 2 / 3
    assert result["missing_fraction"] == 0.25
