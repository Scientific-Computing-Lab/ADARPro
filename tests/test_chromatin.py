import numpy as np
import pandas as pd

from adarpro.features.chromatin import query_peak_metrics


def test_query_peak_metrics_handles_overlap_signal_and_distance():
    peaks = pd.DataFrame(
        {"start": [10, 30], "end": [20, 40], "signal": [2.0, 7.0]}
    )
    overlap, signal, distance = query_peak_metrics(
        peaks, np.array([15, 21, 25, 35]), np.array([16, 22, 36, 36])
    )
    assert overlap.tolist() == [1.0, 0.0, 1.0, 1.0]
    assert signal.tolist() == [2.0, 0.0, 7.0, 7.0]
    assert distance.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_query_peak_metrics_empty_track_is_unknown_distance():
    overlap, signal, distance = query_peak_metrics(
        None, np.array([1]), np.array([2])
    )
    assert overlap.tolist() == [0.0]
    assert signal.tolist() == [0.0]
    assert np.isnan(distance[0])
