import numpy as np
import pandas as pd

from adarpro.rbp.eclip_features import (
    build_indices,
    dot_bracket_partners,
    interval_features,
    local_to_genomic,
    region_features,
)


def test_dot_bracket_partner_mapping():
    assert dot_bracket_partners("((..))") == [5, 4, None, None, 1, 0]


def test_strand_index_local_and_regional_features():
    peaks = pd.DataFrame(
        {
            "chromosome": ["chr1", "chr1", "chr1"],
            "start": [10, 20, 40],
            "end": [15, 25, 45],
            "strand": ["+", "+", "-"],
            "signal": [2.0, 4.0, 100.0],
        }
    )
    index = build_indices(peaks)[("chr1", "+")]
    inside = interval_features(index, 12)
    assert inside["overlap"] == 1
    assert inside["nearest_distance"] == 0
    assert inside["count_w10"] == 2
    assert inside["max_signal_w10"] == 4.0
    outside = interval_features(index, 17)
    assert outside["overlap"] == 0
    assert outside["nearest_distance"] == 3
    region = region_features(index, [(12, 23)])
    assert region["peak_count"] == 2
    assert region["covered_nt"] == 6
    assert np.isclose(region["coverage_fraction"], 6 / 11)


def test_transcript_local_to_genomic_on_both_strands():
    plus = pd.Series(
        {"start1": 100, "end1": 109, "start2": 200, "end2": 214, "strand": "+"}
    )
    assert local_to_genomic(plus, 3) == (103, 1)
    assert local_to_genomic(plus, 22) == (202, 2)
    assert local_to_genomic(plus, 12) is None
    minus = plus.copy()
    minus["strand"] = "-"
    assert local_to_genomic(minus, 3) == (211, 2)
    assert local_to_genomic(minus, 27) == (107, 1)
