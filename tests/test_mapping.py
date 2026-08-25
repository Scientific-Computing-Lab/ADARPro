import pandas as pd

from adarpro.mapping.reconstruct import expected_genomic_zero_based, reverse_complement


def test_reverse_complement() -> None:
    assert reverse_complement("ACGTN") == "NACGT"


def test_coordinate_transform_plus_and_minus() -> None:
    common = {
        "start1": 100,
        "end1": 109,
        "start2": 200,
        "end2": 219,
        "example_id": "example",
    }
    assert expected_genomic_zero_based(pd.Series({**common, "strand": "+", "local_position": 4})) == 104
    assert expected_genomic_zero_based(pd.Series({**common, "strand": "+", "local_position": 23})) == 203
    assert expected_genomic_zero_based(pd.Series({**common, "strand": "-", "local_position": 4})) == 215
    assert expected_genomic_zero_based(pd.Series({**common, "strand": "-", "local_position": 33})) == 106
