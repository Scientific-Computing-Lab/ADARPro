import numpy as np
import pandas as pd

from adarpro.features.genomic_context import (
    alu_age_flags,
    best_repeat,
    interval_distance,
    pair_intrinsic_features,
    parse_attributes,
)


def test_gtf_attribute_parser():
    observed = parse_attributes(
        'gene_id "ENSG1.2"; transcript_id "ENST1.3"; transcript_type "lncRNA";'
    )
    assert observed["gene_id"] == "ENSG1.2"
    assert observed["transcript_id"] == "ENST1.3"


def test_best_repeat_and_age_are_coordinate_based():
    repeats = pd.DataFrame(
        {
            "start": [90, 105],
            "end": [205, 130],
            "repeat_name": ["AluSx", "AluY"],
        },
        index=[4, 5],
    )
    repeat, index, coverage = best_repeat(repeats, 100, 200)
    assert index == 4
    assert repeat["repeat_name"] == "AluSx"
    assert coverage == 1.0
    assert alu_age_flags("AluYb8")["alu_age_young_y"] == 1


def test_interval_distance_and_pair_geometry():
    distances = interval_distance(
        np.array([0, 20, 40]), np.array([10, 30, 50]), 12, 18
    )
    assert distances.tolist() == [2, 2, 22]
    pair = pd.Series(
        {
            "start1": 100,
            "end1": 109,
            "start2": 120,
            "end2": 129,
            "strand": "+",
            "sequence": "A" * 10 + "N" * 10 + "G" * 10,
            "structure": "(" * 10 + "." * 10 + ")" * 10,
        }
    )
    features = pair_intrinsic_features(pair)
    assert features["geometry__genomic_gap"] == 11
    assert features["geometry__arm1_gc"] == 0
    assert features["geometry__arm2_gc"] == 1
    assert features["intrinsic__base_pair_count"] == 10
