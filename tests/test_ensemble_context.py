import pandas as pd

from adarpro.features.ensemble_context import build_features


def test_build_features_validates_and_normalizes_context_fold():
    master = pd.DataFrame(
        {
            "example_id": ["x", "y"],
            "context": ["MuscleSkeletal", "MuscleSkeletal"],
            "pair_id": [1, 1],
            "pair_local_key": ["1:2", "1:3"],
            "bed_row": [4, 4],
            "chr1": ["chr1", "chr1"],
            "start1": [10, 10],
            "end1": [20, 20],
            "start2": [30, 30],
            "end2": [40, 40],
            "strand": ["+", "+"],
        }
    )
    folding = pd.DataFrame(
        {
            "idx": [4],
            "chr": ["chr1"],
            "start1": [10],
            "end1": [20],
            "start2": [30],
            "end2": [40],
            "strand": ["+"],
            "arm1_len": [10],
            "ctx_len": [50],
            "arm1_paired_other": [2],
            "arm1_unpaired": [3],
            "dsrna_frac": [0.5],
            "ctx_mfe_kcal": [-10.0],
            "ensemble_free_energy": [-11.0],
            "delta_delta_g": [1.0],
            "mean_arm1_arm2_prob": [0.2],
        }
    )
    result = build_features(master, folding)
    assert result["ensemble_context__available"].tolist() == [1.0, 1.0]
    assert result["ensemble_context__arm1_other_paired_fraction"].tolist() == [0.2, 0.2]
    assert result["ensemble_context__mfe_per_nt"].tolist() == [-0.2, -0.2]
