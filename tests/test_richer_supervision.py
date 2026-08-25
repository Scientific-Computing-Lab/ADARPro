import pandas as pd
import pytest

from adarpro.evaluation.richer_supervision import validate_alignment


def candidate(probability_shift=0.0):
    return pd.DataFrame(
        {
            "example_id": ["a", "b"],
            "context": ["MuscleSkeletal", "MuscleSkeletal"],
            "pair_id": [1, 2],
            "pair_local_key": ["1:2", "2:3"],
            "label": [0, 1],
            "development_fold": [0, 1],
            "probability": [0.2 + probability_shift, 0.8],
            "prediction": [0, 1],
            "baseline_oof_probability": [0.3, 0.7],
            "baseline_oof_prediction": [0, 1],
        }
    )


def test_alignment_allows_candidate_probability_differences():
    validate_alignment({"one": candidate(), "two": candidate(0.1)})


def test_alignment_rejects_different_development_reference():
    second = candidate()
    second.loc[0, "baseline_oof_probability"] = 0.4
    with pytest.raises(RuntimeError, match="development reference mismatch"):
        validate_alignment({"one": candidate(), "two": second})
