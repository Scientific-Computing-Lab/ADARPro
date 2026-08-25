import pandas as pd
import pytest

from adarpro.evaluation.fixed_ensemble_natural_prevalence import IDENTITY


def test_natural_ensemble_identity_includes_fold_and_site_keys():
    assert IDENTITY == [
        "example_id",
        "context",
        "pair_id",
        "pair_local_key",
        "development_fold",
    ]
    frame = pd.DataFrame({column: [1] for column in IDENTITY})
    with pytest.raises(KeyError):
        frame[IDENTITY + ["label"]]
