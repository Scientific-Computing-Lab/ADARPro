import pandas as pd
import pytest

from adarpro.evaluation.long_oof_pairwise import run


def test_long_oof_pairwise_rejects_same_model(tmp_path):
    path = tmp_path / "predictions.parquet"
    pd.DataFrame({"model": ["a"]}).to_parquet(path)
    with pytest.raises(ValueError, match="must differ"):
        run(tmp_path, path, "a", "a", tmp_path / "out", 1, 1)
