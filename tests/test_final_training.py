import pandas as pd
import pytest

from adarpro.models.final_training import (
    assign_raw_source_split,
    truncate_history_after_checkpoint,
)


def test_raw_rows_inherit_published_pair_split():
    canonical = pd.DataFrame(
        {"pair_id": [1, 1, 2], "source_split": ["train", "train", "valid"]}
    )
    raw = pd.DataFrame({"pair_id": [2, 1, 1], "value": [3, 4, 5]})
    result = assign_raw_source_split(raw, canonical)
    assert result["source_split"].tolist() == ["valid", "train", "train"]


def test_raw_split_mapping_rejects_sealed_pair():
    canonical = pd.DataFrame({"pair_id": [1], "source_split": ["test"]})
    raw = pd.DataFrame({"pair_id": [1]})
    with pytest.raises(RuntimeError, match="sealed pair"):
        assign_raw_source_split(raw, canonical)


def test_resume_truncates_uncheckpointed_history(tmp_path):
    path = tmp_path / "history.csv"
    pd.DataFrame({"epoch": [1, 2, 3], "loss": [0.3, 0.2, 0.1]}).to_csv(
        path, index=False
    )
    truncate_history_after_checkpoint(path, 2)
    observed = pd.read_csv(path)
    assert observed["epoch"].tolist() == [1, 2]


def test_resume_rejects_already_duplicated_history(tmp_path):
    path = tmp_path / "history.csv"
    pd.DataFrame({"epoch": [1, 2, 2], "loss": [0.3, 0.2, 0.1]}).to_csv(
        path, index=False
    )
    with pytest.raises(RuntimeError, match="duplicate epochs"):
        truncate_history_after_checkpoint(path, 2)
