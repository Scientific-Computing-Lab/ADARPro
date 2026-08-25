from pathlib import Path

import pytest

from adarpro.quarantine import (
    QuarantineViolation,
    assert_development_path,
    assert_development_split,
)


@pytest.mark.parametrize("split", ["train", "valid"])
def test_development_splits_allowed(split: str) -> None:
    assert_development_split(split)


@pytest.mark.parametrize("split", ["test", "TEST", "validation", "all"])
def test_non_development_splits_rejected(split: str) -> None:
    with pytest.raises(QuarantineViolation):
        assert_development_split(split)


def test_test_filename_rejected() -> None:
    with pytest.raises(QuarantineViolation):
        assert_development_path(Path("data/human/Liver/test.jsonl"))


def test_development_filename_allowed() -> None:
    assert_development_path(Path("data/human/Liver/valid.jsonl"))
