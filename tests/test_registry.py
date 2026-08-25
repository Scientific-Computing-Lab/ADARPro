from pathlib import Path

from adarpro.registry import EXECUTION_GIT_SHA, register


def test_registry_is_append_only_and_idempotent(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("adarpro.registry.git_sha", lambda repository: "abc")
    row = {
        "run_id": "x",
        "completed_utc": "now",
        "experiment_type": "test",
        "context": "none",
        "input_hashes": {},
        "fold_definition": "none",
        "seed": 1,
        "features": [],
        "hyperparameters": {},
        "artifact": "a",
        "metrics": {},
        "canonical_test_access": False,
    }
    register(tmp_path, row)
    register(tmp_path, row)
    assert len((tmp_path / "experiments/experiments.csv").read_text().splitlines()) == 2


def test_execution_revision_is_captured_at_import():
    assert EXECUTION_GIT_SHA is not None
    assert len(EXECUTION_GIT_SHA) == 40
