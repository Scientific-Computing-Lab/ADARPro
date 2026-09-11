import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_authoritative_results", ROOT / "scripts/verify_authoritative_results.py"
)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def test_retained_registry_accepts_documented_development_diagnostics():
    VERIFIER.verify_registry()


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "renamed", "type", "test_access", "second_test"]
)
def test_registry_rejects_identity_and_evidence_changes(tmp_path, monkeypatch, mutation):
    with (ROOT / "experiments/experiments.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    run_id = next(iter(VERIFIER.POST_HOC_DEVELOPMENT_RUNS))
    target = next(row for row in rows if row["run_id"] == run_id)
    if mutation == "missing":
        rows.remove(target)
    elif mutation == "duplicate":
        rows[-1] = target.copy()
    elif mutation == "renamed":
        target["run_id"] = "unregistered_replacement"
    elif mutation == "type":
        target["experiment_type"] = "confirmatory"
    elif mutation == "test_access":
        target["canonical_test_access"] = "true"
    else:
        rows[0]["experiment_type"] = "single_consumed_canonical_test_evaluation"
        rows[0]["canonical_test_access"] = "true"
    path = tmp_path / "experiments/experiments.csv"
    path.parent.mkdir()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(VERIFIER, "ROOT", tmp_path)
    with pytest.raises(AssertionError):
        VERIFIER.verify_registry()
