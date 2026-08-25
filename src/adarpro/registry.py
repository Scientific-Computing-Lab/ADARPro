"""Append-only experiment registry with conflict detection."""

from __future__ import annotations

import csv
import fcntl
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

FIELDS = (
    "run_id",
    "completed_utc",
    "git_sha",
    "experiment_type",
    "context",
    "input_hashes",
    "fold_definition",
    "seed",
    "features",
    "hyperparameters",
    "artifact",
    "metrics",
    "canonical_test_access",
)


def _capture_execution_git_sha() -> str | None:
    """Freeze the source revision when the Python process imports ADARpro."""

    repository = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


EXECUTION_GIT_SHA = _capture_execution_git_sha()


def git_sha(repository: Path) -> str:
    if EXECUTION_GIT_SHA is not None:
        return EXECUTION_GIT_SHA
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()


def json_value(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def register(repository: Path, row: dict) -> None:
    path = repository / "experiments" / "experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {field: "" for field in FIELDS}
    normalized.update(row)
    if not normalized["completed_utc"]:
        normalized["completed_utc"] = datetime.now(timezone.utc).isoformat()
    if not normalized["git_sha"]:
        normalized["git_sha"] = git_sha(repository)
    for key in ("input_hashes", "features", "hyperparameters", "metrics"):
        if not isinstance(normalized[key], str):
            normalized[key] = json_value(normalized[key])
    normalized["canonical_test_access"] = str(
        bool(normalized["canonical_test_access"])
    ).lower()
    normalized = {key: str(value) for key, value in normalized.items()}
    with path.open("a+", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        reader = csv.DictReader(handle)
        if reader.fieldnames is not None and tuple(reader.fieldnames) != FIELDS:
            raise RuntimeError(
                f"experiment registry schema mismatch: {reader.fieldnames}"
            )
        existing = list(reader)
        conflicts = [item for item in existing if item["run_id"] == normalized["run_id"]]
        if conflicts:
            substantive_fields = [field for field in FIELDS if field != "completed_utc"]
            if any(
                conflicts[0][field] != normalized[field]
                for field in substantive_fields
            ):
                raise RuntimeError(
                    f"experiment run_id already exists with different content: {normalized['run_id']}"
                )
            return
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not existing:
            writer.writeheader()
        writer.writerow(normalized)
        handle.flush()
