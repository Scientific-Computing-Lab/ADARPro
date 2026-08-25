"""Auditable quarantine for the canonical held-out AdarEdit test partition."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
CONTEXTS = ("Artery", "Brain", "Liver", "MuscleSkeletal", "Combined")
DEVELOPMENT_SPLITS = frozenset({"train", "valid"})


class QuarantineViolation(RuntimeError):
    """Raised when code attempts unapproved held-out test access."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _artifact_inventory(repository: Path) -> list[dict]:
    artifacts = []
    for context in CONTEXTS:
        for suffix in ("test.jsonl", "test.metadata.csv"):
            path = repository / "data" / "human" / context / suffix
            if not path.is_file():
                raise FileNotFoundError(path)
            artifacts.append(
                {
                    "path": str(path.relative_to(repository)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return artifacts


def initialize(repository: Path, policy_path: Path, *, force: bool = False) -> dict:
    repository = repository.resolve()
    if policy_path.exists() and not force:
        raise FileExistsError(f"quarantine already initialized: {policy_path}")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "repository": str(repository),
        "upstream_revision": revision,
        "state": "development_sealed",
        "development_splits": sorted(DEVELOPMENT_SPLITS),
        "test_artifacts": _artifact_inventory(repository),
        "published_reproduction": {"runs": 0, "last_result_sha256": None},
        "final_locked_test": {
            "freeze_manifest": None,
            "authorization_sha256": None,
            "evaluations_consumed": 0,
            "consumed_utc": None,
        },
        "audit_chain_head": None,
    }
    _write_json_atomic(policy_path, payload)
    return payload


def load_policy(policy_path: Path) -> dict:
    payload = json.loads(policy_path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise QuarantineViolation("unsupported quarantine policy schema")
    return payload


def assert_development_split(split: str) -> None:
    if split not in DEVELOPMENT_SPLITS:
        raise QuarantineViolation(
            f"split {split!r} is sealed; development permits only "
            f"{sorted(DEVELOPMENT_SPLITS)}"
        )


def assert_development_path(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if "test" in lowered or path.name.lower().startswith("test."):
        raise QuarantineViolation(f"held-out path rejected in development: {path}")


def development_file(repository: Path, context: str, split: str, suffix: str) -> Path:
    if context not in CONTEXTS:
        raise ValueError(f"unknown context: {context}")
    assert_development_split(split)
    path = repository / "data" / "human" / context / f"{split}.{suffix}"
    assert_development_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _audit(policy_path: Path, event: dict) -> None:
    policy = load_policy(policy_path)
    record = {
        "timestamp_utc": utc_now(),
        "previous_sha256": policy.get("audit_chain_head"),
        **event,
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record_sha = hashlib.sha256(canonical).hexdigest()
    audit_path = policy_path.with_name("test_access_audit.jsonl")
    with audit_path.open("a") as handle:
        handle.write(json.dumps({**record, "record_sha256": record_sha}) + "\n")
    policy["audit_chain_head"] = record_sha
    _write_json_atomic(policy_path, policy)


@contextlib.contextmanager
def published_reproduction_access(
    policy_path: Path, *, command: str
) -> Iterator[dict]:
    """Permit aggregate-only scoring of immutable supplied checkpoints.

    The caller must not persist per-site predictions. Every attempt is logged;
    this mode never authorizes a new or modified model.
    """

    policy = load_policy(policy_path)
    if policy["state"] not in {"development_sealed", "final_frozen"}:
        raise QuarantineViolation(f"invalid state: {policy['state']}")
    run_number = int(policy["published_reproduction"]["runs"]) + 1
    policy["published_reproduction"]["runs"] = run_number
    _write_json_atomic(policy_path, policy)
    _audit(
        policy_path,
        {
            "event": "published_reproduction_started",
            "run": run_number,
            "command": command,
            "restriction": "supplied checkpoints; aggregate output only",
        },
    )
    result = {"run": run_number, "result_sha256": None}
    try:
        yield result
    except Exception as exc:
        _audit(
            policy_path,
            {
                "event": "published_reproduction_failed",
                "run": run_number,
                "error_type": type(exc).__name__,
            },
        )
        raise
    else:
        policy = load_policy(policy_path)
        policy["published_reproduction"]["last_result_sha256"] = result[
            "result_sha256"
        ]
        _write_json_atomic(policy_path, policy)
        _audit(
            policy_path,
            {
                "event": "published_reproduction_completed",
                "run": run_number,
                "aggregate_result_sha256": result["result_sha256"],
            },
        )


def freeze_final(policy_path: Path, manifest: Path) -> str:
    policy = load_policy(policy_path)
    if policy["final_locked_test"]["evaluations_consumed"]:
        raise QuarantineViolation("final test authorization has already been consumed")
    manifest_hash = sha256_file(manifest)
    policy["state"] = "final_frozen"
    policy["final_locked_test"]["freeze_manifest"] = {
        "path": str(manifest.resolve()),
        "sha256": manifest_hash,
        "frozen_utc": utc_now(),
    }
    token = hashlib.sha256(os.urandom(32)).hexdigest()
    policy["final_locked_test"]["authorization_sha256"] = hashlib.sha256(
        token.encode()
    ).hexdigest()
    _write_json_atomic(policy_path, policy)
    _audit(
        policy_path,
        {"event": "final_pipeline_frozen", "manifest_sha256": manifest_hash},
    )
    return token


def consume_final_authorization(policy_path: Path, token: str) -> dict:
    policy = load_policy(policy_path)
    final = policy["final_locked_test"]
    if policy["state"] != "final_frozen" or final["freeze_manifest"] is None:
        raise QuarantineViolation("the complete pipeline has not been frozen")
    if final["evaluations_consumed"] != 0:
        raise QuarantineViolation("the one-use final test authorization is spent")
    supplied = hashlib.sha256(token.encode()).hexdigest()
    if supplied != final["authorization_sha256"]:
        raise QuarantineViolation("invalid final test authorization")
    manifest = Path(final["freeze_manifest"]["path"])
    if sha256_file(manifest) != final["freeze_manifest"]["sha256"]:
        raise QuarantineViolation("frozen manifest changed after authorization")
    final["evaluations_consumed"] = 1
    final["consumed_utc"] = utc_now()
    final["authorization_sha256"] = None
    policy["state"] = "final_consumed"
    _write_json_atomic(policy_path, policy)
    _audit(
        policy_path,
        {
            "event": "final_test_authorization_consumed",
            "manifest_sha256": final["freeze_manifest"]["sha256"],
        },
    )
    return policy


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("provenance/test_quarantine.json")
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("init")
    subparsers.add_parser("status")
    freeze = subparsers.add_parser("freeze-final")
    freeze.add_argument("manifest", type=Path)
    args = parser.parse_args()
    policy_path = (
        args.policy
        if args.policy.is_absolute()
        else args.repository.resolve() / args.policy
    )
    if args.action == "init":
        payload = initialize(args.repository, policy_path)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.action == "status":
        print(json.dumps(load_policy(policy_path), indent=2, sort_keys=True))
    elif args.action == "freeze-final":
        token = freeze_final(policy_path, args.manifest.resolve())
        print("Final pipeline frozen. Store this single-use token securely:")
        print(token)


if __name__ == "__main__":
    _cli()
