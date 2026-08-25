#!/usr/bin/env python3
"""Inventory the preserved research workspace and its curated release copy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter
from pathlib import Path


ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "ADARPRO_PROTOCOL.md",
    "LICENSE",
    "README.md",
    "environment.yml",
    "pyproject.toml",
}
ROOTS = {
    "analyses",
    "checkpoints",
    "code",
    "context",
    "data",
    "data_construction",
    "experiments",
    "figures",
    "manuscript",
    "models",
    "provenance",
    "reports",
    "results",
    "src",
    "tests",
}
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", "smoke"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def iter_artifacts(root: Path):
    for current, directories, files in os.walk(root, followlinks=False):
        relative_dir = Path(current).relative_to(root)
        directories[:] = [
            name
            for name in directories
            if name != ".git" and name not in SKIP_PARTS
        ]
        if relative_dir.parts and relative_dir.parts[0] not in ROOTS:
            directories[:] = []
            continue
        for name in files:
            relative = relative_dir / name
            if not relative.parts:
                continue
            if len(relative.parts) == 1 and name not in ROOT_FILES:
                continue
            if relative.parts[0] not in ROOTS and name not in ROOT_FILES:
                continue
            if any(part in SKIP_PARTS for part in relative.parts):
                continue
            yield relative


def category(path: Path) -> tuple[str, str]:
    value = path.as_posix()
    if value.startswith("results/adarpro/final_locked_test/"):
        return "final_result", "final canonical held-out test"
    if value.startswith("figures/final/"):
        return "final_figure", "post-lock descriptive"
    if value.startswith("models/frozen"):
        return "final_model", "validation-selected frozen artifact"
    if value.startswith("experiments/runs/"):
        return "development_result", "pair-disjoint development or control"
    if value == "experiments/experiments.csv":
        return "experiment_registry", "mixed; explicit row-level scope"
    if value.startswith("reports/"):
        return "campaign_report", "scope stated within report"
    if value.startswith("provenance/"):
        return "provenance", "audit or input identity"
    if value.startswith(("data/human/", "data_construction/split/")):
        return "canonical_data", "published AdarEdit benchmark"
    if value.startswith("data/derived/"):
        return "derived_data", "development-only unless explicitly final"
    if value.startswith("data/external/"):
        return "external_data", "outcome-independent external resource"
    if value.startswith("checkpoints/"):
        return "upstream_checkpoint", "published AdarEdit artifact"
    if value.startswith(("src/", "code/", "data_construction/")):
        return "source_code", "implementation"
    if value.startswith(("analyses/", "manuscript/", "data/species")):
        return "upstream_archive", "published AdarEdit reproducibility package"
    if value.startswith("figures/"):
        return "development_figure", "scope stated by source table"
    return "project_file", "documentation or configuration"


def exclusion_reason(path: Path, source_root: Path, release_root: Path) -> str:
    value = path.as_posix()
    source = source_root / path
    release = release_root / path
    if release.exists() or release.is_symlink():
        return "included"
    if value.startswith("data/external/"):
        return "external download; source and checksum retained"
    if value.startswith("experiments/runs/") and path.suffix == ".pth":
        return "exploratory fold checkpoint; training config and result tables retained"
    if value.startswith("data/derived/") and source.stat().st_size > 50 * 1024**2:
        return "large rebuildable feature matrix; source checksum retained"
    if value.startswith(("analyses/", "manuscript/", "data/species")):
        return "available in pinned upstream AdarEdit repository"
    return "workspace-only or superseded artifact"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("provenance/source_workspace_inventory.csv")
    )
    parser.add_argument(
        "--excluded-output",
        type=Path,
        default=Path("provenance/excluded_artifacts.csv"),
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    release_root = args.release_root.resolve()
    rows = []
    for relative in sorted(iter_artifacts(source_root)):
        source = source_root / relative
        link = source.is_symlink()
        source_hash = (
            hashlib.sha256(os.readlink(source).encode()).hexdigest()
            if link
            else digest(source)
        )
        release = release_root / relative
        included = release.is_file() and not release.is_symlink()
        release_hash = digest(release) if included else ""
        artifact_class, epistemic_status = category(relative)
        reason = exclusion_reason(relative, source_root, release_root)
        if included and release_hash != source_hash:
            reason = "included with portability-only path normalization"
        rows.append(
            {
                "path": relative.as_posix(),
                "artifact_class": artifact_class,
                "epistemic_status": epistemic_status,
                "source_bytes": source.lstat().st_size,
                "source_sha256": source_hash,
                "release_sha256": release_hash,
                "repository_status": "included" if included else "excluded",
                "reason": reason,
                "is_symlink": str(link).lower(),
            }
        )
    counts = Counter(row["source_sha256"] for row in rows)
    first_path = {}
    for row in rows:
        first_path.setdefault(row["source_sha256"], row["path"])
        row["duplicate_count"] = counts[row["source_sha256"]]
        row["duplicate_canonical_path"] = first_path[row["source_sha256"]]
    fields = list(rows[0])
    output = args.output if args.output.is_absolute() else release_root / args.output
    excluded = (
        args.excluded_output
        if args.excluded_output.is_absolute()
        else release_root / args.excluded_output
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, selected in (
        (output, rows),
        (excluded, [row for row in rows if row["repository_status"] == "excluded"]),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    print(f"inventoried {len(rows)} artifacts; excluded {sum(row['repository_status'] == 'excluded' for row in rows)}")


if __name__ == "__main__":
    main()
