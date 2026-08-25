"""Capture machine-readable upstream, environment, and input provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args: list[str], *, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture(repository: Path) -> dict:
    repository = repository.resolve()
    patterns = (
        "data/human/**/*.jsonl",
        "data/human/**/*.metadata.csv",
        "data/raw/alu_pairs/Pair_Alu_withStrand.bed",
        "data_construction/split/global_pair_split.json",
        "data_construction/human_alu/published_site_selection.tsv.gz",
        "checkpoints/baseline_*/best.pth",
        "checkpoints/bioaware_*/best.pth",
    )
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in repository.glob(pattern) if path.is_file())
    nvidia = command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        cwd=repository,
    )
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "repository_url": command(["git", "remote", "get-url", "upstream"], cwd=repository),
        "git_revision": command(["git", "rev-parse", "HEAD"], cwd=repository),
        "git_branch": command(["git", "branch", "--show-current"], cwd=repository),
        "checkout_date": "2026-08-21",
        "python": sys.version,
        "platform": platform.platform(),
        "kernel": command(["uname", "-a"], cwd=repository),
        "conda": command(["conda", "--version"], cwd=repository),
        "cuda_driver_gpu": nvidia,
        "canonical_inputs": [
            {
                "path": str(path.relative_to(repository)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(files)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = capture(args.repository)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
