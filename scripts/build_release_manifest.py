#!/usr/bin/env python3
"""Write or verify a SHA-256 manifest for a curated ADARPro checkout."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "build"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def records(root: Path, output: Path) -> list[str]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.resolve() == output.resolve():
            continue
        rows.append(f"{digest(path)}  {relative.as_posix()}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("provenance/release_manifest.sha256")
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.repository.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    current = records(root, output)
    if args.verify:
        expected = output.read_text().splitlines()
        if current != expected:
            raise SystemExit("release manifest mismatch")
        print(f"verified {len(current)} release files")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(current) + "\n")
    print(f"recorded {len(current)} release files")


if __name__ == "__main__":
    main()
