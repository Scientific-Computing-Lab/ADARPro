#!/usr/bin/env python3
"""Trace curated GPT-image components into editable SVG paths.

The PNG files remain provenance masters for the generated candidates.  Only the
SVG derivatives are placed in manuscript figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import vtracer

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "paper/figure_assets/selected"
MANIFEST = ROOT / "paper/figure_assets/selection_manifest.json"


def vectorize(source: Path, target: Path) -> None:
    """Convert a transparent generated asset to compact, smooth SVG paths."""
    vtracer.convert_image_to_svg_py(
        str(source),
        str(target),
        colormode="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=10,
        color_precision=5,
        layer_difference=28,
        corner_threshold=60,
        length_threshold=5.0,
        max_iterations=8,
        splice_threshold=45,
        path_precision=2,
    )
    markup = target.read_text(encoding="utf-8")
    if "<image" in markup or "data:image" in markup:
        raise RuntimeError(f"Raster content remained in {target}")


def update_manifest() -> None:
    if not MANIFEST.exists():
        return
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for record in payload.get("selections", []):
        family = str(record["family"])
        vector_path = ASSET_DIR / f"{family}.svg"
        if vector_path.exists():
            record.pop("placed_asset", None)
            record.pop("placed_format", None)
            record["vector_asset"] = vector_path.relative_to(ROOT).as_posix()
            record["vector_format"] = "SVG paths (no embedded raster image)"
            record["vector_sha256"] = hashlib.sha256(
                vector_path.read_bytes()
            ).hexdigest()
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources = sorted(args.asset_dir.glob("*.png"))
    if not sources:
        raise SystemExit(f"No curated PNG provenance masters found in {args.asset_dir}")
    for source in sources:
        target = source.with_suffix(".svg")
        if args.overwrite or not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
            vectorize(source, target)
            print(f"vectorized {source.name} -> {target.name}")
    update_manifest()


if __name__ == "__main__":
    main()
