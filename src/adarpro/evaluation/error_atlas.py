"""Build an intrinsic-feature atlas from leakage-free development predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def local_fraction(sequence: str, center: int, radius: int, bases: set[str]) -> float:
    region = sequence[max(0, center - radius) : center + radius + 1]
    observed = [base for base in region if base in {"A", "C", "G", "T"}]
    return sum(base in bases for base in observed) / len(observed) if observed else np.nan


def paired_partner(structure: str, position: int) -> int:
    stack, partner = [], {}
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            left = stack.pop()
            partner[left], partner[index] = index, left
    return partner.get(position, -1)


def add_intrinsic_descriptors(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in table.itertuples(index=False):
        position = int(row.local_position)
        partner = paired_partner(row.structure, position)
        rows.append(
            {
                "example_id": row.example_id,
                "sequence_length": len(row.sequence),
                "focal_is_paired": partner >= 0,
                "partner_local_position": partner,
                "pair_span": abs(partner - position) if partner >= 0 else -1,
                "local_gc_21": local_fraction(row.sequence, position, 10, {"G", "C"}),
                "local_gc_51": local_fraction(row.sequence, position, 25, {"G", "C"}),
                "local_purine_21": local_fraction(row.sequence, position, 10, {"A", "G"}),
                "distance_to_sequence_end": min(position, len(row.sequence) - position - 1),
            }
        )
    return table.merge(pd.DataFrame(rows), on="example_id", validate="one_to_one")


def run(site_master_path: Path, output_dir: Path) -> dict:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("error atlas contains a sealed split")
    available = master[master["baseline_oof_probability"].notna()].copy()
    if available.empty:
        raise RuntimeError("no baseline OOF predictions are available")
    atlas = add_intrinsic_descriptors(available)
    atlas["baseline_confidence"] = np.maximum(
        atlas["baseline_oof_probability"], 1 - atlas["baseline_oof_probability"]
    )
    atlas["high_confidence_false_positive"] = (
        (atlas["baseline_oof_status"] == "FP")
        & (atlas["baseline_oof_probability"] >= 0.8)
    )
    status = (
        atlas.groupby(["context", "baseline_oof_status"], as_index=False)
        .agg(
            sites=("example_id", "size"),
            pairs=("pair_id", "nunique"),
            mean_probability=("baseline_oof_probability", "mean"),
            median_probability=("baseline_oof_probability", "median"),
            paired_fraction=("focal_is_paired", "mean"),
            mean_local_gc_21=("local_gc_21", "mean"),
        )
    )
    context = atlas.groupby("context", as_index=False).agg(
        sites=("example_id", "size"),
        pairs=("pair_id", "nunique"),
        false_positives=("baseline_oof_status", lambda values: (values == "FP").sum()),
        high_confidence_false_positives=("high_confidence_false_positive", "sum"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atlas.to_parquet(output_dir / "error_atlas.parquet", index=False)
    status.to_csv(output_dir / "status_summary.csv", index=False)
    context.to_csv(output_dir / "context_summary.csv", index=False)
    result = {
        "scope": "available train+valid grouped-OOF predictions only",
        "contexts": sorted(atlas["context"].unique()),
        "sites": len(atlas),
        "pairs": int(atlas["pair_id"].nunique()),
        "false_positives": int((atlas["baseline_oof_status"] == "FP").sum()),
        "high_confidence_false_positives_probability_ge_0.8": int(
            atlas["high_confidence_false_positive"].sum()
        ),
        "canonical_test_access": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/adarpro/error_atlas")
    )
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(
        json.dumps(
            run(resolve(args.site_master), resolve(args.output_dir)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
