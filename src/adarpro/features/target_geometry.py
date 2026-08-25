"""Target-centered structural-contact features not explicit in pooled AdarEdit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.quarantine import sha256_file
from adarpro.registry import register

WINDOWS = (5, 10, 20, 32)


def pair_map(structure: str) -> np.ndarray:
    partners = np.full(len(structure), -1, dtype=np.int64)
    stack = []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")" and stack:
            partner = stack.pop()
            partners[index] = partner
            partners[partner] = index
    return partners


def run_summary(paired: np.ndarray) -> tuple[float, float, float, float]:
    if not len(paired):
        return (np.nan,) * 4
    transitions = np.sum(paired[1:] != paired[:-1])
    unpaired = ~paired
    padded = np.r_[False, unpaired, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    lengths = ends - starts
    return (
        float(np.mean(paired)),
        float(transitions / max(len(paired) - 1, 1)),
        float(len(lengths) / len(paired)),
        float(lengths.max(initial=0) / len(paired)),
    )


def centered_slice(values: np.ndarray, center: int, radius: int) -> np.ndarray:
    return values[max(0, center - radius) : min(len(values), center + radius + 1)]


def nearest_unpaired_distance(paired: np.ndarray, target: int, direction: int) -> float:
    distance = 0
    position = target
    while 0 <= position < len(paired):
        if not paired[position]:
            return float(distance)
        position += direction
        distance += 1
    return float(len(paired))


def helical_extent(partners: np.ndarray, target: int, direction: int) -> int:
    partner = int(partners[target])
    if partner < 0:
        return 0
    extent = 0
    while True:
        offset = extent + 1
        focal_position = target + direction * offset
        partner_position = partner - direction * offset
        if not (
            0 <= focal_position < len(partners)
            and 0 <= partner_position < len(partners)
            and partners[focal_position] == partner_position
        ):
            return extent
        extent += 1


def contact_features(sequence: str, structure: str, target: int) -> dict[str, float]:
    if len(sequence) != len(structure) or not 0 <= target < len(sequence):
        raise ValueError("invalid target-centered sequence/structure")
    partners = pair_map(structure)
    paired = partners >= 0
    partner = int(partners[target])
    left_extent = helical_extent(partners, target, -1)
    right_extent = helical_extent(partners, target, 1)
    result = {
        "target_geometry__target_paired": float(partner >= 0),
        "target_geometry__stem_extent_left": float(left_extent),
        "target_geometry__stem_extent_right": float(right_extent),
        "target_geometry__stem_extent_total": float(left_extent + 1 + right_extent)
        if partner >= 0
        else 0.0,
        "target_geometry__nearest_unpaired_left": nearest_unpaired_distance(
            paired, target, -1
        ),
        "target_geometry__nearest_unpaired_right": nearest_unpaired_distance(
            paired, target, 1
        ),
        "target_geometry__partner_distance_log1p": float(
            np.log1p(abs(partner - target)) if partner >= 0 else np.nan
        ),
    }
    partner_base = sequence[partner].upper().replace("U", "T") if partner >= 0 else "N"
    for base in "ACGTN":
        result[f"target_geometry__partner_base_{base}"] = float(partner_base == base)
    for radius in WINDOWS:
        focal = centered_slice(paired, target, radius)
        focal_summary = run_summary(focal)
        if partner >= 0:
            opposite = centered_slice(paired, partner, radius)
            opposite_summary = run_summary(opposite)
            valid, intact = 0, 0
            for offset in range(-radius, radius + 1):
                focal_position = target + offset
                partner_position = partner - offset
                if 0 <= focal_position < len(partners) and 0 <= partner_position < len(partners):
                    valid += 1
                    intact += int(partners[focal_position] == partner_position)
            intact_fraction = intact / valid
        else:
            opposite_summary = (np.nan,) * 4
            intact_fraction = np.nan
        names = ("paired_fraction", "transition_density", "unpaired_run_density", "longest_unpaired_fraction")
        for name, value in zip(names, focal_summary):
            result[f"target_geometry__focal_w{radius}_{name}"] = value
        for name, value in zip(names, opposite_summary):
            result[f"target_geometry__partner_w{radius}_{name}"] = value
        result[f"target_geometry__w{radius}_paired_fraction_asymmetry"] = (
            focal_summary[0] - opposite_summary[0]
        )
        result[f"target_geometry__w{radius}_helix_intact_fraction"] = float(
            intact_fraction
        )
    return result


def run(site_master_path: Path, context: str, output_path: Path) -> dict:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("target-geometry construction received a sealed split")
    table = master[master["context"] == context].copy()
    rows = []
    for row in table.itertuples(index=False):
        values = {
            "example_id": row.example_id,
            "context": row.context,
            "pair_id": row.pair_id,
            "pair_local_key": row.pair_local_key,
        }
        values.update(
            contact_features(row.sequence, row.structure, int(row.local_position))
        )
        rows.append(values)
    features = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    summary = {
        "scope": "target-centered development sequence/structure only",
        "context": context,
        "rows": len(features),
        "pairs": int(features["pair_id"].nunique()),
        "feature_count": int(sum(column.startswith("target_geometry__") for column in features)),
        "windows": WINDOWS,
        "canonical_test_access": False,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    register(
        site_master_path.parents[2],
        {
            "run_id": f"target_geometry_{context}",
            "experiment_type": "target_centered_contact_geometry",
            "context": context,
            "input_hashes": {"site_master": sha256_file(site_master_path)},
            "fold_definition": "outcome-independent features on canonical development rows",
            "seed": "none",
            "features": summary,
            "hyperparameters": {"prespecified_radii": WINDOWS},
            "artifact": str(output_path),
            "metrics": summary,
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/features/target_geometry_MuscleSkeletal.parquet"),
    )
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(json.dumps(run(resolve(args.site_master), args.context, resolve(args.output)), indent=2))


if __name__ == "__main__":
    main()
