"""Genomic gap of every Alu pair in the pair BED, keyed by the pair's sequence (label-free).

The gap stage (PNAS integration, D39) needs each pair's genomic gap. Development rows carry BED coordinates in
``site_master`` (``reconstruct.map_pairs`` matched each pair's sequence to a BED pair rebuilt from the reference
genome). Canonical rows carry only the pair sequence. This module does PNAS's matching once for all 905 BED pairs,
before any canonical-test session: it reads only the BED and the reference, so it reads no label and does not know
which pairs are test pairs. ``lookup_gaps`` then maps a split's rows to their gaps by sequence.

The gap is ``start2 - end1`` and its ``log1p`` clipped at 0: the formula of
``genomic_context.pair_intrinsic_features`` (``geometry__genomic_gap`` and ``geometry__genomic_gap_log1p``).

Checks (any failure stops): 905 pairs; pairs that share a sequence (the chrX/chrY pseudoautosomal copies) share
their gap; every development pair in ``pair_mapping.csv``
rebuilds to its ``site_master`` sequence, and its gap equals ``gap_stage.pair_gap_table`` on ``site_master``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.mapping.reconstruct import read_bed, reconstruct_bed_sequences
from adarpro.models.baseline_oof import upstream_modules
from adarpro.models.gap_stage import pair_gap_table
from adarpro.quarantine import sha256_file

COORDINATES = ["chr1", "start1", "end1", "chr2", "start2", "end2", "strand"]


def normalize_sequence(sequence: str, alphabet: set[str]) -> str:
    """The canonical parser's normalization (``parse_openai_json``): upper case, unknown letters become N."""

    return "".join(base if base in alphabet else "N" for base in str(sequence).upper())


def gap_columns(start2, end1) -> tuple[np.ndarray, np.ndarray]:
    gap_bp = np.asarray(start2, dtype=np.int64) - np.asarray(end1, dtype=np.int64)
    return gap_bp.astype(np.float64), np.log1p(np.maximum(gap_bp, 0)).astype(np.float64)


def gap_table(bed_sequences: pd.DataFrame, alphabet: set[str]) -> pd.DataFrame:
    table = bed_sequences[["bed_row", *COORDINATES]].copy()
    table["sequence"] = [normalize_sequence(value, alphabet) for value in bed_sequences["sequence"]]
    table["gap_bp"], table["gap"] = gap_columns(table["start2"], table["end1"])
    # Copies of one sequence (the chrX/chrY pseudoautosomal pairs) are allowed only when every copy has the same gap,
    # so the gap stays a function of the sequence.
    spread = table.groupby("sequence")["gap_bp"].nunique()
    if (spread > 1).any():
        rows = table.loc[table["sequence"].isin(spread[spread > 1].index), "bed_row"].tolist()
        raise RuntimeError(f"BED pairs share a sequence but not a gap, so a lookup would be ambiguous: rows {rows}")
    table["sequence_copies"] = table.groupby("sequence")["bed_row"].transform("size").astype(int)
    if (table["gap_bp"] < 0).any():
        raise RuntimeError("negative genomic gap (overlapping or misordered arms) in the BED")
    return table.sort_values("bed_row").reset_index(drop=True)


def check_development(table: pd.DataFrame, site_master: pd.DataFrame, pair_mapping: pd.DataFrame,
                      alphabet: set[str]) -> dict:
    """Every development pair must rebuild to its own sequence and get the same gap as the development code."""

    pairs = site_master[["pair_id", "sequence", *COORDINATES, "structure"]].drop_duplicates("pair_id")
    mapping = pair_mapping[["pair_id", "bed_row"]]
    if mapping["pair_id"].duplicated().any() or mapping["bed_row"].duplicated().any():
        raise RuntimeError("pair_mapping.csv is not one-to-one")
    merged = pairs.merge(mapping, on="pair_id", how="left", validate="one_to_one")
    if merged["bed_row"].isna().any():
        raise RuntimeError("a development pair has no BED row in pair_mapping.csv")
    merged = merged.merge(table, on="bed_row", how="left", suffixes=("_dev", ""), validate="one_to_one")
    rebuilt = merged["sequence"].to_numpy()
    expected = np.array([normalize_sequence(value, alphabet) for value in merged["sequence_dev"]])
    sequence_mismatch = int((rebuilt != expected).sum())
    coordinate_mismatch = int(sum((merged[f"{c}_dev"] != merged[c]).sum() for c in COORDINATES))
    development_gaps = pair_gap_table(site_master)
    compare = development_gaps.merge(merged[["pair_id", "gap_bp", "gap"]], on="pair_id", suffixes=("_dev", ""),
                                     validate="one_to_one")
    gap_mismatch = int(((compare["gap_dev"] != compare["gap"]) | (compare["gap_bp_dev"] != compare["gap_bp"])).sum())
    result = {"development_pairs": len(merged), "sequence_mismatches": sequence_mismatch,
              "coordinate_mismatches": coordinate_mismatch, "gap_mismatches": gap_mismatch}
    if sequence_mismatch or coordinate_mismatch or gap_mismatch or len(compare) != len(merged):
        raise RuntimeError(f"development pairs do not reproduce from the BED and reference: {result}")
    return result


def lookup_gaps(sequences: pd.Series, gaps: pd.DataFrame, alphabet: set[str]) -> pd.DataFrame:
    """Rows' BED row, gap_bp and gap by sequence; every sequence must match exactly one BED pair."""

    keys = pd.Series([normalize_sequence(value, alphabet) for value in sequences], index=sequences.index)
    unique = gaps.sort_values("bed_row").drop_duplicates("sequence")  # copies share the gap (gap_table)
    found = keys.to_frame("sequence").merge(unique[["sequence", "bed_row", "sequence_copies", "gap_bp", "gap"]],
                                            on="sequence", how="left", validate="many_to_one")
    if found["bed_row"].isna().any():
        raise RuntimeError(f"{int(found['bed_row'].isna().sum())} rows match no BED pair by sequence")
    found.index = sequences.index
    return found[["bed_row", "sequence_copies", "gap_bp", "gap"]].astype({"bed_row": int, "sequence_copies": int})


def run(repository: Path, bed_path: Path, fasta_path: Path, site_master_path: Path, pair_mapping_path: Path,
        output_path: Path) -> dict:
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite {output_path}")
    alphabet = set(upstream_modules(repository).BASE_MAP)
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("site_master holds a sealed split")
    table = gap_table(reconstruct_bed_sequences(read_bed(bed_path), fasta_path), alphabet)
    checks = check_development(table, master, pd.read_csv(pair_mapping_path), alphabet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)

    def relative(path: Path) -> str:
        return str(path.relative_to(repository)) if path.is_relative_to(repository) else str(path)

    provenance = {
        "scope": "genomic gap of every BED Alu pair, keyed by its rebuilt sequence; no label and no split is read",
        "gap": "start2 - end1 and log1p(max(gap, 0)), the formula of genomic_context.pair_intrinsic_features",
        "sequence": "reconstruct.reconstruct_bed_sequences, then parse_openai_json's normalization",
        "bed": {"path": relative(bed_path), "sha256": sha256_file(bed_path)},
        "reference_fasta": {"path": str(fasta_path), "bytes": fasta_path.stat().st_size,
                            "sha256": sha256_file(fasta_path)},
        "site_master_sha256": sha256_file(site_master_path),
        "pair_mapping_sha256": sha256_file(pair_mapping_path),
        "pairs": len(table),
        "pairs_sharing_a_sequence": int((table["sequence_copies"] > 1).sum()),
        "development_check": checks,
        "output": {"path": relative(output_path), "sha256": sha256_file(output_path)},
        "canonical_test_access": False,
    }
    output_path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--alu-bed", type=Path, default=Path("data/raw/alu_pairs/Pair_Alu_withStrand.bed"))
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument("--site-master", type=Path, default=Path("data/derived/site_master.parquet"))
    parser.add_argument("--pair-mapping", type=Path, default=Path("data/derived/genomic_mapping/pair_mapping.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/genomic_mapping/bed_pair_gaps.parquet"))
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(json.dumps(run(root, resolve(args.alu_bed), resolve(args.reference_fasta), resolve(args.site_master),
                         resolve(args.pair_mapping), resolve(args.output)), indent=2))


if __name__ == "__main__":
    main()
