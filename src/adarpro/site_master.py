"""Build the authoritative development-only site table and grouped folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.quarantine import CONTEXTS, development_file, sha256_file

DISPLAY_CONTEXT = {
    "Artery": "Artery Tibial",
    "Brain": "Brain/Cerebellum",
    "Liver": "Liver",
    "MuscleSkeletal": "Muscle Skeletal",
    "Combined": "Combined (pooled GTEx endpoint)",
}
INDIVIDUAL_TISSUES = frozenset(
    {"Artery", "Brain", "Liver", "MuscleSkeletal"}
)


def parse_record(line: str) -> dict:
    record = json.loads(line)
    user = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "user"
    )
    answer = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "assistant"
    )
    fields = {}
    for field in user.split(", "):
        if ":" in field:
            key, value = field.split(":", 1)
            fields[key] = value
    left = fields["L"].upper()
    focal = fields["A"].upper()
    right = fields["R"].upper()
    sequence = left + focal + right
    structure = fields["Alu Vienna Structure"]
    return {
        "left_sequence": left,
        "focal_base": focal,
        "right_sequence": right,
        "sequence": sequence,
        "structure": structure,
        "local_position": len(left),
        "label": 1 if answer.strip().lower() == "yes" else 0,
    }


def _read_context_split(repository: Path, context: str, split: str) -> pd.DataFrame:
    jsonl_path = development_file(repository, context, split, "jsonl")
    metadata_path = development_file(repository, context, split, "metadata.csv")
    with jsonl_path.open() as handle:
        records = [parse_record(line) for line in handle]
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != len(records):
        raise RuntimeError(
            f"row mismatch for {context}/{split}: {len(records)} versus {len(metadata)}"
        )
    table = pd.concat([metadata.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    if not np.array_equal(
        table["label"].to_numpy(), (table["yes_no"] == "yes").astype(int).to_numpy()
    ):
        raise RuntimeError(f"label disagreement in {context}/{split}")
    if not (table["split"] == split).all():
        raise RuntimeError(f"metadata split disagreement in {context}/{split}")
    table["context"] = context
    table["context_display"] = DISPLAY_CONTEXT[context]
    table["is_individual_tissue"] = context in INDIVIDUAL_TISSUES
    table["source_split"] = split
    table["source_row"] = np.arange(len(table), dtype=int)
    table["jsonl_sha256"] = sha256_file(jsonl_path)
    table["metadata_sha256"] = sha256_file(metadata_path)
    return table


def _hash_tiebreak(pair_id: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def assign_balanced_pair_folds(
    sites: pd.DataFrame, n_splits: int = 5, seed: int = 20260821
) -> pd.DataFrame:
    """Greedily balance context-by-label counts while keeping pairs intact."""

    dimensions = [
        (context, label) for context in CONTEXTS for label in (0, 1)
    ]
    grouped_rows = []
    for pair_id, group in sites.groupby("pair_id", sort=True):
        counts = np.asarray(
            [
                int(((group["context"] == context) & (group["label"] == label)).sum())
                for context, label in dimensions
            ],
            dtype=float,
        )
        grouped_rows.append((int(pair_id), counts))
    totals = np.sum([counts for _, counts in grouped_rows], axis=0)
    scale = np.maximum(totals / n_splits, 1.0)
    ordered = sorted(
        grouped_rows,
        key=lambda item: (
            -float(np.sum((item[1] / scale) ** 2)),
            -float(item[1].sum()),
            _hash_tiebreak(item[0], seed),
        ),
    )
    fold_counts = np.zeros((n_splits, len(dimensions)), dtype=float)
    fold_pairs = np.zeros(n_splits, dtype=int)
    assignments: dict[int, int] = {}
    for pair_id, counts in ordered:
        candidates = []
        for fold in range(n_splits):
            proposed = fold_counts.copy()
            proposed[fold] += counts
            count_loss = float(np.sum(((proposed - totals / n_splits) / scale) ** 2))
            pair_loss = float(
                np.sum((fold_pairs + np.eye(1, n_splits, fold).ravel() - len(grouped_rows) / n_splits) ** 2)
                / max(1.0, len(grouped_rows) ** 2)
            )
            candidates.append((count_loss + pair_loss, int(fold_pairs[fold]), fold))
        chosen = min(candidates)[2]
        assignments[pair_id] = chosen
        fold_counts[chosen] += counts
        fold_pairs[chosen] += 1
    output = pd.DataFrame(
        {
            "pair_id": sorted(assignments),
            "development_fold": [assignments[pair] for pair in sorted(assignments)],
            "fold_seed": seed,
            "n_splits": n_splits,
        }
    )
    if output["pair_id"].duplicated().any():
        raise RuntimeError("a pair received more than one fold")
    return output


def build(repository: Path, output: Path, fold_output: Path) -> tuple[pd.DataFrame, dict]:
    repository = repository.resolve()
    tables = [
        _read_context_split(repository, context, split)
        for context in CONTEXTS
        for split in ("train", "valid")
    ]
    sites = pd.concat(tables, ignore_index=True)
    if (sites["focal_base"] != "A").any():
        raise RuntimeError("non-adenosine focal base in canonical data")
    if (sites["sequence"].str.len() != sites["structure"].str.len()).any():
        raise RuntimeError("sequence/structure length mismatch")
    if sites.groupby("pair_id")["sequence"].nunique().max() != 1:
        raise RuntimeError("pair_id does not map to exactly one sequence")
    if sites.groupby("pair_id")["structure"].nunique().max() != 1:
        raise RuntimeError("pair_id does not map to exactly one structure")
    sites["pair_local_key"] = (
        sites["pair_id"].astype(str) + ":" + sites["local_position"].astype(str)
    )
    biological_key_variation = sites.groupby("pair_local_key").agg(
        sequences=("sequence", "nunique"), structures=("structure", "nunique")
    )
    if (biological_key_variation > 1).any(axis=None):
        raise RuntimeError("pair/local site identity is inconsistent across contexts")
    occurrence = sites.groupby(["context", "pair_local_key"]).cumcount()
    sites["example_occurrence"] = occurrence
    sites["example_id"] = (
        sites["context"]
        + ":"
        + sites["pair_local_key"]
        + ":"
        + occurrence.astype(str)
    )
    sites["site_id"] = pd.Series(pd.NA, index=sites.index, dtype="string")
    sites["chromosome"] = pd.Series(pd.NA, index=sites.index, dtype="string")
    sites["genomic_position_1based"] = pd.Series(pd.NA, index=sites.index, dtype="Int64")
    sites["strand"] = pd.Series(pd.NA, index=sites.index, dtype="string")
    sites["genome_build"] = "GRCh38_pending_coordinate_reconstruction"
    sites["editing_index"] = np.nan
    folds = assign_balanced_pair_folds(sites)
    sites = sites.merge(folds[["pair_id", "development_fold"]], on="pair_id", how="left", validate="many_to_one")
    if sites["development_fold"].isna().any():
        raise RuntimeError("missing development fold")
    if sites.groupby("pair_id")["development_fold"].nunique().max() != 1:
        raise RuntimeError("pair leakage across development folds")
    columns = [
        "example_id",
        "context",
        "context_display",
        "is_individual_tissue",
        "pair_id",
        "pair_local_key",
        "site_id",
        "chromosome",
        "genomic_position_1based",
        "strand",
        "genome_build",
        "local_position",
        "label",
        "editing_index",
        "source_split",
        "development_fold",
        "source_row",
        "example_occurrence",
        "sequence",
        "structure",
        "left_sequence",
        "focal_base",
        "right_sequence",
        "pair_prefix",
        "jsonl_sha256",
        "metadata_sha256",
    ]
    sites = sites[columns].sort_values(
        ["context", "pair_id", "local_position", "source_split", "source_row"]
    ).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fold_output.parent.mkdir(parents=True, exist_ok=True)
    sites.to_parquet(output, index=False)
    folds.to_csv(fold_output, index=False)
    summary = {
        "scope": "canonical train+valid only; held-out test excluded",
        "rows": len(sites),
        "contexts": sites.groupby("context").size().astype(int).to_dict(),
        "unique_pairs": int(sites["pair_id"].nunique()),
        "unique_pair_local_sites": int(sites["pair_local_key"].nunique()),
        "source_splits": sites.groupby("source_split").size().astype(int).to_dict(),
        "fold_pairs": folds.groupby("development_fold")["pair_id"].nunique().astype(int).to_dict(),
        "duplicate_context_sites": int(
            sum(count - 1 for count in Counter(zip(sites["context"], sites["pair_local_key"])).values() if count > 1)
        ),
        "site_master_sha256": sha256_file(output),
        "folds_sha256": sha256_file(fold_output),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return sites, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument(
        "--fold-output", type=Path, default=Path("data/derived/folds/pair_folds.csv")
    )
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.output if args.output.is_absolute() else repository / args.output
    fold_output = (
        args.fold_output
        if args.fold_output.is_absolute()
        else repository / args.fold_output
    )
    _, summary = build(repository, output, fold_output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
