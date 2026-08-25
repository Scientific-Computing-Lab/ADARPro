"""Build a development-pair-only matched cross-tissue editing atlas."""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.mapping.reconstruct import (
    CONTEXT_TO_RAW,
    expected_genomic_zero_based,
)
from adarpro.quarantine import sha256_file

TISSUES = ("Artery", "Brain", "Liver", "MuscleSkeletal")


def editing_state(value: float) -> str:
    if value >= 15.0:
        return "edited"
    if value < 1.0:
        return "nonedited"
    return "intermediate"


def read_development_raw_sites(
    site_master_path: Path,
    raw_dir: Path,
    contexts: tuple[str, ...] = TISSUES,
) -> pd.DataFrame:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("site master contains held-out records")
    pair_columns = [
        "pair_id",
        "sequence",
        "structure",
        "chr1",
        "start1",
        "end1",
        "chr2",
        "start2",
        "end2",
        "strand",
        "development_fold",
    ]
    pairs = master[pair_columns].drop_duplicates()
    if pairs["pair_id"].duplicated().any() or pairs["sequence"].duplicated().any():
        raise RuntimeError("development pair map is not one-to-one")
    pair_lookup = pairs.set_index("sequence").to_dict("index")
    frames = []
    usecols = ["PositionChr", "PositionStart", "structure", "L", "R", "EditingIndex"]
    for tissue in contexts:
        raw_path = raw_dir / CONTEXT_TO_RAW[tissue]
        selected_chunks = []
        for chunk in pd.read_csv(raw_path, usecols=usecols, chunksize=20_000):
            chunk = chunk.dropna(
                subset=["PositionChr", "PositionStart", "structure", "L", "R", "EditingIndex"]
            ).copy()
            chunk["L"] = chunk["L"].astype(str).str.upper()
            chunk["R"] = chunk["R"].astype(str).str.upper()
            chunk["sequence"] = chunk["L"] + "A" + chunk["R"]
            chunk = chunk[chunk["sequence"].isin(pair_lookup)]
            if not chunk.empty:
                selected_chunks.append(chunk)
        if not selected_chunks:
            raise RuntimeError(f"no development-pair raw sites for {tissue}")
        tissue_table = pd.concat(selected_chunks, ignore_index=True)
        annotation = pd.DataFrame(
            [pair_lookup[sequence] for sequence in tissue_table["sequence"]]
        ).rename(columns={"structure": "canonical_structure"}).reset_index(drop=True)
        tissue_table = pd.concat(
            [tissue_table.reset_index(drop=True), annotation], axis=1
        )
        if not (tissue_table["structure"] == tissue_table["canonical_structure"]).all():
            raise RuntimeError(f"raw/canonical structure disagreement for {tissue}")
        tissue_table = tissue_table.rename(
            columns={"EditingIndex": "editing_index"}
        ).drop(columns="canonical_structure")
        tissue_table["context"] = tissue
        tissue_table["local_position"] = tissue_table["L"].str.len().astype(int)
        tissue_table["example_id"] = (
            tissue
            + ":raw:"
            + tissue_table["pair_id"].astype(str)
            + ":"
            + tissue_table["local_position"].astype(str)
        )
        tissue_table["expected_position"] = tissue_table.apply(
            expected_genomic_zero_based, axis=1
        )
        if not np.array_equal(
            tissue_table["expected_position"].to_numpy(),
            tissue_table["PositionStart"].astype(int).to_numpy(),
        ):
            raise RuntimeError(f"coordinate transform disagreement for {tissue}")
        if not (tissue_table["PositionChr"] == tissue_table["chr1"]).all():
            raise RuntimeError(f"chromosome disagreement for {tissue}")
        tissue_table["genomic_position_0based"] = tissue_table["expected_position"]
        tissue_table["genomic_position_1based"] = tissue_table["expected_position"] + 1
        tissue_table["site_id"] = (
            tissue_table["chr1"]
            + ":"
            + tissue_table["genomic_position_1based"].astype(str)
            + ":"
            + tissue_table["strand"]
        )
        tissue_table["pair_local_key"] = (
            tissue_table["pair_id"].astype(str)
            + ":"
            + tissue_table["local_position"].astype(str)
        )
        tissue_table["editing_state"] = tissue_table["editing_index"].map(editing_state)
        tissue_table["binary_label"] = tissue_table["editing_state"].map(
            {"nonedited": 0, "edited": 1}
        ).astype("Int64")
        duplicates = tissue_table[tissue_table.duplicated(["context", "site_id"], keep=False)]
        if not duplicates.empty:
            variability = duplicates.groupby(["context", "site_id"]).agg(
                pair_ids=("pair_id", "nunique"),
                positions=("local_position", "nunique"),
                levels=("editing_index", "nunique"),
            )
            if (variability > 1).any(axis=None):
                raise RuntimeError(f"ambiguous duplicate biological sites for {tissue}")
            tissue_table = tissue_table.drop_duplicates(["context", "site_id"])
        frames.append(tissue_table)
    long = pd.concat(frames, ignore_index=True)
    identity = long.groupby("pair_local_key").agg(
        site_ids=("site_id", "nunique"), strands=("strand", "nunique")
    )
    if (identity != 1).any(axis=None):
        raise RuntimeError("pair/local identity is inconsistent across tissues")
    return long


def categorize(states: pd.Series) -> str:
    values = set(states.dropna())
    if values == {"edited"}:
        return "edited_in_all_observed"
    if values == {"nonedited"}:
        return "nonedited_in_all_observed"
    if "edited" in values and "nonedited" in values:
        return "strict_edited_nonedited_switch"
    if "intermediate" in values:
        return "intermediate_involved"
    return "other"


def analyze(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = long.pivot_table(
        index=["site_id", "pair_id", "pair_local_key", "development_fold"],
        columns="context",
        values="editing_index",
        aggfunc="first",
    ).reset_index()
    for tissue in TISSUES:
        wide[f"{tissue}_state"] = wide[tissue].map(
            lambda value: editing_state(value) if pd.notna(value) else pd.NA
        )
    state_columns = [f"{tissue}_state" for tissue in TISSUES]
    wide["observed_tissues"] = wide[list(TISSUES)].notna().sum(axis=1)
    wide["switch_category"] = wide[state_columns].apply(categorize, axis=1)
    matched = wide[wide["observed_tissues"] >= 2].copy()
    switch_rows = []
    difference_rows = []
    for first, second in permutations(TISSUES, 2):
        shared = matched[matched[first].notna() & matched[second].notna()]
        switch_rows.append(
            {
                "edited_tissue": first,
                "nonedited_tissue": second,
                "shared_sites": len(shared),
                "strict_switch_sites": int(
                    ((shared[f"{first}_state"] == "edited") & (shared[f"{second}_state"] == "nonedited")).sum()
                ),
            }
        )
        differences = shared[first] - shared[second]
        difference_rows.append(
            {
                "first_tissue": first,
                "second_tissue": second,
                "shared_sites": len(shared),
                "mean_editing_index_difference": float(differences.mean()),
                "median_editing_index_difference": float(differences.median()),
            }
        )
    return matched, pd.DataFrame(switch_rows), pd.DataFrame(difference_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument(
        "--raw-editing-dir", type=Path, default=Path("data/external/gtex_editing")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/derived/cross_tissue")
    )
    args = parser.parse_args()
    repository = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repository / path

    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    long = read_development_raw_sites(resolve(args.site_master), resolve(args.raw_editing_dir))
    combined = read_development_raw_sites(
        resolve(args.site_master), resolve(args.raw_editing_dir), ("Combined",)
    )
    matched, switch_matrix, differences = analyze(long)
    long_path = output / "development_raw_individual_tissue_sites.parquet"
    combined_path = output / "development_raw_combined_sites.parquet"
    matched_path = output / "matched_sites.parquet"
    switch_path = output / "label_switch_matrix.csv"
    difference_path = output / "continuous_pairwise_differences.csv"
    long.to_parquet(long_path, index=False)
    combined.to_parquet(combined_path, index=False)
    matched.to_parquet(matched_path, index=False)
    switch_matrix.to_csv(switch_path, index=False)
    differences.to_csv(difference_path, index=False)
    categories = matched["switch_category"].value_counts().astype(int).to_dict()
    summary = {
        "scope": "raw sites on the 705 development pairs only; canonical held-out pairs excluded",
        "combined_excluded": True,
        "combined_natural_prevalence_rows": int(combined["binary_label"].notna().sum()),
        "raw_site_tissue_rows": len(long),
        "unique_sites": int(long["site_id"].nunique()),
        "sites_observed_in_at_least_two_tissues": len(matched),
        "categories": categories,
        "strict_switch_sites": int(
            (matched["switch_category"] == "strict_edited_nonedited_switch").sum()
        ),
        "natural_prevalence_binary_rows": int(long["binary_label"].notna().sum()),
        "intermediate_rows": int((long["editing_state"] == "intermediate").sum()),
        "artifacts": {
            path.name: sha256_file(path)
            for path in (
                long_path,
                combined_path,
                matched_path,
                switch_path,
                difference_path,
            )
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
