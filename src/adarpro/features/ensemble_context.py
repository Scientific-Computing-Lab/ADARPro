"""Map precomputed genomic-context ensemble folding to development pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.quarantine import sha256_file


def build_features(site_master: pd.DataFrame, folding: pd.DataFrame) -> pd.DataFrame:
    pair_columns = [
        "pair_id",
        "bed_row",
        "chr1",
        "start1",
        "end1",
        "start2",
        "end2",
        "strand",
    ]
    pairs = site_master[pair_columns].drop_duplicates("pair_id")
    annotated = pairs.merge(
        folding,
        left_on="bed_row",
        right_on="idx",
        how="left",
        suffixes=("_model", "_fold"),
        validate="one_to_one",
    )
    mapped = annotated["idx"].notna()
    coordinate_pairs = (
        ("chr1", "chr"),
        ("start1_model", "start1_fold"),
        ("end1_model", "end1_fold"),
        ("start2_model", "start2_fold"),
        ("end2_model", "end2_fold"),
        ("strand_model", "strand_fold"),
    )
    for left, right in coordinate_pairs:
        if not (
            annotated.loc[mapped, left].astype(str).str.replace(".0", "", regex=False)
            == annotated.loc[mapped, right].astype(str).str.replace(".0", "", regex=False)
        ).all():
            raise RuntimeError(f"context-fold coordinate mismatch: {left} vs {right}")
    arm_length = annotated["arm1_len"].replace(0, np.nan)
    context_length = annotated["ctx_len"].replace(0, np.nan)
    features = pd.DataFrame(
        {
            "pair_id": annotated["pair_id"],
            "ensemble_context__available": mapped.astype(float),
            "ensemble_context__context_length_log1p": np.log1p(
                annotated["ctx_len"]
            ),
            "ensemble_context__arm1_arm2_paired_fraction": annotated["dsrna_frac"],
            "ensemble_context__arm1_other_paired_fraction": annotated[
                "arm1_paired_other"
            ]
            / arm_length,
            "ensemble_context__arm1_unpaired_fraction": annotated["arm1_unpaired"]
            / arm_length,
            "ensemble_context__mfe_per_nt": annotated["ctx_mfe_kcal"]
            / context_length,
            "ensemble_context__ensemble_free_energy_per_nt": annotated[
                "ensemble_free_energy"
            ]
            / context_length,
            "ensemble_context__delta_delta_g_per_nt": annotated["delta_delta_g"]
            / context_length,
            "ensemble_context__mean_arm1_arm2_probability": annotated[
                "mean_arm1_arm2_prob"
            ],
        }
    )
    keys = site_master[["example_id", "context", "pair_id", "pair_local_key"]]
    return keys.merge(features, on="pair_id", how="left", validate="many_to_one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-master", type=Path, default=Path("data/derived/site_master.parquet"))
    parser.add_argument("--folding", type=Path, default=Path("analyses/substrate_stability/raw_data/context_folding_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/features/ensemble_context.parquet"))
    parser.add_argument("--provenance", type=Path, default=Path("provenance/ensemble_context_features.json"))
    args = parser.parse_args()
    site_master = pd.read_parquet(args.site_master)
    if set(site_master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("ensemble-context feature build received a sealed split")
    folding = pd.read_csv(args.folding)
    result = build_features(site_master, folding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    payload = {
        "source": str(args.folding),
        "source_sha256": sha256_file(args.folding),
        "source_scope": "outcome-independent ViennaRNA folding with native pair gap and +/-200 nt flanks",
        "join": "development pair bed_row to context-fold idx, coordinate-validated",
        "rows": len(result),
        "development_pairs": int(result["pair_id"].nunique()),
        "mapped_pairs": int(
            result.loc[result["ensemble_context__available"] == 1, "pair_id"].nunique()
        ),
        "canonical_test_access": False,
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
