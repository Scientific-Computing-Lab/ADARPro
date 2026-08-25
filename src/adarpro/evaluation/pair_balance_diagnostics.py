"""Diagnose what pair-first sampling changes in grouped development predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from adarpro.quarantine import sha256_file
from adarpro.registry import register


def binary_log_loss(labels: pd.Series, probabilities: pd.Series) -> np.ndarray:
    values = np.clip(probabilities.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    truth = labels.to_numpy(dtype=float)
    return -(truth * np.log(values) + (1 - truth) * np.log(1 - values))


def run(
    repository: Path,
    pair_path: Path,
    site_path: Path,
    raw_path: Path,
    output_dir: Path,
) -> dict:
    pair = pd.read_parquet(pair_path).sort_values("example_id").reset_index(drop=True)
    site = pd.read_parquet(site_path).sort_values("example_id").reset_index(drop=True)
    identity = [
        "example_id",
        "context",
        "pair_id",
        "pair_local_key",
        "label",
        "development_fold",
    ]
    if not pair[identity].equals(site[identity]):
        raise RuntimeError("pair- and site-balanced predictions are not aligned")
    if set(pair["development_fold"]) != set(range(5)):
        raise RuntimeError(
            "diagnostic input does not contain all five development folds"
        )
    context_values = pair["context"].unique()
    if len(context_values) != 1:
        raise RuntimeError("diagnostic input must contain exactly one context")
    context = str(context_values[0])

    joined = pair.copy()
    joined["site_probability"] = site["probability"]
    joined["site_prediction"] = site["prediction"]
    raw = pd.read_parquet(raw_path)
    raw = raw[(raw["context"] == context) & raw["binary_label"].notna()]
    raw_summary = (
        raw.groupby("pair_id")
        .agg(
            raw_site_count=("binary_label", "size"),
            raw_high_count=("binary_label", "sum"),
        )
        .reset_index()
    )
    raw_summary["raw_high_fraction"] = (
        raw_summary["raw_high_count"] / raw_summary["raw_site_count"]
    )
    joined = joined.merge(raw_summary, on="pair_id", validate="many_to_one")
    joined["development_reference_log_loss"] = binary_log_loss(
        joined["label"], joined["baseline_oof_probability"]
    )
    joined["site_balanced_log_loss"] = binary_log_loss(
        joined["label"], joined["site_probability"]
    )
    joined["pair_balanced_log_loss"] = binary_log_loss(
        joined["label"], joined["probability"]
    )
    joined["development_reference_fp"] = joined["label"].eq(0) & joined[
        "baseline_oof_prediction"
    ].eq(1)
    joined["pair_balanced_fp_rescue"] = joined["development_reference_fp"] & joined[
        "prediction"
    ].eq(0)
    joined["site_balanced_fp_rescue"] = joined["development_reference_fp"] & joined[
        "site_prediction"
    ].eq(0)

    per_pair = (
        joined.groupby("pair_id")
        .agg(
            raw_site_count=("raw_site_count", "first"),
            raw_high_fraction=("raw_high_fraction", "first"),
            canonical_site_count=("label", "size"),
            canonical_high_fraction=("label", "mean"),
            development_reference_log_loss=("development_reference_log_loss", "mean"),
            site_balanced_log_loss=("site_balanced_log_loss", "mean"),
            pair_balanced_log_loss=("pair_balanced_log_loss", "mean"),
        )
        .reset_index()
    )
    per_pair["pair_gain_vs_development_reference"] = (
        per_pair["development_reference_log_loss"] - per_pair["pair_balanced_log_loss"]
    )
    per_pair["pair_gain_vs_site_balanced"] = (
        per_pair["site_balanced_log_loss"] - per_pair["pair_balanced_log_loss"]
    )
    per_pair["raw_site_count_quintile"] = pd.qcut(
        per_pair["raw_site_count"], 5, labels=False, duplicates="drop"
    )
    quintile_lookup = per_pair.set_index("pair_id")["raw_site_count_quintile"]
    joined["raw_site_count_quintile"] = joined["pair_id"].map(quintile_lookup)
    quintiles = (
        per_pair.groupby("raw_site_count_quintile", observed=True)
        .agg(
            pairs=("pair_id", "size"),
            raw_site_count_median=("raw_site_count", "median"),
            development_reference_log_loss=("development_reference_log_loss", "mean"),
            site_balanced_log_loss=("site_balanced_log_loss", "mean"),
            pair_balanced_log_loss=("pair_balanced_log_loss", "mean"),
            pair_gain_vs_development_reference=(
                "pair_gain_vs_development_reference",
                "mean",
            ),
            pair_gain_vs_site_balanced=("pair_gain_vs_site_balanced", "mean"),
        )
        .reset_index()
    )
    fp_quintiles = (
        joined.groupby("raw_site_count_quintile", observed=True)
        .agg(
            canonical_rows=("label", "size"),
            development_reference_fp=("development_reference_fp", "sum"),
            pair_balanced_fp_rescue=("pair_balanced_fp_rescue", "sum"),
            site_balanced_fp_rescue=("site_balanced_fp_rescue", "sum"),
        )
        .reset_index()
    )
    correlations = []
    for outcome in (
        "pair_gain_vs_development_reference",
        "pair_gain_vs_site_balanced",
    ):
        for covariate in (
            "raw_site_count",
            "canonical_site_count",
            "raw_high_fraction",
        ):
            result = spearmanr(per_pair[covariate], per_pair[outcome])
            correlations.append(
                {
                    "covariate": covariate,
                    "outcome": outcome,
                    "spearman": float(result.statistic),
                    "p_value": float(result.pvalue),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_pair.to_parquet(output_dir / "per_pair_diagnostics.parquet", index=False)
    quintiles.to_csv(output_dir / "raw_site_count_quintiles.csv", index=False)
    fp_quintiles.to_csv(output_dir / "fp_rescue_by_quintile.csv", index=False)
    pd.DataFrame(correlations).to_csv(
        output_dir / "spearman_correlations.csv", index=False
    )
    summary = {
        "scope": "descriptive grouped-development diagnostic only",
        "context": context,
        "rows": len(joined),
        "pairs": int(joined["pair_id"].nunique()),
        "canonical_test_access": False,
        "correlations": correlations,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"pair_balance_diagnostic_{context}",
            "experiment_type": "descriptive_pair_balance_diagnostic",
            "context": context,
            "input_hashes": {
                "pair_balanced_oof": sha256_file(pair_path),
                "site_balanced_oof": sha256_file(site_path),
                "raw_development_sites": sha256_file(raw_path),
            },
            "fold_definition": "completed identical five-fold global-pair OOF predictions",
            "seed": "none",
            "features": "raw eligible-site count per development pair",
            "hyperparameters": {"quintiles": 5},
            "artifact": str((output_dir / "summary.json").relative_to(repository)),
            "metrics": summary,
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--pair-balanced", type=Path, required=True)
    parser.add_argument("--site-balanced", type=Path, required=True)
    parser.add_argument(
        "--raw-sites",
        type=Path,
        default=Path(
            "data/derived/cross_tissue/development_raw_individual_tissue_sites.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/richer_supervision/pair_balance_diagnostic"),
    )
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(
        json.dumps(
            run(
                root,
                resolve(args.pair_balanced),
                resolve(args.site_balanced),
                resolve(args.raw_sites),
                resolve(args.output_dir),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
