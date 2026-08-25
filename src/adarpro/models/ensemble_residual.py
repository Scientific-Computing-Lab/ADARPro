"""Nested residual test of native-gap, flank-aware ensemble folding features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adarpro.models.genomic_residual import fold_differences
from adarpro.models.residual import (
    bootstrap_differences,
    fit_cross_fitted,
    metrics,
    rescue_metrics,
)
from adarpro.quarantine import sha256_file
from adarpro.registry import register


def feature_sets(table: pd.DataFrame) -> dict[str, list[str]]:
    baseline = ["baseline_oof_logit"]
    available = ["ensemble_context__available"]
    structural = available + sorted(
        column
        for column in table
        if column.startswith("ensemble_context__arm1_")
    )
    thermodynamic = available + sorted(
        column
        for column in table
        if column.startswith("ensemble_context__")
        and any(
            token in column
            for token in (
                "mfe_per_nt",
                "ensemble_free_energy_per_nt",
                "delta_delta_g_per_nt",
                "mean_arm1_arm2_probability",
            )
        )
    )
    full = sorted(
        column for column in table if column.startswith("ensemble_context__")
    )
    return {
        "baseline_logit": baseline,
        "baseline_plus_context_dsrna": baseline
        + available
        + ["ensemble_context__arm1_arm2_paired_fraction"],
        "baseline_plus_context_structural_fractions": baseline + structural,
        "baseline_plus_context_thermodynamics": baseline + thermodynamic,
        "baseline_plus_context_ensemble_full": baseline + full,
    }


def run(
    site_master_path: Path,
    feature_path: Path,
    context: str,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    master = pd.read_parquet(
        site_master_path,
        columns=[
            "example_id",
            "context",
            "pair_id",
            "pair_local_key",
            "label",
            "development_fold",
            "source_split",
            "baseline_oof_probability",
            "baseline_oof_logit",
            "baseline_oof_prediction",
        ],
    )
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("ensemble residual input contains a sealed split")
    table = master[master["context"] == context].copy()
    features = pd.read_parquet(feature_path)
    features = features[features["context"] == context].drop(columns="context")
    table = table.merge(
        features,
        on=["example_id", "pair_id", "pair_local_key"],
        validate="one_to_one",
    ).reset_index(drop=True)
    if table["baseline_oof_probability"].isna().any():
        raise RuntimeError(f"incomplete development reference for {context}")
    sets = feature_sets(table)
    prediction_tables, selections = [], []
    for model_name, columns in sets.items():
        predictions, selection = fit_cross_fitted(table, columns, model_name, seed)
        prediction_tables.append(predictions)
        selections.extend(selection)
    dimension = len(sets["baseline_plus_context_ensemble_full"]) - 1
    random_predictions, random_selection = fit_cross_fitted(
        table,
        [],
        "baseline_plus_matched_random",
        seed + 99,
        random_dimension=dimension,
    )
    prediction_tables.append(random_predictions)
    selections.extend(random_selection)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    reference = predictions[predictions["model"] == "baseline_logit"].copy()
    metric_rows, bootstrap_tables = [], []
    for model_name, group in predictions.groupby("model", sort=False):
        row = metrics(group["label"], group["probability"], group["prediction"])
        row.update(rescue_metrics(group, "prediction"))
        row["model"] = model_name
        metric_rows.append(row)
        bootstrap = bootstrap_differences(
            group, reference, bootstrap_replicates, seed + 1000
        )
        bootstrap["model"] = model_name
        bootstrap_tables.append(bootstrap)
    metric_table = pd.DataFrame(metric_rows)
    bootstrap = pd.concat(bootstrap_tables, ignore_index=True)
    confidence = (
        bootstrap.groupby(["model", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    destination = output_dir / context
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(destination / "oof_predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(destination / "fold_selections.csv", index=False)
    metric_table.to_csv(destination / "metrics.csv", index=False)
    fold_differences(predictions, reference).to_csv(
        destination / "fold_metric_differences.csv", index=False
    )
    bootstrap.to_parquet(destination / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(destination / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "canonical train+valid grouped OOF predictions only",
        "context": context,
        "rows": len(table),
        "pairs": int(table["pair_id"].nunique()),
        "mapped_pairs": int(
            table.loc[table["ensemble_context__available"] == 1, "pair_id"].nunique()
        ),
        "feature_sets": {name: len(value) - 1 for name, value in sets.items()},
        "matched_random_dimension": dimension,
        "selection": "nested pair folds; C and threshold selected on inner fold",
        "bootstrap_replicates": bootstrap_replicates,
        "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        site_master_path.parents[2],
        {
            "run_id": f"ensemble_context_residual_{context}_seed{seed}",
            "experiment_type": "nested_ensemble_context_residual",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "features": sha256_file(feature_path),
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": {name: len(value) for name, value in sets.items()},
            "hyperparameters": {
                "model": "L2 logistic regression",
                "selection": "inner-fold F1 and threshold",
                "bootstrap_replicates": bootstrap_replicates,
            },
            "artifact": str(destination / "oof_predictions.parquet"),
            "metrics": metric_table.to_dict(orient="records"),
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--site-master", type=Path, default=Path("data/derived/site_master.parquet"))
    parser.add_argument("--features", type=Path, default=Path("data/derived/features/ensemble_context.parquet"))
    parser.add_argument("--context", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/runs/ensemble_context_residual"))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(
        json.dumps(
            run(
                resolve(args.site_master),
                resolve(args.features),
                args.context,
                resolve(args.output_dir),
                args.bootstrap_replicates,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
