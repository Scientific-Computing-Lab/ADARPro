"""Nested residual test of target-centered structural-contact descriptors."""

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
    all_features = sorted(column for column in table if column.startswith("target_geometry__"))
    core = [
        column
        for column in all_features
        if "_w" not in column and "partner_base" not in column
    ]
    structural = [column for column in all_features if "partner_base" not in column]
    return {
        "baseline_logit": baseline,
        "baseline_plus_contact_core": baseline + core,
        "baseline_plus_multiscale_structure": baseline + structural,
        "baseline_plus_contact_full": baseline + all_features,
    }


def run(
    site_master_path: Path,
    feature_path: Path,
    context: str,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("target-geometry residual input contains a sealed split")
    table = master[master["context"] == context].copy()
    features = pd.read_parquet(feature_path)
    features = features[features["context"] == context].drop(columns="context")
    table = table.merge(
        features,
        on=["example_id", "pair_id", "pair_local_key"],
        validate="one_to_one",
    ).reset_index(drop=True)
    sets = feature_sets(table)
    prediction_tables, selections = [], []
    for model_name, columns in sets.items():
        prediction, selection = fit_cross_fitted(table, columns, model_name, seed)
        prediction_tables.append(prediction)
        selections.extend(selection)
    dimension = len(sets["baseline_plus_contact_full"]) - 1
    random_prediction, random_selection = fit_cross_fitted(
        table,
        [],
        "baseline_plus_matched_random",
        seed + 99,
        random_dimension=dimension,
    )
    prediction_tables.append(random_prediction)
    selections.extend(random_selection)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    reference = predictions[predictions["model"] == "baseline_logit"].copy()
    metric_rows, bootstrap_tables = [], []
    for model_name, group in predictions.groupby("model", sort=False):
        row = metrics(group["label"], group["probability"], group["prediction"])
        row.update(rescue_metrics(group, "prediction"))
        row["model"] = model_name
        metric_rows.append(row)
        bootstrap = bootstrap_differences(group, reference, bootstrap_replicates, seed + 1000)
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
        "feature_sets": {name: len(columns) - 1 for name, columns in sets.items()},
        "matched_random_dimension": dimension,
        "selection": "nested pair folds; C and threshold selected on inner fold",
        "bootstrap_replicates": bootstrap_replicates,
        "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    register(
        site_master_path.parents[2],
        {
            "run_id": f"target_geometry_residual_{context}_seed{seed}",
            "experiment_type": "nested_target_geometry_residual",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "target_geometry": sha256_file(feature_path),
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": {name: len(columns) for name, columns in sets.items()},
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
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/derived/features/target_geometry_MuscleSkeletal.parquet"),
    )
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/runs/target_geometry_residual")
    )
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
