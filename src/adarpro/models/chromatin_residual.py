"""Nested grouped residual test of exact-tissue H3K36me3 peak features."""

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
    observed = sorted(
        column for column in table if column.startswith("chromatin__h3k36me3__")
    )
    shifted = sorted(
        column
        for column in table
        if column.startswith("control_shift1mb__h3k36me3__")
    )
    occupancy = [column for column in observed if "__overlap__" in column]
    site = [column for column in observed if "__site__" in column]
    pair = [column for column in observed if "__pair__" in column]
    return {
        "baseline_logit": baseline,
        "baseline_plus_h3k36me3_site": baseline + site,
        "baseline_plus_h3k36me3_pair": baseline + pair,
        "baseline_plus_h3k36me3_multiscale_occupancy": baseline + occupancy,
        "baseline_plus_h3k36me3_multiscale": baseline + observed,
        "baseline_plus_shift1mb_control": baseline + shifted,
    }


def run(
    site_master_path: Path,
    chromatin_path: Path,
    context: str,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    columns = [
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
    ]
    master = pd.read_parquet(site_master_path, columns=columns)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("chromatin residual input contains a sealed split")
    table = master[master["context"] == context].copy()
    annotations = pd.read_parquet(chromatin_path)
    annotations = annotations[annotations["context"] == context].drop(columns="context")
    table = table.merge(
        annotations,
        on=["example_id", "pair_id", "pair_local_key"],
        validate="one_to_one",
    ).reset_index(drop=True)
    if table["baseline_oof_probability"].isna().any():
        raise RuntimeError(f"incomplete baseline OOF predictions for {context}")
    sets = feature_sets(table)
    if len(sets["baseline_plus_h3k36me3_multiscale"]) == 1:
        raise RuntimeError(f"no tissue-matched H3K36me3 features for {context}")

    prediction_tables, selections = [], []
    for model_name, model_columns in sets.items():
        prediction, selection = fit_cross_fitted(
            table, model_columns, model_name, seed
        )
        prediction_tables.append(prediction)
        selections.extend(selection)
    dimension = len(sets["baseline_plus_h3k36me3_multiscale"]) - 1
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

    metric_rows, bootstrap_tables, matched_bootstrap_tables = [], [], []
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
        matched_group = group.copy()
        matched_group["prediction"] = matched_group["matched_recall_prediction"]
        matched_reference = reference.copy()
        matched_reference["prediction"] = matched_reference[
            "matched_recall_prediction"
        ]
        matched_bootstrap = bootstrap_differences(
            matched_group, matched_reference, bootstrap_replicates, seed + 2000
        )
        matched_bootstrap["model"] = model_name
        matched_bootstrap_tables.append(matched_bootstrap)
    metric_table = pd.DataFrame(metric_rows)
    bootstrap = pd.concat(bootstrap_tables, ignore_index=True)
    confidence = (
        bootstrap.groupby(["model", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    matched_bootstrap = pd.concat(matched_bootstrap_tables, ignore_index=True)
    matched_confidence = (
        matched_bootstrap.groupby(["model", "metric"])["difference"]
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
    matched_bootstrap.to_parquet(
        destination / "pair_bootstrap_matched_recall.parquet", index=False
    )
    matched_confidence.to_csv(
        destination / "pair_bootstrap_matched_recall_ci.csv", index=False
    )
    summary = {
        "scope": "canonical train+valid grouped OOF predictions only",
        "context": context,
        "rows": len(table),
        "pairs": int(table["pair_id"].nunique()),
        "feature_sets": {
            name: len(model_columns) - 1 for name, model_columns in sets.items()
        },
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
            "run_id": f"chromatin_residual_{context}_seed{seed}",
            "experiment_type": "nested_chromatin_residual",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "chromatin": sha256_file(chromatin_path),
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
    parser.add_argument("--chromatin", type=Path, default=Path("data/derived/features/chromatin_h3k36me3.parquet"))
    parser.add_argument("--context", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/runs/chromatin_residual"))
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
                resolve(args.chromatin),
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
