"""Nested grouped residual test of IRAlu architecture and transcript features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from xgboost import XGBClassifier

from adarpro.models.residual import (
    best_f1_threshold,
    bootstrap_differences,
    fit_cross_fitted,
    matched_recall_threshold,
    metrics,
    rescue_metrics,
)
from adarpro.quarantine import sha256_file
from adarpro.registry import register


def feature_sets(table: pd.DataFrame) -> dict[str, list[str]]:
    baseline = ["baseline_oof_logit"]

    def prefixed(*prefixes: str) -> list[str]:
        return sorted(
            column
            for column in table.columns
            if any(column.startswith(prefix) for prefix in prefixes)
        )

    geometry = prefixed("geometry__")
    repeat = prefixed("repeat__")
    transcript = prefixed("transcript__", "site__")
    intrinsic = prefixed("intrinsic__", "intrinsic_site__")
    architecture = sorted(geometry + repeat + transcript)
    chromosome = sorted(column for column in table if column.startswith("chromosome__"))
    return {
        "baseline_logit": baseline,
        "baseline_plus_geometry": baseline + geometry,
        "baseline_plus_repeat": baseline + repeat,
        "baseline_plus_transcript": baseline + transcript,
        "baseline_plus_architecture": baseline + architecture,
        "baseline_plus_intrinsic_engineered_control": baseline + intrinsic,
        "baseline_plus_architecture_and_intrinsic": baseline + architecture + intrinsic,
        "baseline_plus_chromosome_control": baseline + chromosome,
    }


def fold_differences(predictions: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    reference = reference[
        ["example_id", "development_fold", "label", "probability", "prediction"]
    ].rename(
        columns={
            "probability": "reference_probability",
            "prediction": "reference_prediction",
        }
    )
    rows = []
    for model_name, model in predictions.groupby("model", sort=False):
        combined = model.merge(
            reference,
            on=["example_id", "development_fold", "label"],
            validate="one_to_one",
        )
        for fold, subset in combined.groupby("development_fold", sort=True):
            candidate = metrics(subset["label"], subset["probability"], subset["prediction"])
            baseline = metrics(
                subset["label"],
                subset["reference_probability"],
                subset["reference_prediction"],
            )
            for metric in ("auroc", "auprc", "f1", "mcc", "brier"):
                rows.append(
                    {
                        "model": model_name,
                        "fold": int(fold),
                        "metric": metric,
                        "candidate": candidate[metric],
                        "reference": baseline[metric],
                        "difference": candidate[metric] - baseline[metric],
                    }
                )
    return pd.DataFrame(rows)


def fit_shallow_geometry(
    table: pd.DataFrame, columns: list[str], seed: int
) -> tuple[pd.DataFrame, list[dict]]:
    grid = [
        {"max_depth": depth, "n_estimators": estimators, "learning_rate": rate}
        for depth in (1, 2)
        for estimators, rate in ((100, 0.05), (300, 0.03))
    ]
    predictions, selections = [], []
    for outer_fold in range(5):
        inner_fold = (outer_fold + 1) % 5
        train_mask = ~table["development_fold"].isin([outer_fold, inner_fold])
        inner_mask = table["development_fold"] == inner_fold
        outer_mask = table["development_fold"] == outer_fold
        pair_sets = [
            set(table.loc[mask, "pair_id"])
            for mask in (train_mask, inner_mask, outer_mask)
        ]
        if pair_sets[0] & pair_sets[1] or pair_sets[0] & pair_sets[2] or pair_sets[1] & pair_sets[2]:
            raise RuntimeError("pair leakage in boosted geometry folds")
        x_train = table.loc[train_mask, columns].replace([np.inf, -np.inf], np.nan)
        x_inner = table.loc[inner_mask, columns].replace([np.inf, -np.inf], np.nan)
        x_outer = table.loc[outer_mask, columns].replace([np.inf, -np.inf], np.nan)
        y_train = table.loc[train_mask, "label"].to_numpy()
        y_inner = table.loc[inner_mask, "label"].to_numpy()
        best = None
        for parameters in grid:
            model = XGBClassifier(
                **parameters,
                objective="binary:logistic",
                eval_metric="logloss",
                min_child_weight=20,
                subsample=0.8,
                colsample_bytree=1.0,
                reg_lambda=10.0,
                reg_alpha=0.0,
                n_jobs=8,
                random_state=seed + outer_fold,
                tree_method="hist",
            )
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_inner)[:, 1]
            threshold, score = best_f1_threshold(y_inner, probability)
            candidate = (score, -parameters["max_depth"], -parameters["n_estimators"])
            if best is None or candidate > best[0]:
                best = (candidate, model, probability, threshold, parameters)
        _, model, inner_probability, threshold, parameters = best
        baseline_inner_recall = recall_score(
            y_inner,
            table.loc[inner_mask, "baseline_oof_prediction"],
            zero_division=0,
        )
        recall_threshold = matched_recall_threshold(
            y_inner, inner_probability, baseline_inner_recall
        )
        outer_probability = model.predict_proba(x_outer)[:, 1]
        outer = table.loc[
            outer_mask,
            [
                "example_id",
                "context",
                "pair_id",
                "pair_local_key",
                "label",
                "development_fold",
                "baseline_oof_probability",
                "baseline_oof_prediction",
            ],
        ].copy()
        outer["model"] = "baseline_plus_geometry_shallow_xgb"
        outer["probability"] = outer_probability
        outer["threshold"] = threshold
        outer["prediction"] = (outer_probability >= threshold).astype(int)
        outer["matched_recall_threshold"] = recall_threshold
        outer["matched_recall_prediction"] = (
            outer_probability >= recall_threshold
        ).astype(int)
        predictions.append(outer)
        selections.append(
            {
                "model": "baseline_plus_geometry_shallow_xgb",
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                **parameters,
                "threshold": threshold,
                "matched_recall_threshold": recall_threshold,
                "inner_f1": best[0][0],
                "inner_baseline_recall": baseline_inner_recall,
                "feature_count": len(columns),
            }
        )
    return pd.concat(predictions, ignore_index=True), selections


def run(
    site_master_path: Path,
    genomic_context_path: Path,
    context: str,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    master_columns = [
        "example_id",
        "context",
        "pair_id",
        "pair_local_key",
        "label",
        "development_fold",
        "source_split",
        "chromosome",
        "baseline_oof_probability",
        "baseline_oof_logit",
        "baseline_oof_prediction",
    ]
    master = pd.read_parquet(site_master_path, columns=master_columns)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("genomic residual input contains a sealed split")
    table = master[master["context"] == context].copy()
    genomic = pd.read_parquet(genomic_context_path)
    genomic = genomic[genomic["context"] == context].drop(columns="context")
    table = table.merge(
        genomic,
        on=["example_id", "pair_id", "pair_local_key"],
        validate="one_to_one",
    )
    if table[
        ["baseline_oof_probability", "baseline_oof_logit", "baseline_oof_prediction"]
    ].isna().any(axis=None):
        raise RuntimeError(f"incomplete baseline OOF predictions for {context}")
    chromosome = pd.get_dummies(table["chromosome"], prefix="chromosome", dtype=float)
    chromosome.columns = [f"chromosome__{column}" for column in chromosome.columns]
    table = pd.concat([table.reset_index(drop=True), chromosome.reset_index(drop=True)], axis=1)
    sets = feature_sets(table)
    architecture_dimension = len(sets["baseline_plus_architecture"]) - 1

    prediction_tables, selections = [], []
    for model_name, columns in sets.items():
        prediction, selection = fit_cross_fitted(
            table, columns, model_name, seed
        )
        prediction_tables.append(prediction)
        selections.extend(selection)
    boosted, boosted_selection = fit_shallow_geometry(
        table, sets["baseline_plus_geometry"], seed + 41
    )
    prediction_tables.append(boosted)
    selections.extend(boosted_selection)
    random_prediction, random_selection = fit_cross_fitted(
        table,
        [],
        "baseline_plus_matched_random",
        seed + 99,
        random_dimension=architecture_dimension,
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
        bootstrap = bootstrap_differences(
            group, reference, bootstrap_replicates, seed + 1000
        )
        bootstrap["model"] = model_name
        bootstrap_tables.append(bootstrap)
    metric_table = pd.DataFrame(metric_rows)
    bootstrap = pd.concat(bootstrap_tables, ignore_index=True)
    ci = (
        bootstrap.groupby(["model", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    fold_table = fold_differences(predictions, reference)

    destination = output_dir / context
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(destination / "oof_predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(destination / "fold_selections.csv", index=False)
    metric_table.to_csv(destination / "metrics.csv", index=False)
    fold_table.to_csv(destination / "fold_metric_differences.csv", index=False)
    bootstrap.to_parquet(destination / "pair_bootstrap.parquet", index=False)
    ci.to_csv(destination / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "canonical train+valid grouped-OOF predictions only",
        "context": context,
        "rows": len(table),
        "pairs": int(table["pair_id"].nunique()),
        "feature_sets": {name: len(columns) - 1 for name, columns in sets.items()},
        "matched_random_dimension": architecture_dimension,
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
            "run_id": f"genomic_residual_v2_{context}_seed{seed}",
            "experiment_type": "nested_genomic_context_residual",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "genomic_context": sha256_file(genomic_context_path),
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
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument(
        "--genomic-context",
        type=Path,
        default=Path("data/derived/features/genomic_context.parquet"),
    )
    parser.add_argument("--context", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/genomic_residual_v2"),
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
                resolve(args.genomic_context),
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
