"""Pair-disjoint pilot for a substrate-level editing-engagement target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adarpro.quarantine import sha256_file
from adarpro.registry import register


def prepare_pair_table(
    raw: pd.DataFrame, features: pd.DataFrame, context: str
) -> tuple[pd.DataFrame, list[str], list[str]]:
    raw = raw[raw["context"] == context].copy()
    raw["high"] = raw["editing_index"] >= 15
    unambiguous = raw[(raw["editing_index"] < 1) | raw["high"]]
    outcomes = (
        unambiguous.groupby(["pair_id", "development_fold"], as_index=False)
        .agg(
            high_low_site_count=("high", "size"),
            high_site_count=("high", "sum"),
            edited_fraction=("high", "mean"),
        )
    )
    outcomes["any_edited"] = outcomes["high_site_count"].gt(0).astype(int)
    continuous = (
        raw.groupby("pair_id", as_index=False)["editing_index"]
        .mean()
        .rename(columns={"editing_index": "mean_editing_index"})
    )
    outcomes = outcomes.merge(continuous, on="pair_id", validate="one_to_one")
    feature_prefixes = ("geometry__", "intrinsic__", "repeat__", "transcript__")
    feature_columns = [c for c in features if c.startswith(feature_prefixes)]
    feature_rows = features[features["context"] == context][
        ["pair_id", *feature_columns]
    ]
    uniqueness = feature_rows.groupby("pair_id")[feature_columns].nunique(dropna=False)
    if (uniqueness > 1).any().any():
        raise RuntimeError("pair-level feature varies among sites on the same substrate")
    feature_rows = feature_rows.drop_duplicates("pair_id")
    table = outcomes.merge(feature_rows, on="pair_id", validate="one_to_one")
    intrinsic = [c for c in feature_columns if c.startswith(("geometry__", "intrinsic__"))]
    return table, intrinsic, feature_columns


def make_model(kind: str, parameter: float, seed: int):
    if kind == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=parameter),
        )
    if kind == "hist_gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=7,
                min_samples_leaf=20,
                l2_regularization=parameter,
                random_state=seed,
            ),
        )
    raise ValueError(kind)


def score(table: pd.DataFrame, probability_column: str) -> dict[str, float | int]:
    observed = table["edited_fraction"].to_numpy()
    predicted = np.clip(table[probability_column].to_numpy(), 0, 1)
    any_edited = table["any_edited"].to_numpy()
    correlation = spearmanr(observed, predicted).statistic
    return {
        "pairs": len(table),
        "r2": float(r2_score(observed, predicted)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "spearman": float(correlation),
        "any_edited_auroc": float(roc_auc_score(any_edited, predicted)),
        "any_edited_auprc": float(average_precision_score(any_edited, predicted)),
    }


def fit_cross_fitted(
    table: pd.DataFrame,
    columns: list[str],
    model_name: str,
    kind: str,
    parameters: tuple[float, ...],
    seed: int,
    shuffled_target: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    outputs, selections = [], []
    for outer_fold in range(5):
        inner_fold = (outer_fold + 1) % 5
        train = table[~table["development_fold"].isin([outer_fold, inner_fold])]
        inner = table[table["development_fold"] == inner_fold]
        outer = table[table["development_fold"] == outer_fold].copy()
        if set(train["pair_id"]) & set(inner["pair_id"]):
            raise RuntimeError("pair leakage between engagement train and inner folds")
        if (set(train["pair_id"]) | set(inner["pair_id"])) & set(outer["pair_id"]):
            raise RuntimeError("pair leakage into engagement outer fold")
        y_train = train["edited_fraction"].to_numpy().copy()
        if shuffled_target:
            np.random.default_rng(seed + outer_fold).shuffle(y_train)
        best = None
        for parameter in parameters:
            model = make_model(kind, parameter, seed + outer_fold)
            model.fit(train[columns], y_train)
            prediction = np.clip(model.predict(inner[columns]), 0, 1)
            loss = mean_squared_error(inner["edited_fraction"], prediction)
            candidate = (-loss, -parameter, model)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        model = best[2]
        outer["prediction"] = np.clip(model.predict(outer[columns]), 0, 1)
        outer["model"] = model_name
        outputs.append(outer)
        selections.append(
            {
                "model": model_name,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "parameter": -best[1],
                "inner_mse": -best[0],
                "feature_count": len(columns),
            }
        )
    return pd.concat(outputs, ignore_index=True), selections


def bootstrap_differences(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    reference = reference[["pair_id", "prediction"]].rename(
        columns={"prediction": "reference_prediction"}
    )
    table = candidate.merge(reference, on="pair_id", validate="one_to_one")
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        indices = rng.integers(0, len(table), size=len(table))
        subset = table.iloc[indices]
        left = score(subset, "prediction")
        right = score(subset, "reference_prediction")
        for metric in ("r2", "mae", "rmse", "spearman", "any_edited_auroc", "any_edited_auprc"):
            rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "difference": left[metric] - right[metric],
                }
            )
    return pd.DataFrame(rows)


def run(
    repository: Path,
    raw_path: Path,
    feature_path: Path,
    context: str,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    raw = pd.read_parquet(raw_path)
    features = pd.read_parquet(feature_path)
    table, intrinsic, full = prepare_pair_table(raw, features, context)
    if set(table["development_fold"].unique()) != set(range(5)):
        raise RuntimeError("pair engagement table lacks all development folds")
    predictions, selections = [], []
    # A fold-specific mean is a no-information reference with no outer-label use.
    null_outputs = []
    for outer_fold in range(5):
        inner_fold = (outer_fold + 1) % 5
        train = table[~table["development_fold"].isin([outer_fold, inner_fold])]
        outer = table[table["development_fold"] == outer_fold].copy()
        outer["prediction"] = train["edited_fraction"].mean()
        outer["model"] = "training_mean"
        null_outputs.append(outer)
    predictions.append(pd.concat(null_outputs, ignore_index=True))
    specifications = (
        ("ridge_intrinsic", intrinsic, "ridge", (0.1, 1.0, 10.0, 100.0), False),
        ("ridge_full", full, "ridge", (0.1, 1.0, 10.0, 100.0), False),
        (
            "shallow_hgb_full",
            full,
            "hist_gradient_boosting",
            (0.1, 1.0, 10.0),
            False,
        ),
        (
            "shuffled_target_control",
            full,
            "ridge",
            (0.1, 1.0, 10.0, 100.0),
            True,
        ),
    )
    for name, columns, kind, parameters, shuffled in specifications:
        prediction, selection = fit_cross_fitted(
            table, columns, name, kind, parameters, seed, shuffled
        )
        predictions.append(prediction)
        selections.extend(selection)
    all_predictions = pd.concat(predictions, ignore_index=True)
    metric_table = pd.DataFrame(
        [
            {"model": name, **score(group, "prediction")}
            for name, group in all_predictions.groupby("model", sort=False)
        ]
    )
    reference = all_predictions[all_predictions["model"] == "training_mean"]
    bootstrap_tables = []
    for name, group in all_predictions.groupby("model", sort=False):
        values = bootstrap_differences(
            group, reference, bootstrap_replicates, seed + 1000
        )
        values["model"] = name
        bootstrap_tables.append(values)
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
    all_predictions.to_parquet(destination / "oof_pair_predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(destination / "fold_selections.csv", index=False)
    metric_table.to_csv(destination / "metrics.csv", index=False)
    bootstrap.to_parquet(destination / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(destination / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "development pairs only; outer pair folds",
        "context": context,
        "pairs": len(table),
        "edited_fraction_quantiles": {
            str(q): float(table["edited_fraction"].quantile(q))
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
        "feature_counts": {"intrinsic": len(intrinsic), "full": len(full)},
        "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"pair_engagement_{context}_seed{seed}",
            "experiment_type": "nested_pair_engagement_pilot",
            "context": context,
            "input_hashes": {
                "raw_development_sites": sha256_file(raw_path),
                "genomic_context": sha256_file(feature_path),
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": summary["feature_counts"],
            "hyperparameters": {
                "target": "fraction high among EditingIndex <1 or >=15 sites",
                "selection": "inner-fold pair MSE",
                "bootstrap_replicates": bootstrap_replicates,
            },
            "artifact": str(destination / "oof_pair_predictions.parquet"),
            "metrics": metric_table.to_dict(orient="records"),
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--raw-sites",
        type=Path,
        default=Path(
            "data/derived/cross_tissue/development_raw_individual_tissue_sites.parquet"
        ),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/derived/features/genomic_context.parquet"),
    )
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/runs/pair_engagement")
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
                root,
                resolve(args.raw_sites),
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
