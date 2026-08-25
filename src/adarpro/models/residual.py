"""Leakage-free low-capacity residual models for incremental RBP information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adarpro.quarantine import sha256_file
from adarpro.registry import register

THRESHOLDS = np.linspace(0.1, 0.9, 33)
CS = (0.01, 0.1, 1.0, 10.0)
ECLIP_SUFFIXES = (
    "eclip_focal_overlap_any",
    "eclip_focal_near_w25_any",
    "eclip_partner_overlap_any",
    "eclip_duplex_peak_any",
    "eclip_duplex_coverage_fraction_max",
)
SHIFTED_ECLIP_SUFFIXES = (
    "eclip_shifted_focal_overlap_any",
    "eclip_shifted_focal_near_w25_any",
    "eclip_shifted_partner_overlap_any",
    "eclip_shifted_duplex_peak_any",
    "eclip_shifted_duplex_coverage_fraction_max",
)
PREDICTED_SUFFIXES = (
    "target_focal_value",
    "target_focal_mean_w25",
    "target_focal_mass_w25",
    "target_partner_value",
    "target_partner_mean_w25",
    "target_duplex_max",
    "target_focal_arm_mass",
    "target_opposite_arm_mass",
    "target_arm_mass_asymmetry",
    "mixing_coefficient",
)
INTERACTION_SUFFIXES = tuple(
    f"{name}_x_log1p_tpm"
    for name in PREDICTED_SUFFIXES
    if name not in {"target_focal_mass_w25", "target_arm_mass_asymmetry", "mixing_coefficient"}
)
PRISMNET_SUFFIXES = (
    "prismnet_focal_overlap_any",
    "prismnet_focal_near_w25_any",
    "prismnet_focal_max_score_w25",
    "prismnet_partner_overlap_any",
    "prismnet_duplex_peak_any",
    "prismnet_duplex_coverage_fraction_max",
    "prismnet_duplex_max_score",
)
PRISMNET_INTERACTION_SUFFIXES = tuple(
    f"{name}_x_log1p_tpm" for name in PRISMNET_SUFFIXES
)
SUPPRESSORS = {"ILF3", "ILF2", "STRBP"}
NEGATIVE_CONTROLS = {"RBFOX2", "TIA1", "IGF2BP1"}
ECLIP_GENES = {
    "ADAR",
    "DROSHA",
    "FXR1",
    "HNRNPC",
    "HNRNPM",
    "IGF2BP1",
    "ILF3",
    "MATR3",
    "PUS1",
    "RBFOX2",
    "SLBP",
    "TIA1",
    "TROVE2",
    "XRCC6",
}


def split_wide_name(column: str) -> tuple[str, str, str] | None:
    parts = column.split("__", 2)
    return (
        tuple(parts)
        if len(parts) == 3 and parts[0] in {"rbp", "pred", "prism"}
        else None
    )


def select_columns(matrix: pd.DataFrame) -> dict[str, list[str]]:
    baseline = ["baseline_oof_logit"]
    eclip, shifted_eclip, predicted, interactions, prismnet, prismnet_interactions, expression = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    suppressor, negative = [], []
    for column in matrix.columns:
        parsed = split_wide_name(column)
        if parsed is None:
            continue
        source, gene, feature = parsed
        if source == "rbp" and gene in ECLIP_GENES and feature in ECLIP_SUFFIXES:
            eclip.append(column)
        if source == "rbp" and gene in ECLIP_GENES and feature in SHIFTED_ECLIP_SUFFIXES:
            shifted_eclip.append(column)
        if source == "pred" and feature in PREDICTED_SUFFIXES:
            predicted.append(column)
        if source == "pred" and feature in INTERACTION_SUFFIXES:
            interactions.append(column)
        if source == "prism" and feature in PRISMNET_SUFFIXES:
            prismnet.append(column)
        if source == "prism" and feature in PRISMNET_INTERACTION_SUFFIXES:
            prismnet_interactions.append(column)
        if source == "rbp" and feature in {"log1p_tpm", "tpm_over_adar"}:
            expression.append(column)
        if gene in SUPPRESSORS and (
            (gene in ECLIP_GENES and feature in ECLIP_SUFFIXES)
            or feature in PREDICTED_SUFFIXES
            or feature in INTERACTION_SUFFIXES
            or feature in PRISMNET_SUFFIXES
            or feature in PRISMNET_INTERACTION_SUFFIXES
            or feature in {"log1p_tpm", "tpm_over_adar"}
        ):
            suppressor.append(column)
        if gene in NEGATIVE_CONTROLS and (
            (gene in ECLIP_GENES and feature in ECLIP_SUFFIXES)
            or feature in PREDICTED_SUFFIXES
            or feature in INTERACTION_SUFFIXES
            or feature in PRISMNET_SUFFIXES
            or feature in PRISMNET_INTERACTION_SUFFIXES
        ):
            negative.append(column)
    mappability = [
        column for column in matrix if column.startswith("umap100_")
    ]
    sets = {
        "baseline_logit": baseline,
        "baseline_plus_eclip": baseline + sorted(eclip),
        "baseline_plus_shifted_eclip_control": baseline + sorted(shifted_eclip),
        "baseline_plus_predicted_binding": baseline + sorted(predicted),
        "baseline_plus_prismnet": baseline + sorted(prismnet),
        "baseline_plus_all_predicted_binding": baseline
        + sorted(predicted + prismnet),
        "baseline_plus_expression": baseline + sorted(expression),
        "baseline_plus_binding_expression": baseline
        + sorted(
            predicted
            + prismnet
            + expression
            + interactions
            + prismnet_interactions
        ),
        "baseline_plus_biological_suppressors": baseline + sorted(suppressor),
        "baseline_plus_broad_rbp": baseline
        + sorted(
            eclip
            + predicted
            + prismnet
            + expression
            + interactions
            + prismnet_interactions
        ),
        "baseline_plus_negative_control_rbps": baseline + sorted(negative),
        "baseline_plus_mappability": baseline + sorted(mappability),
        "predicted_binding_without_baseline": sorted(predicted),
    }
    for name, columns in sets.items():
        if len(columns) != len(set(columns)):
            raise RuntimeError(f"duplicate feature in {name}")
    return sets


def make_model(c_value: float, seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        ),
    )


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    scores = [f1_score(labels, probabilities >= value, zero_division=0) for value in THRESHOLDS]
    index = int(np.argmax(scores))
    return float(THRESHOLDS[index]), float(scores[index])


def matched_recall_threshold(
    labels: np.ndarray, probabilities: np.ndarray, target_recall: float
) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    positive_probabilities = probabilities[labels == 1]
    if not len(positive_probabilities):
        return 0.0
    required = int(
        np.ceil(float(target_recall) * len(positive_probabilities) - 1e-12)
    )
    if required <= 0:
        return 1.0
    if required > len(positive_probabilities):
        return 0.0
    # The required-th largest positive probability is exactly the largest
    # observed threshold retaining the requested number of true positives.
    # This order-statistic form is equivalent to scanning all probabilities,
    # but scales as O(n) rather than O(n^2).
    index = len(positive_probabilities) - required
    return float(np.partition(positive_probabilities, index)[index])


def metrics(labels, probabilities, predictions) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    return {
        "n": len(labels),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(recall_score(labels, predictions, pos_label=0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def fit_cross_fitted(
    table: pd.DataFrame,
    columns: list[str],
    model_name: str,
    seed: int,
    random_dimension: int = 0,
) -> tuple[pd.DataFrame, list[dict]]:
    predictions, selections = [], []
    random_values = None
    if random_dimension:
        noise = np.random.default_rng(seed).normal(size=(len(table), random_dimension))
        random_values = np.column_stack(
            [table["baseline_oof_logit"].to_numpy(), noise]
        )
    for outer_fold in range(5):
        inner_fold = (outer_fold + 1) % 5
        train_mask = ~table["development_fold"].isin([outer_fold, inner_fold])
        inner_mask = table["development_fold"] == inner_fold
        outer_mask = table["development_fold"] == outer_fold
        train_pairs = set(table.loc[train_mask, "pair_id"])
        inner_pairs = set(table.loc[inner_mask, "pair_id"])
        outer_pairs = set(table.loc[outer_mask, "pair_id"])
        if train_pairs & inner_pairs or train_pairs & outer_pairs or inner_pairs & outer_pairs:
            raise RuntimeError("pair leakage in residual nested folds")
        if random_values is None:
            x_train = table.loc[train_mask, columns]
            x_inner = table.loc[inner_mask, columns]
            x_outer = table.loc[outer_mask, columns]
        else:
            x_train = random_values[train_mask.to_numpy()]
            x_inner = random_values[inner_mask.to_numpy()]
            x_outer = random_values[outer_mask.to_numpy()]
        y_train = table.loc[train_mask, "label"].to_numpy()
        y_inner = table.loc[inner_mask, "label"].to_numpy()
        best = None
        for c_value in CS:
            model = make_model(c_value, seed + outer_fold)
            model.fit(x_train, y_train)
            inner_probability = model.predict_proba(x_inner)[:, 1]
            threshold, score = best_f1_threshold(y_inner, inner_probability)
            candidate = (score, -c_value, model, inner_probability, threshold)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        _, negative_c, model, inner_probability, threshold = best
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
        outer["model"] = model_name
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
                "model": model_name,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "C": -negative_c,
                "threshold": threshold,
                "matched_recall_threshold": recall_threshold,
                "inner_f1": best[0],
                "inner_baseline_recall": baseline_inner_recall,
                "feature_count": (random_dimension + 1) if random_dimension else len(columns),
            }
        )
    return pd.concat(predictions, ignore_index=True), selections


def rescue_metrics(predictions: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    baseline_fp = (predictions["label"] == 0) & (
        predictions["baseline_oof_prediction"] == 1
    )
    baseline_tp = (predictions["label"] == 1) & (
        predictions["baseline_oof_prediction"] == 1
    )
    return {
        "fp_rescue_rate": float(
            np.mean(predictions.loc[baseline_fp, prediction_column] == 0)
        ),
        "tp_damage_rate": float(
            np.mean(predictions.loc[baseline_tp, prediction_column] == 0)
        ),
    }


def bootstrap_differences(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    reference_columns = reference[["example_id", "probability", "prediction"]].rename(
        columns={"probability": "reference_probability", "prediction": "reference_prediction"}
    )
    predictions = predictions.merge(
        reference_columns, on="example_id", validate="one_to_one"
    )
    pairs = predictions["pair_id"].unique()
    pair_indices = {
        pair: np.flatnonzero(predictions["pair_id"].to_numpy() == pair) for pair in pairs
    }
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        indices = np.concatenate([pair_indices[pair] for pair in sampled])
        subset = predictions.iloc[indices]
        labels = subset["label"].to_numpy()
        baseline = metrics(
            labels,
            subset["reference_probability"],
            subset["reference_prediction"],
        )
        candidate = metrics(labels, subset["probability"], subset["prediction"])
        for metric in (
            "auroc",
            "auprc",
            "f1",
            "precision",
            "recall",
            "specificity",
            "balanced_accuracy",
            "mcc",
            "brier",
        ):
            rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "difference": candidate[metric] - baseline[metric],
                }
            )
    return pd.DataFrame(rows)


def run(
    matrix_path: Path,
    context: str,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    matrix = pd.read_parquet(matrix_path)
    if set(matrix["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("residual pilot matrix contains a sealed split")
    if context == "IndividualPooled":
        table = matrix[matrix["context"] != "Combined"].copy()
    else:
        table = matrix[matrix["context"] == context].copy()
    required = [
        "baseline_oof_probability",
        "baseline_oof_logit",
        "baseline_oof_prediction",
    ]
    if table[required].isna().any(axis=None):
        raise RuntimeError(f"incomplete OOF baseline predictions for {context}")
    table = table.reset_index(drop=True)
    tissue_dummies = pd.get_dummies(table["context"], prefix="tissue", dtype=float)
    table = pd.concat([table, tissue_dummies], axis=1)
    feature_sets = select_columns(table)
    feature_sets["baseline_plus_tissue_id"] = [
        "baseline_oof_logit", *tissue_dummies.columns.tolist()
    ]
    predicted_dimension = len(feature_sets["baseline_plus_all_predicted_binding"]) - 1
    predictions, selections = [], []
    for name, columns in feature_sets.items():
        result, selection = fit_cross_fitted(table, columns, name, seed)
        predictions.append(result)
        selections.extend(selection)
    random_result, random_selection = fit_cross_fitted(
        table,
        [],
        "baseline_plus_matched_random",
        seed + 99,
        random_dimension=predicted_dimension,
    )
    predictions.append(random_result)
    selections.extend(random_selection)
    all_predictions = pd.concat(predictions, ignore_index=True)
    reference = all_predictions[all_predictions["model"] == "baseline_logit"].copy()
    if len(reference) != len(table):
        raise RuntimeError("incomplete cross-fitted baseline-logit reference")

    metric_rows, bootstrap_tables = [], []
    baseline_metrics = metrics(
        table["label"],
        table["baseline_oof_probability"],
        table["baseline_oof_prediction"],
    )
    baseline_metrics.update({"model": "raw_baseline_oof", "threshold_mode": "baseline_inner"})
    metric_rows.append(baseline_metrics)
    for name, group in all_predictions.groupby("model", sort=False):
        row = metrics(group["label"], group["probability"], group["prediction"])
        row.update(rescue_metrics(group, "prediction"))
        row.update({"model": name, "threshold_mode": "inner_f1"})
        metric_rows.append(row)
        matched = metrics(
            group["label"], group["probability"], group["matched_recall_prediction"]
        )
        matched.update(rescue_metrics(group, "matched_recall_prediction"))
        matched.update({"model": name, "threshold_mode": "inner_matched_recall"})
        metric_rows.append(matched)
        boot = bootstrap_differences(
            group, reference, bootstrap_replicates, seed + 1000
        )
        boot["model"] = name
        boot["reference_model"] = "baseline_logit"
        bootstrap_tables.append(boot)
    metric_table = pd.DataFrame(metric_rows)
    bootstrap = pd.concat(bootstrap_tables, ignore_index=True)
    ci = (
        bootstrap.groupby(["model", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    context_dir = output_dir / context
    context_dir.mkdir(parents=True, exist_ok=True)
    all_predictions.to_parquet(context_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(context_dir / "fold_selections.csv", index=False)
    metric_table.to_csv(context_dir / "metrics.csv", index=False)
    bootstrap.to_parquet(context_dir / "pair_bootstrap.parquet", index=False)
    ci.to_csv(context_dir / "pair_bootstrap_ci.csv", index=False)
    result = {
        "scope": "development train+valid only",
        "context": context,
        "rows": len(table),
        "pairs": int(table["pair_id"].nunique()),
        "models": sorted(all_predictions["model"].unique()),
        "bootstrap_replicates": bootstrap_replicates,
        "selection": "nested pair folds; C and threshold selected on inner fold",
        "canonical_test_access": False,
    }
    (context_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    register(
        matrix_path.parents[3],
        {
            "run_id": f"residual_pilot_{context}_seed{seed}",
            "experiment_type": "nested_residual_pilot",
            "context": context,
            "input_hashes": {"residual_matrix": sha256_file(matrix_path)},
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": sorted(feature_sets),
            "hyperparameters": {
                "model": "L2 logistic regression",
                "C_grid": list(CS),
                "threshold_grid": THRESHOLDS.tolist(),
                "bootstrap_replicates": bootstrap_replicates,
            },
            "artifact": str(context_dir / "oof_predictions.parquet"),
            "metrics": metric_table.to_dict(orient="records"),
            "canonical_test_access": False,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("data/derived/features/residual_model_matrix.parquet"),
    )
    parser.add_argument(
        "--context",
        choices=("Artery", "Brain", "Liver", "MuscleSkeletal", "Combined", "IndividualPooled"),
        required=True,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/runs/residual_pilot")
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
                resolve(args.matrix),
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
