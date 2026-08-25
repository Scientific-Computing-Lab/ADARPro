"""Generate final-test figures only after the one-use evaluation is consumed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from adarpro.models.baseline_oof import metric_row
from adarpro.models.residual import matched_recall_threshold
from adarpro.quarantine import CONTEXTS, QuarantineViolation, load_policy, sha256_file
from adarpro.registry import register

COLORS = {
    "Artery": "#4C78A8",
    "Brain": "#F58518",
    "Liver": "#54A24B",
    "MuscleSkeletal": "#E45756",
    "Combined": "#B279A2",
}
MATCHED_RECALL_METRICS = (
    "f1",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "mcc",
)


def configure() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(figure: plt.Figure, output_dir: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(figure)


def error_transitions(
    table: pd.DataFrame, context: str
) -> dict[str, float | int | str]:
    label = table["label"].to_numpy()
    baseline = table["baseline_prediction"].to_numpy()
    candidate = table["prediction"].to_numpy()
    baseline_fp = (label == 0) & (baseline == 1)
    baseline_tp = (label == 1) & (baseline == 1)
    rescued = baseline_fp & (candidate == 0)
    damaged = baseline_tp & (candidate == 0)
    return {
        "context": context,
        "baseline_fp": int(baseline_fp.sum()),
        "fp_rescued": int(rescued.sum()),
        "fp_rescue_rate": float(rescued.sum() / max(baseline_fp.sum(), 1)),
        "new_fp": int(((label == 0) & (baseline == 0) & (candidate == 1)).sum()),
        "baseline_tp": int(baseline_tp.sum()),
        "tp_damaged": int(damaged.sum()),
        "tp_damage_rate": float(damaged.sum() / max(baseline_tp.sum(), 1)),
        "fn_recovered": int(((label == 1) & (baseline == 0) & (candidate == 1)).sum()),
    }


def evaluate_at_baseline_recall(table: pd.DataFrame) -> tuple[float, dict, dict]:
    """Descriptively place the candidate at the baseline's observed recall."""

    labels = table["label"].to_numpy(dtype=int)
    baseline_prediction = table["baseline_prediction"].to_numpy(dtype=int)
    target_recall = float(baseline_prediction[labels == 1].mean())
    threshold = matched_recall_threshold(
        labels, table["probability"].to_numpy(dtype=float), target_recall
    )
    candidate_prediction = (table["probability"].to_numpy() >= threshold).astype(int)
    candidate = metric_row(labels, table["probability"], candidate_prediction)
    baseline = metric_row(labels, table["baseline_probability"], baseline_prediction)
    return threshold, candidate, baseline


def matched_recall_bootstrap(
    table: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    """Pair-bootstrap a descriptive, within-resample matched-recall contrast."""

    pairs = table["pair_id"].unique()
    pair_values = table["pair_id"].to_numpy()
    indices = {pair: np.flatnonzero(pair_values == pair) for pair in pairs}
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        subset = table.iloc[np.concatenate([indices[pair] for pair in sampled])]
        threshold, candidate, baseline = evaluate_at_baseline_recall(subset)
        for metric in MATCHED_RECALL_METRICS:
            rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "difference": candidate[metric] - baseline[metric],
                    "candidate_threshold": threshold,
                }
            )
        rows.append(
            {
                "replicate": replicate,
                "metric": "false_positives",
                "difference": candidate["fp"] - baseline["fp"],
                "candidate_threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def run(
    repository: Path,
    policy_path: Path,
    final_dir: Path,
    output_dir: Path,
) -> dict:
    policy = load_policy(policy_path)
    if policy["state"] != "final_consumed":
        raise QuarantineViolation("final figures require a consumed final evaluation")
    summary_path = final_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("canonical_test_access") is not True:
        raise RuntimeError("final-test summary lacks authorized-access provenance")
    configure()
    candidate_name = str(summary["candidate"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_tables"
    source_dir.mkdir(exist_ok=True)
    tables = {
        context: pd.read_parquet(final_dir / f"{context}_predictions.parquet")
        for context in CONTEXTS
    }

    curve_rows = []
    calibration_rows = []
    transition_rows = []
    matched_recall_rows = []
    matched_recall_confidence = []
    for context, table in tables.items():
        transition_rows.append(error_transitions(table, context))
        threshold, candidate_at_recall, baseline_at_recall = (
            evaluate_at_baseline_recall(table)
        )
        matched_recall_rows.append(
            {
                "context": context,
                "candidate_threshold": threshold,
                "target_recall": baseline_at_recall["recall"],
                "candidate_recall": candidate_at_recall["recall"],
                "candidate_precision": candidate_at_recall["precision"],
                "baseline_precision": baseline_at_recall["precision"],
                "candidate_specificity": candidate_at_recall["specificity"],
                "baseline_specificity": baseline_at_recall["specificity"],
                "candidate_f1": candidate_at_recall["f1"],
                "baseline_f1": baseline_at_recall["f1"],
                "candidate_fp": candidate_at_recall["fp"],
                "baseline_fp": baseline_at_recall["fp"],
                "false_positives_avoided": baseline_at_recall["fp"]
                - candidate_at_recall["fp"],
            }
        )
        bootstrap = matched_recall_bootstrap(
            table,
            int(summary["bootstrap_replicates"]),
            20260821 + CONTEXTS.index(context),
        )
        confidence = (
            bootstrap.groupby("metric")["difference"]
            .quantile([0.025, 0.5, 0.975])
            .unstack()
            .reset_index()
            .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
        )
        confidence["context"] = context
        matched_recall_confidence.append(confidence)
        for model, probability_column in (
            ("Published AdarEdit", "baseline_probability"),
            (candidate_name, "probability"),
        ):
            fpr, tpr, _ = roc_curve(table["label"], table[probability_column])
            precision, recall, _ = precision_recall_curve(
                table["label"], table[probability_column]
            )
            curve_rows.extend(
                {"context": context, "model": model, "curve": "ROC", "x": x, "y": y}
                for x, y in zip(fpr, tpr)
            )
            curve_rows.extend(
                {"context": context, "model": model, "curve": "PR", "x": x, "y": y}
                for x, y in zip(recall, precision)
            )
            observed, predicted = calibration_curve(
                table["label"],
                table[probability_column],
                n_bins=10,
                strategy="quantile",
            )
            calibration_rows.extend(
                {
                    "context": context,
                    "model": model,
                    "mean_predicted_probability": predicted_value,
                    "observed_fraction": observed_value,
                }
                for predicted_value, observed_value in zip(predicted, observed)
            )
    curves = pd.DataFrame(curve_rows)
    calibration = pd.DataFrame(calibration_rows)
    transitions = pd.DataFrame(transition_rows)
    matched_recall = pd.DataFrame(matched_recall_rows)
    matched_confidence = pd.concat(matched_recall_confidence, ignore_index=True)
    curves.to_csv(source_dir / "final_roc_pr_curves.csv", index=False)
    calibration.to_csv(source_dir / "final_calibration.csv", index=False)
    transitions.to_csv(source_dir / "final_error_transitions.csv", index=False)
    matched_recall.to_csv(source_dir / "final_matched_recall.csv", index=False)
    matched_confidence.to_csv(
        source_dir / "final_matched_recall_pair_bootstrap_ci.csv", index=False
    )

    for curve_name, axis_labels in (
        ("ROC", ("False-positive rate", "True-positive rate")),
        ("PR", ("Recall", "Precision")),
    ):
        figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), sharex=True, sharey=True)
        for axis, context in zip(axes.flat, CONTEXTS):
            subset = curves[
                (curves["context"] == context) & (curves["curve"] == curve_name)
            ]
            for model, linestyle in (
                ("Published AdarEdit", "--"),
                (candidate_name, "-"),
            ):
                values = subset[subset["model"] == model]
                axis.plot(values["x"], values["y"], linestyle=linestyle, label=model)
            if curve_name == "ROC":
                axis.plot([0, 1], [0, 1], color="#999999", linewidth=0.7, linestyle=":")
            axis.set_title(context.replace("MuscleSkeletal", "Muscle Skeletal"))
            axis.set_xlabel(axis_labels[0])
            axis.set_ylabel(axis_labels[1])
        axes.flat[-1].axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower right", frameon=False)
        figure.suptitle(f"Locked canonical-test {curve_name} curves")
        figure.tight_layout()
        save(figure, output_dir, f"final_{curve_name.lower()}_curves")

    metric_path = final_dir / "metrics.csv"
    differences = []
    for context in CONTEXTS:
        confidence = pd.read_csv(final_dir / f"{context}_pair_bootstrap_ci.csv")
        confidence["context"] = context
        differences.append(confidence)
    confidence = pd.concat(differences, ignore_index=True)
    confidence.to_csv(source_dir / "final_metric_difference_ci.csv", index=False)
    shown = confidence[confidence["metric"].isin(["auroc", "auprc", "f1"])].copy()
    metric_order = {"auroc": 0, "auprc": 1, "f1": 2}
    context_order = {context: index for index, context in enumerate(CONTEXTS)}
    shown["y"] = (
        shown["metric"].map(metric_order)
        + (shown["context"].map(context_order) - 2) * 0.07
    )
    figure, axis = plt.subplots(figsize=(7.6, 4.2))
    for context in CONTEXTS:
        selected = shown[shown["context"] == context].sort_values("y")
        errors = np.vstack(
            [
                selected["median"] - selected["lower"],
                selected["upper"] - selected["median"],
            ]
        )
        axis.errorbar(
            selected["median"],
            selected["y"],
            xerr=errors,
            fmt="o",
            color=COLORS[context],
            capsize=3,
            label=context.replace("MuscleSkeletal", "Muscle Skeletal"),
        )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(range(3), ["AUROC", "AUPRC", "F1"])
    axis.set_xlabel(f"{candidate_name} minus published AdarEdit (95% pair CI)")
    axis.set_title("Locked canonical-test improvement")
    axis.legend(frameon=False, ncol=2)
    save(figure, output_dir, "final_metric_improvement")

    figure, axis = plt.subplots(figsize=(7.2, 4.1))
    positions = np.arange(len(CONTEXTS))
    width = 0.34
    axis.bar(
        positions - width / 2,
        transitions["fp_rescue_rate"],
        width,
        color="#2A9D8F",
        label="FP rescue rate",
    )
    axis.bar(
        positions + width / 2,
        transitions["tp_damage_rate"],
        width,
        color="#E45756",
        label="TP damage rate",
    )
    axis.set_xticks(
        positions,
        [context.replace("MuscleSkeletal", "Muscle") for context in CONTEXTS],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("Fraction of published-baseline calls")
    axis.set_title("False-positive rescue versus true-positive damage")
    axis.legend(frameon=False)
    save(figure, output_dir, "final_fp_rescue_tp_damage")

    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), sharex=True, sharey=True)
    for axis, context in zip(axes.flat, CONTEXTS):
        subset = calibration[calibration["context"] == context]
        for model, linestyle in (("Published AdarEdit", "--"), (candidate_name, "-")):
            values = subset[subset["model"] == model]
            axis.plot(
                values["mean_predicted_probability"],
                values["observed_fraction"],
                marker="o",
                markersize=3,
                linestyle=linestyle,
                label=model,
            )
        axis.plot([0, 1], [0, 1], color="#999999", linewidth=0.7, linestyle=":")
        axis.set_title(context.replace("MuscleSkeletal", "Muscle Skeletal"))
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed fraction")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", frameon=False)
    figure.suptitle("Locked canonical-test calibration")
    figure.tight_layout()
    save(figure, output_dir, "final_calibration")

    result = {
        "scope": "descriptive figures after single consumed canonical-test evaluation",
        "candidate": candidate_name,
        "input_hashes": {
            "summary": sha256_file(summary_path),
            "metrics": sha256_file(metric_path),
            **{
                f"predictions_{context}": sha256_file(
                    final_dir / f"{context}_predictions.parquet"
                )
                for context in CONTEXTS
            },
        },
        "figures": sorted(path.name for path in output_dir.glob("final_*.png")),
        "canonical_test_access": True,
    }
    result_path = output_dir / "final_figure_manifest.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    register(
        repository,
        {
            "run_id": f"final_figures_{candidate_name}",
            "experiment_type": "post_lock_descriptive_figures",
            "context": "all five canonical contexts",
            "input_hashes": result["input_hashes"],
            "fold_definition": "single consumed canonical test; descriptive only",
            "seed": "none",
            "features": "ROC, PR, calibration, paired metric differences and error transitions",
            "hyperparameters": {"calibration_bins": 10},
            "artifact": str(result_path.relative_to(repository)),
            "metrics": result,
            "canonical_test_access": True,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--policy", type=Path, default=Path("provenance/test_quarantine.json")
    )
    parser.add_argument(
        "--final-dir", type=Path, default=Path("results/adarpro/final_locked_test")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures/final"))
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(
        json.dumps(
            run(
                root,
                resolve(args.policy),
                resolve(args.final_dir),
                resolve(args.output_dir),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
