"""Descriptive clustered comparison at the development-reference OOF recall.

This audit uses completed out-of-fold probabilities.  It is an operating-curve
comparison, not a deployable threshold-selection procedure: the candidate
threshold is chosen using the evaluated OOF labels, both for the pooled point
estimate and within every pair-bootstrap resample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.models.baseline_oof import metric_row
from adarpro.models.residual import matched_recall_threshold
from adarpro.quarantine import sha256_file
from adarpro.registry import register

METRICS = (
    "f1",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "mcc",
)


def evaluate_at_reference_recall(table: pd.DataFrame) -> tuple[float, dict, dict]:
    labels = table["label"].to_numpy(dtype=int)
    baseline_prediction = table["baseline_oof_prediction"].to_numpy(dtype=int)
    target_recall = float(baseline_prediction[labels == 1].mean())
    threshold = matched_recall_threshold(
        labels, table["probability"].to_numpy(dtype=float), target_recall
    )
    candidate_prediction = (table["probability"].to_numpy() >= threshold).astype(int)
    candidate = metric_row(labels, table["probability"], candidate_prediction)
    baseline = metric_row(
        labels, table["baseline_oof_probability"], baseline_prediction
    )
    return threshold, candidate, baseline


def pair_bootstrap(
    table: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    pairs = table["pair_id"].unique()
    pair_values = table["pair_id"].to_numpy()
    indices = {pair: np.flatnonzero(pair_values == pair) for pair in pairs}
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        subset = table.iloc[np.concatenate([indices[pair] for pair in sampled])]
        threshold, candidate, baseline = evaluate_at_reference_recall(subset)
        for metric in METRICS:
            rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "difference": candidate[metric] - baseline[metric],
                    "candidate": candidate[metric],
                    "development_reference": baseline[metric],
                    "candidate_threshold": threshold,
                }
            )
        rows.append(
            {
                "replicate": replicate,
                "metric": "false_positives",
                "difference": candidate["fp"] - baseline["fp"],
                "candidate": candidate["fp"],
                "development_reference": baseline["fp"],
                "candidate_threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def run(
    repository: Path,
    predictions_path: Path,
    output_dir: Path,
    context: str,
    mode: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    table = pd.read_parquet(predictions_path)
    required = {
        "pair_id",
        "label",
        "probability",
        "baseline_oof_probability",
        "baseline_oof_prediction",
    }
    missing = required - set(table)
    if missing:
        raise RuntimeError(f"missing prediction columns: {sorted(missing)}")
    if "source_split" in table and set(table["source_split"].unique()) - {
        "train",
        "valid",
    }:
        raise RuntimeError("matched-recall curve audit received a sealed split")
    threshold, candidate, baseline = evaluate_at_reference_recall(table)
    bootstrap = pair_bootstrap(table, bootstrap_replicates, seed)
    confidence = (
        bootstrap.groupby("metric")["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(output_dir / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "descriptive pooled grouped-OOF operating-curve comparison",
        "interpretation": (
            "candidate threshold uses evaluated OOF labels and is not a deployable "
            "or model-selection threshold"
        ),
        "context": context,
        "mode": mode,
        "candidate_threshold": threshold,
        "candidate_metrics": candidate,
        "development_reference_metrics": baseline,
        "false_positives_avoided": baseline["fp"] - candidate["fp"],
        "canonical_test_access": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"descriptive_matched_recall_{mode}_{context}_seed{seed}",
            "experiment_type": "descriptive_oof_matched_recall_curve",
            "context": context,
            "input_hashes": {"oof_predictions": sha256_file(predictions_path)},
            "fold_definition": "completed five-fold global-pair OOF predictions",
            "seed": seed,
            "features": f"completed {mode} OOF probability",
            "hyperparameters": {
                "bootstrap_replicates": bootstrap_replicates,
                "threshold_selection": "within evaluated OOF sample/resample",
            },
            "artifact": str(output_dir / "summary.json"),
            "metrics": summary,
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument("--mode", default="binary_natural")
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
                resolve(args.predictions),
                resolve(args.output_dir),
                args.context,
                args.mode,
                args.bootstrap_replicates,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
