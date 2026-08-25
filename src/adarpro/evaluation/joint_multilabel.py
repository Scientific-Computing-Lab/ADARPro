"""Clustered evaluation of four-tissue joint multi-label OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.models.joint_expanded_oof import TISSUES
from adarpro.models.residual import metrics
from adarpro.quarantine import sha256_file
from adarpro.registry import register

METRICS = (
    "auroc",
    "auprc",
    "f1",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "brier",
)


def score(table: pd.DataFrame, candidate: bool) -> dict:
    prefix = "" if candidate else "baseline_oof_"
    return metrics(
        table["label"], table[f"{prefix}probability"], table[f"{prefix}prediction"]
    )


def bootstrap_joint(
    predictions: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    """Resample global pairs once, retaining cross-tissue dependence."""

    pairs = predictions["pair_id"].unique()
    pair_values = predictions["pair_id"].to_numpy()
    indices = {pair: np.flatnonzero(pair_values == pair) for pair in pairs}
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        subset = predictions.iloc[np.concatenate([indices[pair] for pair in sampled])]
        tissue_differences = {metric: [] for metric in METRICS}
        for context in TISSUES:
            tissue = subset[subset["context"] == context]
            candidate = score(tissue, True)
            reference = score(tissue, False)
            for metric in METRICS:
                difference = candidate[metric] - reference[metric]
                tissue_differences[metric].append(difference)
                rows.append(
                    {
                        "replicate": replicate,
                        "context": context,
                        "metric": metric,
                        "difference": difference,
                    }
                )
        for metric in METRICS:
            rows.append(
                {
                    "replicate": replicate,
                    "context": "macro",
                    "metric": metric,
                    "difference": float(np.mean(tissue_differences[metric])),
                }
            )
    return pd.DataFrame(rows)


def run(
    repository: Path,
    predictions_path: Path,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    table = pd.read_parquet(predictions_path)
    if set(table["context"].unique()) != set(TISSUES):
        raise RuntimeError("joint OOF table does not contain exactly four tissues")
    if table.duplicated("example_id").any():
        raise RuntimeError("joint OOF example identifiers are not unique")
    metric_rows = []
    for context in TISSUES:
        tissue = table[table["context"] == context]
        metric_rows.extend(
            [
                {"context": context, "model": "candidate", **score(tissue, True)},
                {
                    "context": context,
                    "model": "development_reference",
                    **score(tissue, False),
                },
            ]
        )
    for model in ("candidate", "development_reference"):
        rows = [row for row in metric_rows if row["model"] == model]
        metric_rows.append(
            {
                "context": "macro",
                "model": model,
                **{
                    metric: float(np.mean([row[metric] for row in rows]))
                    for metric in METRICS
                },
            }
        )
    metric_table = pd.DataFrame(metric_rows)
    bootstrap = bootstrap_joint(table, bootstrap_replicates, seed)
    confidence = (
        bootstrap.groupby(["context", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    fold_rows = []
    for fold in range(5):
        for context in TISSUES:
            tissue = table[
                (table["development_fold"] == fold) & (table["context"] == context)
            ]
            candidate = score(tissue, True)
            reference = score(tissue, False)
            fold_rows.extend(
                {
                    "development_fold": fold,
                    "context": context,
                    "metric": metric,
                    "difference": candidate[metric] - reference[metric],
                }
                for metric in METRICS
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_table.to_csv(output_dir / "metrics.csv", index=False)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(output_dir / "pair_bootstrap_ci.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_differences.csv", index=False)
    summary = {
        "scope": "four-tissue joint grouped OOF; global pair-clustered bootstrap",
        "rows": len(table),
        "pairs": int(table["pair_id"].nunique()),
        "bootstrap_replicates": bootstrap_replicates,
        "canonical_test_access": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"joint_multilabel_evaluation_n{bootstrap_replicates}_seed{seed}",
            "experiment_type": "joint_multilabel_grouped_oof_evaluation",
            "context": "four_individual_tissues",
            "input_hashes": {"predictions": sha256_file(predictions_path)},
            "fold_definition": "identical five global pair folds",
            "seed": seed,
            "features": "paired candidate and development-reference predictions",
            "hyperparameters": {"bootstrap_replicates": bootstrap_replicates},
            "artifact": str(
                (output_dir / "pair_bootstrap_ci.csv").relative_to(repository)
            ),
            "metrics": metric_table.to_dict(orient="records"),
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
                args.bootstrap_replicates,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
