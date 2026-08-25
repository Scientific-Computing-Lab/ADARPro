"""Paired control comparisons for a completed hierarchical-gate OOF run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adarpro.models.residual import bootstrap_differences, metrics
from adarpro.quarantine import sha256_file
from adarpro.registry import register


def run(
    repository: Path,
    predictions_path: Path,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    table = pd.read_parquet(predictions_path)
    candidate_name = "expanded_plus_combined_engagement"
    candidate = table[table["model"] == candidate_name].copy()
    references = [
        "expanded_reference",
        "expanded_recalibrated",
        "expanded_plus_graph_engagement",
        "expanded_plus_shuffled_engagement",
    ]
    if candidate.empty or any(table[table["model"] == name].empty for name in references):
        raise RuntimeError("hierarchical gate prediction table is incomplete")
    rows, bootstraps = [], []
    candidate_metrics = metrics(
        candidate["label"], candidate["probability"], candidate["prediction"]
    )
    for reference_name in references:
        reference = table[table["model"] == reference_name].copy()
        reference_metrics = metrics(
            reference["label"], reference["probability"], reference["prediction"]
        )
        for metric in candidate_metrics:
            if metric == "n":
                continue
            rows.append(
                {
                    "candidate": candidate_name,
                    "reference": reference_name,
                    "metric": metric,
                    "candidate_value": candidate_metrics[metric],
                    "reference_value": reference_metrics[metric],
                    "difference": candidate_metrics[metric] - reference_metrics[metric],
                }
            )
        bootstrap = bootstrap_differences(
            candidate, reference, bootstrap_replicates, seed
        )
        bootstrap["candidate"] = candidate_name
        bootstrap["reference"] = reference_name
        bootstraps.append(bootstrap)
    point = pd.DataFrame(rows)
    bootstrap = pd.concat(bootstraps, ignore_index=True)
    confidence = (
        bootstrap.groupby(["candidate", "reference", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    point.to_csv(output_dir / "metric_differences.csv", index=False)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(output_dir / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "completed grouped-OOF hierarchical-gate control contrasts",
        "candidate": candidate_name,
        "references": references,
        "rows": len(candidate),
        "pairs": int(candidate["pair_id"].nunique()),
        "canonical_test_access": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"hierarchical_gate_controls_seed{seed}",
            "experiment_type": "paired_hierarchical_gate_controls",
            "context": str(candidate["context"].iloc[0]),
            "input_hashes": {"oof_predictions": sha256_file(predictions_path)},
            "fold_definition": "identical five global pair OOF folds",
            "seed": seed,
            "features": "paired candidate/control probabilities",
            "hyperparameters": {"bootstrap_replicates": bootstrap_replicates},
            "artifact": str(output_dir / "pair_bootstrap_ci.csv"),
            "metrics": point.to_dict(orient="records"),
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
