"""Pair-bootstrap two named models stored in one long-format OOF table."""

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
    candidate_name: str,
    reference_name: str,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    predictions_hash = sha256_file(predictions_path)
    table = pd.read_parquet(predictions_path)
    if candidate_name == reference_name:
        raise ValueError("candidate and reference must differ")
    candidate = table[table["model"] == candidate_name].copy()
    reference = table[table["model"] == reference_name].copy()
    if candidate.empty or reference.empty:
        raise RuntimeError("candidate or reference model is absent")
    identity = ["example_id", "context", "pair_id", "pair_local_key", "label"]
    left = candidate.sort_values("example_id").reset_index(drop=True)
    right = reference.sort_values("example_id").reset_index(drop=True)
    if not left[identity].equals(right[identity]):
        raise RuntimeError("long-format OOF models are misaligned")
    bootstrap = bootstrap_differences(
        candidate, reference, bootstrap_replicates, seed
    )
    confidence = (
        bootstrap.groupby("metric")["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    metric_table = pd.DataFrame(
        [
            {
                "model": candidate_name,
                **metrics(left["label"], left["probability"], left["prediction"]),
            },
            {
                "model": reference_name,
                **metrics(
                    right["label"], right["probability"], right["prediction"]
                ),
            },
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(output_dir / "pair_bootstrap_ci.csv", index=False)
    metric_table.to_csv(output_dir / "metrics.csv", index=False)
    summary = {
        "scope": "paired comparison of aligned grouped-OOF development predictions",
        "candidate": candidate_name,
        "reference": reference_name,
        "rows": len(left),
        "pairs": int(left["pair_id"].nunique()),
        "bootstrap_replicates": bootstrap_replicates,
        "canonical_test_access": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": (
                f"long_oof_pairwise_{candidate_name}_vs_{reference_name}_"
                f"{predictions_hash[:12]}_n{bootstrap_replicates}_seed{seed}"
            ),
            "experiment_type": "paired_long_format_grouped_oof_comparison",
            "context": str(left["context"].iloc[0]),
            "input_hashes": {"predictions": predictions_hash},
            "fold_definition": "identical five global pair OOF folds",
            "seed": seed,
            "features": [candidate_name, reference_name],
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
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
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
                args.candidate,
                args.reference,
                resolve(args.output_dir),
                args.bootstrap_replicates,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
