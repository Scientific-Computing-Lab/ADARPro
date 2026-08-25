"""Evaluate a neural control stopped after prespecified outer-fold futility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adarpro.evaluation.richer_supervision import (
    METRICS,
    paired_bootstrap,
    score,
    validate_alignment,
)
from adarpro.quarantine import sha256_file
from adarpro.registry import register


def run(
    repository: Path,
    control_run_dir: Path,
    candidate_paths: dict[str, Path],
    completed_folds: tuple[int, ...],
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
    stop_reason: str,
) -> dict:
    control_parts, control_hashes = [], {}
    for fold in completed_folds:
        path = control_run_dir / f"outer_{fold}" / "outer_predictions.parquet"
        control_hashes[f"outer_{fold}"] = sha256_file(path)
        table = pd.read_parquet(path)
        if set(table["development_fold"]) != {fold}:
            raise RuntimeError(f"control fold mismatch: {path}")
        control_parts.append(table)
    tables = {
        "control": pd.concat(control_parts, ignore_index=True)
        .sort_values("example_id")
        .reset_index(drop=True)
    }
    for name, path in candidate_paths.items():
        table = pd.read_parquet(path)
        table = table[table["development_fold"].isin(completed_folds)]
        tables[name] = table.sort_values("example_id").reset_index(drop=True)
    validate_alignment(tables)
    first = next(iter(tables.values()))
    metric_rows = [
        {"model": "development_reference", **score(first, "development_reference")}
    ]
    metric_rows.extend(
        {"model": name, **score(table, name)} for name, table in tables.items()
    )
    bootstrap = paired_bootstrap(tables, bootstrap_replicates, seed)
    confidence = (
        bootstrap.groupby(["candidate", "reference", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    fold_rows = []
    for fold in completed_folds:
        subsets = {
            name: table[table["development_fold"] == fold]
            for name, table in tables.items()
        }
        for candidate, candidate_table in subsets.items():
            for reference, reference_table in subsets.items():
                if candidate == reference:
                    continue
                candidate_scores = score(candidate_table, candidate)
                reference_scores = score(reference_table, reference)
                fold_rows.extend(
                    {
                        "development_fold": fold,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "difference": candidate_scores[metric]
                        - reference_scores[metric],
                    }
                    for metric in METRICS
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(output_dir / "pair_bootstrap_ci.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_differences.csv", index=False)
    summary = {
        "scope": "completed development outer folds only",
        "completed_folds": list(completed_folds),
        "rows": len(first),
        "pairs": int(first["pair_id"].nunique()),
        "stop_reason": stop_reason,
        "bootstrap_replicates": bootstrap_replicates,
        "canonical_test_access": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"partial_oof_control_{output_dir.name}_seed{seed}",
            "experiment_type": "stopped_neural_control_grouped_oof",
            "context": str(first["context"].iloc[0]),
            "input_hashes": {
                "control_outer_predictions": control_hashes,
                **{name: sha256_file(path) for name, path in candidate_paths.items()},
            },
            "fold_definition": f"completed outer folds {list(completed_folds)} only",
            "seed": seed,
            "features": list(tables),
            "hyperparameters": {
                "bootstrap_replicates": bootstrap_replicates,
                "stop_reason": stop_reason,
            },
            "artifact": str(
                (output_dir / "pair_bootstrap_ci.csv").relative_to(repository)
            ),
            "metrics": pd.DataFrame(metric_rows).to_dict(orient="records"),
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--control-run-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="NAME=path/to/complete/oof_predictions.parquet",
    )
    parser.add_argument("--completed-folds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--stop-reason", required=True)
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    candidates = {}
    for specification in args.candidate:
        name, separator, raw_path = specification.partition("=")
        if not separator or not name or not raw_path:
            parser.error(f"invalid --candidate value: {specification}")
        candidates[name] = resolve(Path(raw_path))
    print(
        json.dumps(
            run(
                root,
                resolve(args.control_run_dir),
                candidates,
                tuple(args.completed_folds),
                resolve(args.output_dir),
                args.bootstrap_replicates,
                args.seed,
                args.stop_reason,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
