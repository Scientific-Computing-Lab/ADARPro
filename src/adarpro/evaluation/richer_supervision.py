"""Paired comparison of richer-supervision grouped-OOF candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.models.residual import metrics
from adarpro.quarantine import sha256_file
from adarpro.registry import register

IDENTITY = [
    "example_id",
    "context",
    "pair_id",
    "pair_local_key",
    "label",
    "development_fold",
]
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


def load_candidate(name: str, path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path).sort_values("example_id").reset_index(drop=True)
    required = set(IDENTITY) | {
        "probability",
        "prediction",
        "baseline_oof_probability",
        "baseline_oof_prediction",
    }
    missing = required - set(table)
    if missing:
        raise RuntimeError(f"{name} lacks required columns: {sorted(missing)}")
    if table["example_id"].duplicated().any():
        raise RuntimeError(f"{name} contains duplicate examples")
    table["candidate"] = name
    return table


def validate_alignment(tables: dict[str, pd.DataFrame]) -> None:
    names = list(tables)
    reference = tables[names[0]]
    for name in names[1:]:
        candidate = tables[name]
        if not reference[IDENTITY].equals(candidate[IDENTITY]):
            raise RuntimeError(f"candidate identity mismatch: {name}")
        for column in ("baseline_oof_probability", "baseline_oof_prediction"):
            if not np.allclose(
                reference[column], candidate[column], rtol=0, atol=1e-12
            ):
                raise RuntimeError(
                    f"development reference mismatch in {name}: {column}"
                )


def score(table: pd.DataFrame, source: str) -> dict[str, float | int]:
    if source == "development_reference":
        return metrics(
            table["label"],
            table["baseline_oof_probability"],
            table["baseline_oof_prediction"],
        )
    return metrics(table["label"], table["probability"], table["prediction"])


def paired_bootstrap(
    tables: dict[str, pd.DataFrame], replicates: int, seed: int
) -> pd.DataFrame:
    first = next(iter(tables.values()))
    pairs = first["pair_id"].unique()
    pair_values = first["pair_id"].to_numpy()
    indices = {pair: np.flatnonzero(pair_values == pair) for pair in pairs}
    rng = np.random.default_rng(seed)
    comparisons = [(name, "development_reference") for name in tables]
    comparisons.extend(combinations(tables, 2))
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        selected = np.concatenate([indices[pair] for pair in sampled])
        subsets = {name: table.iloc[selected] for name, table in tables.items()}
        reference_scores = score(first.iloc[selected], "development_reference")
        candidate_scores = {name: score(table, name) for name, table in subsets.items()}
        for candidate, reference in comparisons:
            left = candidate_scores[candidate]
            right = (
                reference_scores
                if reference == "development_reference"
                else candidate_scores[reference]
            )
            for metric in METRICS:
                rows.append(
                    {
                        "replicate": replicate,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "difference": left[metric] - right[metric],
                    }
                )
    return pd.DataFrame(rows)


def run(
    repository: Path,
    candidate_paths: dict[str, Path],
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
    scope: str = "canonical train+valid grouped OOF comparison only",
) -> dict:
    if len(candidate_paths) < 2:
        raise ValueError("at least two candidates are required")
    tables = {
        name: load_candidate(name, path) for name, path in candidate_paths.items()
    }
    validate_alignment(tables)
    first = next(iter(tables.values()))
    metric_rows = [
        {"model": "development_reference", **score(first, "development_reference")}
    ]
    for name, table in tables.items():
        metric_rows.append({"model": name, **score(table, name)})
    metric_table = pd.DataFrame(metric_rows)
    bootstrap = paired_bootstrap(tables, bootstrap_replicates, seed)
    confidence = (
        bootstrap.groupby(["candidate", "reference", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    fold_rows = []
    for fold in range(5):
        mask = first["development_fold"] == fold
        fold_reference = score(first.loc[mask], "development_reference")
        for name, table in tables.items():
            candidate_score = score(table.loc[mask], name)
            fold_rows.extend(
                {
                    "development_fold": fold,
                    "candidate": name,
                    "reference": "development_reference",
                    "metric": metric,
                    "difference": candidate_score[metric] - fold_reference[metric],
                }
                for metric in METRICS
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_table.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_differences.csv", index=False)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(output_dir / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": scope,
        "candidates": list(candidate_paths),
        "rows": len(first),
        "pairs": int(first["pair_id"].nunique()),
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
                "richer_supervision_comparison_"
                + hashlib.sha256(
                    str(output_dir.resolve()).encode()
                ).hexdigest()[:12]
                + f"_seed{seed}"
            ),
            "experiment_type": "paired_richer_supervision_comparison",
            "context": str(first["context"].iloc[0]),
            "input_hashes": {
                name: sha256_file(path) for name, path in candidate_paths.items()
            },
            "fold_definition": "identical five global pair OOF folds",
            "seed": seed,
            "features": "paired candidate predictions and shared development reference",
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
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="NAME=path/to/oof_predictions.parquet; specify at least twice",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/richer_supervision/comparison"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--scope",
        default="canonical train+valid grouped OOF comparison only",
        help="Human-readable scope stored with the comparison provenance.",
    )
    args = parser.parse_args()
    root = args.repository.resolve()
    paths = {}
    for specification in args.candidate:
        name, separator, raw_path = specification.partition("=")
        if not separator or not name or not raw_path:
            parser.error(f"invalid --candidate value: {specification}")
        path = Path(raw_path)
        paths[name] = path if path.is_absolute() else root / path
    output = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    print(
        json.dumps(
            run(
                root,
                paths,
                output,
                args.bootstrap_replicates,
                args.seed,
                args.scope,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
