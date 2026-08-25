"""Paired global-pair comparison of joint and independent development models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.models.residual import metrics
from adarpro.quarantine import sha256_file
from adarpro.registry import register

CONTEXTS = ("Artery", "Brain", "Liver", "MuscleSkeletal")
IDENTITY = ("example_id", "context", "pair_id", "pair_local_key", "label", "development_fold")
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


def score(table: pd.DataFrame) -> dict[str, float | int]:
    return metrics(table["label"], table["probability"], table["prediction"])


def load_aligned(
    joint_path: Path, independent_paths: dict[str, Path]
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    if set(independent_paths) != set(CONTEXTS):
        raise ValueError("independent paths must cover exactly four individual tissues")
    joint = pd.read_parquet(joint_path)
    if set(joint["context"].unique()) != set(CONTEXTS):
        raise RuntimeError("joint predictions must contain four individual tissues only")
    if set(joint["development_fold"].unique()) != set(range(5)):
        raise RuntimeError("joint predictions lack five development folds")
    result = {}
    for context in CONTEXTS:
        candidate = (
            joint[joint["context"].eq(context)]
            .sort_values("example_id")
            .reset_index(drop=True)
        )
        reference = (
            pd.read_parquet(independent_paths[context])
            .sort_values("example_id")
            .reset_index(drop=True)
        )
        missing = set(IDENTITY) | {"probability", "prediction"}
        for name, table in (("joint", candidate), ("independent", reference)):
            absent = missing - set(table)
            if absent:
                raise RuntimeError(f"{context} {name} lacks columns: {sorted(absent)}")
        if not candidate[list(IDENTITY)].equals(reference[list(IDENTITY)]):
            raise RuntimeError(f"joint/independent identity mismatch: {context}")
        result[context] = (candidate, reference)
    return result


def observed_metrics(
    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for context, (candidate, reference) in tables.items():
        for model, table in (("joint", candidate), ("independent", reference)):
            rows.append({"context": context, "model": model, **score(table)})
    observed = pd.DataFrame(rows)
    for model in ("joint", "independent"):
        subset = observed[observed["model"].eq(model)]
        rows.append(
            {
                "context": "macro",
                "model": model,
                **{metric: float(subset[metric].mean()) for metric in METRICS},
            }
        )
    return pd.DataFrame(rows)


def paired_global_pair_bootstrap(
    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    pairs = np.unique(
        np.concatenate(
            [candidate["pair_id"].unique() for candidate, _ in tables.values()]
        )
    )
    indices = {
        context: {
            pair: np.flatnonzero(candidate["pair_id"].to_numpy() == pair)
            for pair in candidate["pair_id"].unique()
        }
        for context, (candidate, _) in tables.items()
    }
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        context_differences = {metric: [] for metric in METRICS}
        for context, (candidate, reference) in tables.items():
            selected = [indices[context][pair] for pair in sampled if pair in indices[context]]
            if not selected:
                raise RuntimeError(f"bootstrap omitted every {context} pair")
            positions = np.concatenate(selected)
            candidate_score = score(candidate.iloc[positions])
            reference_score = score(reference.iloc[positions])
            for metric in METRICS:
                difference = candidate_score[metric] - reference_score[metric]
                context_differences[metric].append(difference)
                rows.append(
                    {
                        "replicate": replicate,
                        "context": context,
                        "metric": metric,
                        "difference": difference,
                    }
                )
        for metric, values in context_differences.items():
            rows.append(
                {
                    "replicate": replicate,
                    "context": "macro",
                    "metric": metric,
                    "difference": float(np.mean(values)),
                }
            )
    return pd.DataFrame(rows)


def run(
    repository: Path,
    joint_path: Path,
    independent_paths: dict[str, Path],
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    tables = load_aligned(joint_path, independent_paths)
    observed = observed_metrics(tables)
    bootstrap = paired_global_pair_bootstrap(tables, bootstrap_replicates, seed)
    confidence = (
        bootstrap.groupby(["context", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    macro = observed[observed["context"].eq("macro")].set_index("model")
    macro_differences = {
        metric: float(macro.loc["joint", metric] - macro.loc["independent", metric])
        for metric in METRICS
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    observed.to_csv(output_dir / "metrics.csv", index=False)
    bootstrap.to_parquet(output_dir / "global_pair_bootstrap.parquet", index=False)
    confidence.to_csv(output_dir / "global_pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "four individual tissues; canonical train+valid grouped OOF only",
        "comparison": "joint pair-balanced minus independent pair-balanced",
        "contexts": list(CONTEXTS),
        "global_pairs": len(
            set().union(
                *(set(candidate["pair_id"]) for candidate, _ in tables.values())
            )
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "macro_differences": macro_differences,
        "canonical_test_access": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    register(
        repository,
        {
            "run_id": f"joint_vs_independent_pair_balanced_seed{seed}",
            "experiment_type": "joint_vs_independent_global_pair_bootstrap",
            "context": "four individual tissues",
            "input_hashes": {
                "joint_predictions": sha256_file(joint_path),
                **{
                    f"independent_{context}": sha256_file(path)
                    for context, path in independent_paths.items()
                },
            },
            "fold_definition": "identical five global pair OOF folds; global-pair clustered bootstrap",
            "seed": seed,
            "features": "joint masked multi-label pair-balanced versus independent pair-balanced",
            "hyperparameters": {"bootstrap_replicates": bootstrap_replicates},
            "artifact": str(summary_path.relative_to(repository)),
            "metrics": summary,
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument(
        "--independent",
        action="append",
        required=True,
        help="CONTEXT=path/to/oof_predictions.parquet; provide four tissues",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/joint_multilabel/vs_independent"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    independent = {}
    for specification in args.independent:
        context, separator, raw_path = specification.partition("=")
        if not separator or context not in CONTEXTS or context in independent:
            parser.error(f"invalid --independent value: {specification}")
        independent[context] = resolve(Path(raw_path))
    print(
        json.dumps(
            run(
                root,
                resolve(args.joint),
                independent,
                resolve(args.output_dir),
                args.bootstrap_replicates,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
