"""Ablate graph-inferable and genomic inputs to substrate engagement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.models.pair_engagement import (
    bootstrap_differences,
    fit_cross_fitted,
    prepare_pair_table,
    score,
)
from adarpro.quarantine import sha256_file
from adarpro.registry import register


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
    table, _, _ = prepare_pair_table(raw, features, context)
    sequences = (
        raw[raw["context"] == context]
        .drop_duplicates("pair_id")
        .set_index("pair_id")["sequence"]
    )
    table["a_count_log1p"] = table["pair_id"].map(
        sequences.map(lambda value: np.log1p(value.upper().count("A")))
    )
    graph_intrinsic = sorted(c for c in table if c.startswith("intrinsic__"))
    genomic_geometry = sorted(c for c in table if c.startswith("geometry__"))
    sets = {
        "adenosine_count_only": ["a_count_log1p"],
        "graph_intrinsic": graph_intrinsic,
        "genomic_geometry": genomic_geometry,
        "combined_16": genomic_geometry + graph_intrinsic,
    }
    predictions, selections = [], []
    for name, columns in sets.items():
        prediction, selection = fit_cross_fitted(
            table,
            columns,
            name,
            "ridge",
            (0.1, 1.0, 10.0, 100.0),
            seed,
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
    reference = all_predictions[all_predictions["model"] == "graph_intrinsic"]
    bootstraps = []
    for name, group in all_predictions.groupby("model", sort=False):
        values = bootstrap_differences(
            group, reference, bootstrap_replicates, seed + 2000
        )
        values["model"] = name
        values["reference"] = "graph_intrinsic"
        bootstraps.append(values)
    bootstrap = pd.concat(bootstraps, ignore_index=True)
    confidence = (
        bootstrap.groupby(["model", "reference", "metric"])["difference"]
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
        "scope": "development pairs only; nested outer-pair ablation",
        "context": context,
        "pairs": len(table),
        "feature_sets": {name: len(columns) for name, columns in sets.items()},
        "reference": "graph_intrinsic",
        "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"pair_engagement_ablation_{context}_seed{seed}",
            "experiment_type": "nested_pair_engagement_information_ablation",
            "context": context,
            "input_hashes": {
                "raw_development_sites": sha256_file(raw_path),
                "genomic_context": sha256_file(feature_path),
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": summary["feature_sets"],
            "hyperparameters": {
                "model": "ridge",
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
    parser.add_argument("--context", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/pair_engagement_ablation"),
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
