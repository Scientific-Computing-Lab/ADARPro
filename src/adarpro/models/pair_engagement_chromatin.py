"""Test exact-tissue H3K36me3 as a substrate-engagement variable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    genomic_path: Path,
    chromatin_path: Path,
    context: str,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    raw = pd.read_parquet(raw_path)
    genomic = pd.read_parquet(genomic_path)
    chromatin = pd.read_parquet(chromatin_path)
    table, intrinsic, _ = prepare_pair_table(raw, genomic, context)
    pair_chromatin = chromatin[chromatin["context"] == context].copy()
    observed = sorted(
        column
        for column in pair_chromatin
        if column.startswith("chromatin__h3k36me3__pair__")
    )
    shifted = sorted(
        column
        for column in pair_chromatin
        if column.startswith("control_shift1mb__h3k36me3__pair__")
    )
    if not observed or pair_chromatin[observed].isna().all().all():
        raise RuntimeError(f"no exact-tissue H3K36me3 pair features for {context}")
    feature_rows = pair_chromatin[["pair_id", *observed, *shifted]]
    uniqueness = feature_rows.groupby("pair_id").nunique(dropna=False)
    if (uniqueness > 1).any().any():
        raise RuntimeError("pair-scale chromatin feature varies within a substrate")
    table = table.merge(
        feature_rows.drop_duplicates("pair_id"), on="pair_id", validate="one_to_one"
    )
    sets = {
        "intrinsic_geometry": intrinsic,
        "h3k36me3_pair_only": observed,
        "intrinsic_geometry_plus_h3k36me3": intrinsic + observed,
        "intrinsic_geometry_plus_shift1mb": intrinsic + shifted,
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
    reference = all_predictions[
        all_predictions["model"] == "intrinsic_geometry"
    ]
    bootstraps = []
    for name, group in all_predictions.groupby("model", sort=False):
        values = bootstrap_differences(
            group, reference, bootstrap_replicates, seed + 3000
        )
        values["model"] = name
        values["reference"] = "intrinsic_geometry"
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
        "scope": "development pairs only; exact-tissue H3K36me3 engagement test",
        "context": context,
        "pairs": len(table),
        "feature_sets": {name: len(columns) for name, columns in sets.items()},
        "reference": "intrinsic_geometry",
        "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"pair_engagement_chromatin_{context}_seed{seed}",
            "experiment_type": "nested_pair_engagement_h3k36me3",
            "context": context,
            "input_hashes": {
                "raw_development_sites": sha256_file(raw_path),
                "genomic_context": sha256_file(genomic_path),
                "chromatin": sha256_file(chromatin_path),
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": summary["feature_sets"],
            "hyperparameters": {
                "model": "ridge",
                "selection": "inner-fold pair MSE",
                "negative_control": "+1 Mb shifted exact-tissue peaks",
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
        "--genomic-features",
        type=Path,
        default=Path("data/derived/features/genomic_context.parquet"),
    )
    parser.add_argument(
        "--chromatin",
        type=Path,
        default=Path("data/derived/features/chromatin_h3k36me3.parquet"),
    )
    parser.add_argument("--context", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/pair_engagement_chromatin"),
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
                resolve(args.genomic_features),
                resolve(args.chromatin),
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
