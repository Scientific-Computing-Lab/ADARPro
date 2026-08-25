"""Apply a completed fixed OOF ensemble to natural-prevalence development rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adarpro.models.baseline_oof import metric_row
from adarpro.models.fixed_checkpoint_ensemble import equal_logit_average_many
from adarpro.models.multitask_oof import bootstrap_delta
from adarpro.quarantine import sha256_file
from adarpro.registry import register

IDENTITY = ["example_id", "context", "pair_id", "pair_local_key", "development_fold"]


def run_components(
    repository: Path,
    component_paths: dict[str, Path],
    selections_path: Path,
    output_dir: Path,
    context: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    if len(component_paths) < 2:
        raise ValueError("an ensemble requires at least two components")
    tables = {
        name: pd.read_parquet(path).sort_values("example_id").reset_index(drop=True)
        for name, path in component_paths.items()
    }
    first = next(iter(tables.values()))
    for name, table in list(tables.items())[1:]:
        if not first[IDENTITY + ["label"]].equals(table[IDENTITY + ["label"]]):
            raise RuntimeError(
                f"natural-prevalence ensemble component is misaligned: {name}"
            )
        for column in (
            "baseline_oof_probability",
            "baseline_oof_threshold",
            "baseline_oof_prediction",
        ):
            if not first[column].equals(table[column]):
                raise RuntimeError(
                    f"natural-prevalence reference mismatch in {name}: {column}"
                )
    selections = pd.read_csv(selections_path)
    if set(selections["outer_fold"]) != set(range(5)):
        raise RuntimeError("ensemble threshold table does not contain all outer folds")
    threshold_by_fold = selections.set_index("outer_fold")["threshold"]
    predictions = first[IDENTITY + ["label", "editing_index"]].copy()
    predictions["probability"] = equal_logit_average_many(
        [table["probability"].to_numpy() for table in tables.values()]
    )
    predictions["threshold"] = predictions["development_fold"].map(threshold_by_fold)
    if predictions["threshold"].isna().any():
        raise RuntimeError("missing ensemble threshold for a development fold")
    predictions["prediction"] = (
        predictions["probability"] >= predictions["threshold"]
    ).astype(int)
    for name, table in tables.items():
        predictions[f"{name}_probability"] = table["probability"]
    predictions["baseline_oof_probability"] = first["baseline_oof_probability"]
    predictions["baseline_oof_threshold"] = first["baseline_oof_threshold"]
    predictions["baseline_oof_prediction"] = first["baseline_oof_prediction"]
    candidate_metrics = metric_row(
        predictions["label"], predictions["probability"], predictions["prediction"]
    )
    reference_metrics = metric_row(
        predictions["label"],
        predictions["baseline_oof_probability"],
        predictions["baseline_oof_prediction"],
    )
    bootstrap = bootstrap_delta(predictions, bootstrap_replicates, seed)
    confidence = (
        bootstrap.groupby("metric")["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "oof_predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(output_dir / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "all development high/low sites at natural prevalence; held-out global pair folds",
        "context": context,
        "ensemble": (
            f"fixed equal-logit average of {len(tables)} components; "
            "canonical inner-fold thresholds"
        ),
        "components": list(tables),
        "rows": len(predictions),
        "pairs": int(predictions["pair_id"].nunique()),
        "prevalence": float(predictions["label"].mean()),
        "candidate_metrics": candidate_metrics,
        "development_reference_metrics": reference_metrics,
        "canonical_test_access": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": (
                "fixed_ensemble_natural_prevalence_"
                + "_".join(tables)
                + f"_{context}_seed{seed}"
            ),
            "experiment_type": "fixed_ensemble_natural_prevalence_oof",
            "context": context,
            "input_hashes": {
                **{
                    f"{name}_predictions": sha256_file(path)
                    for name, path in component_paths.items()
                },
                "fold_selections": sha256_file(selections_path),
            },
            "fold_definition": "five global pair folds; thresholds selected on canonical inner folds",
            "seed": seed,
            "features": "fixed equal-logit ensemble probabilities",
            "hyperparameters": {
                "weights": [1 / len(tables)] * len(tables),
                "bootstrap_replicates": bootstrap_replicates,
            },
            "artifact": str(prediction_path.relative_to(repository)),
            "metrics": summary,
            "canonical_test_access": False,
        },
    )
    return summary


def run(
    repository: Path,
    left_path: Path,
    right_path: Path,
    selections_path: Path,
    output_dir: Path,
    context: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    """Backward-compatible two-component natural-prevalence evaluation."""

    return run_components(
        repository,
        {"left": left_path, "right": right_path},
        selections_path,
        output_dir,
        context,
        bootstrap_replicates,
        seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--left-predictions", type=Path)
    parser.add_argument("--right-predictions", type=Path)
    parser.add_argument(
        "--component",
        action="append",
        help="NAME=path/to/natural_predictions.parquet; repeat for an ensemble",
    )
    parser.add_argument("--fold-selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    if args.component:
        components = {}
        for specification in args.component:
            name, separator, raw_path = specification.partition("=")
            if not separator or not name or not raw_path:
                parser.error(f"invalid --component value: {specification}")
            if name in components:
                parser.error(f"duplicate component name: {name}")
            components[name] = resolve(Path(raw_path))
        result = run_components(
            root,
            components,
            resolve(args.fold_selections),
            resolve(args.output_dir),
            args.context,
            args.bootstrap_replicates,
            args.seed,
        )
    else:
        if args.left_predictions is None or args.right_predictions is None:
            parser.error(
                "--left-predictions and --right-predictions are required without "
                "--component"
            )
        result = run(
            root,
            resolve(args.left_predictions),
            resolve(args.right_predictions),
            resolve(args.fold_selections),
            resolve(args.output_dir),
            args.context,
            args.bootstrap_replicates,
            args.seed,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
