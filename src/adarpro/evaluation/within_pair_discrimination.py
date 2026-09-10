"""Quantify equal-pair within-duplex discrimination from development OOF scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from adarpro.quarantine import sha256_file
from adarpro.registry import register

CONTEXTS = ("Artery", "Brain", "Liver", "MuscleSkeletal", "Combined")
IDENTITY = ("example_id", "context", "pair_id", "pair_local_key", "label")
MODEL_LABELS = {
    "development_reference": "Internal AdarEdit reference",
    "pair_first": "Expanded pair-first",
    "focal": "Focal representation",
    "target_local": "Target-centered local",
    "fixed_ensemble": "Fixed ensemble",
}


def within_pair_auc(labels: pd.Series, scores: pd.Series) -> float:
    """Return positive-versus-negative score concordance for one mixed pair."""

    if labels.nunique() != 2:
        raise ValueError("within-pair AUROC requires both binary classes")
    return float(roc_auc_score(labels.to_numpy(), scores.to_numpy()))


def align_predictions(paths: dict[str, Path]) -> pd.DataFrame:
    """Load OOF predictions and verify exact row identity across models."""

    if "pair_first" not in paths:
        raise ValueError("pair_first predictions are required")
    tables = {
        model: pd.read_parquet(path).sort_values("example_id").reset_index(drop=True)
        for model, path in paths.items()
    }
    reference = tables["pair_first"]
    required = set(IDENTITY) | {
        "development_fold",
        "probability",
        "baseline_oof_probability",
    }
    missing = required - set(reference)
    if missing:
        raise RuntimeError(f"pair_first predictions lack columns: {sorted(missing)}")
    if set(reference["development_fold"].unique()) != set(range(5)):
        raise RuntimeError("pair_first predictions do not contain five OOF folds")
    aligned = reference[[*IDENTITY, "development_fold"]].copy()
    aligned["development_reference"] = reference["baseline_oof_probability"]
    aligned["pair_first"] = reference["probability"]
    for model, table in tables.items():
        if model == "pair_first":
            continue
        model_missing = (set(IDENTITY) | {"development_fold", "probability"}) - set(
            table
        )
        if model_missing:
            raise RuntimeError(f"{model} predictions lack: {sorted(model_missing)}")
        if not reference[[*IDENTITY, "development_fold"]].equals(
            table[[*IDENTITY, "development_fold"]]
        ):
            raise RuntimeError(f"OOF identity mismatch for {model}")
        aligned[model] = table["probability"]
    return aligned


def per_pair_table(table: pd.DataFrame, model_names: tuple[str, ...]) -> pd.DataFrame:
    """Calculate one AUROC per mixed-label duplex and model."""

    rows: list[dict[str, float | int | str]] = []
    for pair_id, subset in table.groupby("pair_id", sort=True):
        positives = int(subset["label"].sum())
        negatives = int(len(subset) - positives)
        if positives == 0 or negatives == 0:
            continue
        for model in model_names:
            rows.append(
                {
                    "context": str(subset["context"].iloc[0]),
                    "pair_id": int(pair_id),
                    "development_fold": int(subset["development_fold"].iloc[0]),
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "sites": len(subset),
                    "positive_sites": positives,
                    "negative_sites": negatives,
                    "within_pair_auc": within_pair_auc(
                        subset["label"], subset[model]
                    ),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_summary(
    per_pair: pd.DataFrame,
    contrasts: tuple[tuple[str, str], ...],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap equal-pair model means and paired mean contrasts."""

    wide = per_pair.pivot(index="pair_id", columns="model", values="within_pair_auc")
    models = tuple(per_pair["model"].drop_duplicates())
    rng = np.random.default_rng(seed)
    model_draws = {model: np.empty(replicates) for model in models}
    contrast_draws = {contrast: np.empty(replicates) for contrast in contrasts}
    values = wide.loc[:, models].to_numpy(dtype=float)
    model_index = {model: index for index, model in enumerate(models)}
    for replicate in range(replicates):
        sampled = rng.integers(0, len(wide), size=len(wide))
        draw = values[sampled]
        for model, index in model_index.items():
            model_draws[model][replicate] = float(draw[:, index].mean())
        for candidate, reference in contrasts:
            contrast_draws[(candidate, reference)][replicate] = float(
                (draw[:, model_index[candidate]] - draw[:, model_index[reference]]).mean()
            )

    summary_rows = []
    for model in models:
        observed = wide[model]
        lower, median, upper = np.quantile(model_draws[model], [0.025, 0.5, 0.975])
        summary_rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "mixed_pairs": len(observed),
                "mean_within_pair_auc": float(observed.mean()),
                "median_pair_auc": float(observed.median()),
                "lower": float(lower),
                "bootstrap_median": float(median),
                "upper": float(upper),
            }
        )

    contrast_rows = []
    for candidate, reference in contrasts:
        differences = wide[candidate] - wide[reference]
        lower, median, upper = np.quantile(
            contrast_draws[(candidate, reference)], [0.025, 0.5, 0.975]
        )
        contrast_rows.append(
            {
                "candidate": candidate,
                "candidate_label": MODEL_LABELS[candidate],
                "reference": reference,
                "reference_label": MODEL_LABELS[reference],
                "mixed_pairs": len(differences),
                "mean_difference": float(differences.mean()),
                "median_pair_difference": float(differences.median()),
                "pairs_improved": int((differences > 0).sum()),
                "pairs_tied": int((differences == 0).sum()),
                "pairs_worsened": int((differences < 0).sum()),
                "lower": float(lower),
                "bootstrap_median": float(median),
                "upper": float(upper),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def available_paths(repository: Path, context: str) -> dict[str, Path]:
    base = repository / "experiments/runs/richer_supervision"
    candidates = {
        "pair_first": base / "binary_pair_balanced" / context / "oof_predictions.parquet",
        "focal": base
        / "binary_pair_balanced_target_pool"
        / context
        / "oof_predictions.parquet",
        "target_local": base
        / "binary_pair_balanced_local"
        / context
        / "oof_predictions.parquet",
        "fixed_ensemble": base
        / "fixed_pair_local_ensemble"
        / context
        / "oof_predictions.parquet",
    }
    return {model: path for model, path in candidates.items() if path.is_file()}


def run(
    repository: Path,
    output_dir: Path,
    source_dir: Path,
    replicates: int,
    seed: int,
) -> dict:
    all_pairs = []
    all_summary = []
    all_contrasts = []
    input_hashes: dict[str, str] = {}
    for context_index, context in enumerate(CONTEXTS):
        paths = available_paths(repository, context)
        if "pair_first" not in paths:
            raise FileNotFoundError(f"missing pair-first OOF predictions for {context}")
        input_hashes.update(
            {
                f"{context}_{model}": sha256_file(path)
                for model, path in paths.items()
            }
        )
        aligned = align_predictions(paths)
        models = tuple(
            model
            for model in MODEL_LABELS
            if model == "development_reference" or model in paths
        )
        pair_table = per_pair_table(aligned, models)
        if pair_table.empty:
            raise RuntimeError(f"no mixed-label pairs for {context}")
        contrasts = [("pair_first", "development_reference")]
        contrasts.extend(
            (model, "pair_first")
            for model in ("focal", "target_local", "fixed_ensemble")
            if model in models
        )
        summary, contrast = bootstrap_summary(
            pair_table,
            tuple(contrasts),
            replicates,
            seed + 1000 * context_index,
        )
        summary.insert(0, "context", context)
        contrast.insert(0, "context", context)
        all_pairs.append(pair_table)
        all_summary.append(summary)
        all_contrasts.append(contrast)

    per_pair = pd.concat(all_pairs, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    contrasts = pd.concat(all_contrasts, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    per_pair.to_parquet(output_dir / "per_pair_within_duplex_auc.parquet", index=False)
    summary.to_csv(output_dir / "within_duplex_summary.csv", index=False)
    contrasts.to_csv(output_dir / "within_duplex_contrasts.csv", index=False)
    summary.to_csv(source_dir / "within_duplex_discrimination.csv", index=False)
    contrasts.to_csv(source_dir / "within_duplex_contrasts.csv", index=False)

    payload = {
        "scope": "post-hoc descriptive analysis of pair-disjoint development OOF predictions",
        "contexts": list(CONTEXTS),
        "bootstrap_replicates": replicates,
        "seed": seed,
        "canonical_test_access": False,
        "mixed_pairs": {
            context: int(
                per_pair.loc[
                    (per_pair["context"] == context)
                    & (per_pair["model"] == "development_reference"),
                    "pair_id",
                ].nunique()
            )
            for context in CONTEXTS
        },
        "contrasts": contrasts.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": "post_analysis_within_duplex_discrimination_v1",
            "experiment_type": "post_hoc_development_oof_within_pair_discrimination",
            "context": "all human endpoints; local models where retained",
            "input_hashes": input_hashes,
            "fold_definition": "existing five global pair-disjoint development OOF folds",
            "seed": seed,
            "features": "model scores grouped within mixed-label Alu pairs",
            "hyperparameters": {
                "bootstrap_replicates": replicates,
                "pair_weight": "equal",
                "eligibility": "at least one positive and one negative canonical site",
            },
            "artifact": str((output_dir / "summary.json").relative_to(repository)),
            "metrics": payload,
            "canonical_test_access": False,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/post_analysis/within_duplex"),
    )
    parser.add_argument(
        "--source-dir", type=Path, default=Path("paper/source_tables")
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    repository = args.repository.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repository / args.output_dir
    )
    source_dir = (
        args.source_dir
        if args.source_dir.is_absolute()
        else repository / args.source_dir
    )
    print(
        json.dumps(
            run(
                repository,
                output_dir,
                source_dir,
                args.bootstrap_replicates,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
