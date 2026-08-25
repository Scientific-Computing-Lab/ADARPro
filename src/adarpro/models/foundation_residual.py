"""Outer-isolated residual test of frozen RiNALMo representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from adarpro.models.baseline_oof import build_graphs, torch_load, upstream_modules
from adarpro.models.fixed_checkpoint_ensemble import (
    equal_logit_average,
    model_for_mode,
)
from adarpro.models.multitask_oof import predict_binary
from adarpro.models.residual import (
    CS,
    best_f1_threshold,
    bootstrap_differences,
    matched_recall_threshold,
    metrics,
    rescue_metrics,
)
from adarpro.quarantine import sha256_file
from adarpro.registry import register

VIEWS = ("target", "local_mean", "global_mean")


def load_embedding_cache(
    array_path: Path, index_path: Path, canonical: pd.DataFrame
) -> dict[str, np.ndarray]:
    index = pd.read_parquet(index_path).sort_values("example_id").reset_index(drop=True)
    expected = canonical.sort_values("example_id").reset_index(drop=True)
    identity = [
        "example_id",
        "context",
        "pair_id",
        "pair_local_key",
        "local_position",
        "development_fold",
        "source_split",
    ]
    if not index[identity].equals(expected[identity]):
        raise RuntimeError("RiNALMo cache index does not match canonical sites")
    with np.load(array_path) as cache:
        if not np.array_equal(cache["example_id"].astype(str), index["example_id"]):
            raise RuntimeError("RiNALMo array/index example IDs do not match")
        views = {name: cache[name].astype(np.float32) for name in VIEWS}
    if any(values.shape[0] != len(index) for values in views.values()):
        raise RuntimeError("RiNALMo cache has an invalid row count")
    if any(not np.isfinite(values).all() for values in views.values()):
        raise RuntimeError("RiNALMo cache contains non-finite values")
    return views


def project_views(
    views: dict[str, np.ndarray],
    train_mask: np.ndarray,
    components_per_view: int,
    seed: int,
    names: tuple[str, ...] = VIEWS,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    projected, explained = {}, {}
    for offset, name in enumerate(names):
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(views[name][train_mask])
        dimension = min(
            components_per_view,
            scaled_train.shape[0] - 1,
            scaled_train.shape[1],
        )
        if dimension < 1:
            raise RuntimeError("insufficient training data for fold-fitted PCA")
        pca = PCA(n_components=dimension, random_state=seed + offset)
        pca.fit(scaled_train)
        projected[name] = pca.transform(scaler.transform(views[name]))
        explained[name] = float(pca.explained_variance_ratio_.sum())
    return projected, explained


def shuffle_within_folds(
    values: np.ndarray, folds: np.ndarray, seed: int
) -> np.ndarray:
    """Break site alignment while preserving each fold's feature distribution."""

    shuffled = np.empty_like(values)
    rng = np.random.default_rng(seed)
    for fold in np.unique(folds):
        indices = np.flatnonzero(folds == fold)
        shuffled[indices] = values[rng.permutation(indices)]
    return shuffled


def fit_gate(
    table: pd.DataFrame,
    features: np.ndarray,
    model_name: str,
    outer_fold: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    inner_fold = (outer_fold + 1) % 5
    train_mask = ~table["development_fold"].isin([outer_fold, inner_fold]).to_numpy()
    inner_mask = table["development_fold"].eq(inner_fold).to_numpy()
    outer_mask = table["development_fold"].eq(outer_fold).to_numpy()
    train_pairs = set(table.loc[train_mask, "pair_id"])
    inner_pairs = set(table.loc[inner_mask, "pair_id"])
    outer_pairs = set(table.loc[outer_mask, "pair_id"])
    if train_pairs & inner_pairs or train_pairs & outer_pairs or inner_pairs & outer_pairs:
        raise RuntimeError("pair leakage in foundation residual gate")
    y_train = table.loc[train_mask, "label"].to_numpy()
    y_inner = table.loc[inner_mask, "label"].to_numpy()
    best = None
    for c_value in CS:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                penalty="l2",
                solver="liblinear",
                max_iter=5000,
                random_state=seed + outer_fold,
            ),
        )
        model.fit(features[train_mask], y_train)
        inner_probability = model.predict_proba(features[inner_mask])[:, 1]
        threshold, score = best_f1_threshold(y_inner, inner_probability)
        candidate = (score, -c_value, model, inner_probability, threshold)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    _, negative_c, model, inner_probability, threshold = best
    inner_reference_recall = recall_score(
        y_inner,
        table.loc[inner_mask, "base_prediction"],
        zero_division=0,
    )
    recall_threshold = matched_recall_threshold(
        y_inner, inner_probability, inner_reference_recall
    )
    probability = model.predict_proba(features[outer_mask])[:, 1]
    outer = table.loc[
        outer_mask,
        [
            "example_id",
            "context",
            "pair_id",
            "pair_local_key",
            "label",
            "development_fold",
            "base_probability",
            "base_prediction",
        ],
    ].copy()
    outer["model"] = model_name
    outer["probability"] = probability
    outer["threshold"] = threshold
    outer["prediction"] = (probability >= threshold).astype(int)
    outer["matched_recall_threshold"] = recall_threshold
    outer["matched_recall_prediction"] = (probability >= recall_threshold).astype(int)
    return outer, {
        "model": model_name,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "C": -negative_c,
        "threshold": threshold,
        "matched_recall_threshold": recall_threshold,
        "inner_f1": best[0],
        "feature_count": int(features.shape[1]),
    }


def predict_outer_isolated_base(
    canonical: pd.DataFrame,
    graphs: list,
    left_run_dir: Path,
    left_mode: str,
    right_run_dir: Path,
    right_mode: str,
    ensemble_run_dir: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, pd.DataFrame], dict[str, str], float]:
    thresholds = pd.read_csv(ensemble_run_dir / "fold_selections.csv").set_index(
        "outer_fold"
    )["threshold"]
    saved_ensemble = pd.read_parquet(
        ensemble_run_dir / "oof_predictions.parquet"
    ).sort_values("example_id")
    outputs, checkpoint_hashes = {}, {}
    maximum_difference = 0.0
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    for outer_fold in range(5):
        probabilities = []
        labels = None
        for name, mode, run_dir in (
            ("left", left_mode, left_run_dir),
            ("right", right_mode, right_run_dir),
        ):
            checkpoint = run_dir / f"outer_{outer_fold}" / "best.pth"
            checkpoint_hashes[f"{name}_outer_{outer_fold}"] = sha256_file(checkpoint)
            saved = torch_load(checkpoint, device)
            model = model_for_mode(mode, device)
            model.load_state_dict(saved["model_state"])
            probability, observed_labels = predict_binary(model, loader, device)
            if labels is None:
                labels = observed_labels
            elif not np.array_equal(labels, observed_labels):
                raise RuntimeError("foundation base component labels differ")
            probabilities.append(probability)
            del model
        if not np.array_equal(labels, canonical["label"].to_numpy()):
            raise RuntimeError("foundation base prediction order mismatch")
        probability = equal_logit_average(probabilities[0], probabilities[1])
        clipped = np.clip(probability, 1e-7, 1 - 1e-7)
        table = canonical[
            [
                "example_id",
                "context",
                "pair_id",
                "pair_local_key",
                "label",
                "development_fold",
            ]
        ].copy()
        table["base_probability"] = probability
        table["base_logit"] = np.log(clipped / (1 - clipped))
        table["base_prediction"] = (
            probability >= float(thresholds.loc[outer_fold])
        ).astype(int)
        observed_outer = table[table["development_fold"] == outer_fold].sort_values(
            "example_id"
        )
        expected_outer = saved_ensemble[
            saved_ensemble["development_fold"] == outer_fold
        ]
        if not np.array_equal(
            observed_outer["example_id"].to_numpy(),
            expected_outer["example_id"].to_numpy(),
        ):
            raise RuntimeError("saved ensemble outer row order mismatch")
        maximum_difference = max(
            maximum_difference,
            float(
                np.max(
                    np.abs(
                        observed_outer["base_probability"].to_numpy()
                        - expected_outer["probability"].to_numpy()
                    )
                )
            ),
        )
        outputs[outer_fold] = table
    if maximum_difference > 1e-5:
        raise RuntimeError(
            f"recomputed base ensemble mismatch: {maximum_difference}"
        )
    return outputs, checkpoint_hashes, maximum_difference


def run(
    repository: Path,
    site_master_path: Path,
    embedding_path: Path,
    embedding_index_path: Path,
    left_run_dir: Path,
    left_mode: str,
    right_run_dir: Path,
    right_mode: str,
    ensemble_run_dir: Path,
    output_dir: Path,
    context: str,
    foundation_name: str,
    representation: str,
    components_per_view: int,
    batch_size: int,
    bootstrap_replicates: int,
    seed: int,
    device: torch.device,
) -> dict:
    if representation not in {"all_views", "target_only"}:
        raise ValueError(f"unknown foundation representation: {representation}")
    if components_per_view < 1:
        raise ValueError("components_per_view must be positive")
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("foundation residual received a sealed split")
    canonical = (
        master[master["context"] == context]
        .sort_values("example_id")
        .reset_index(drop=True)
    )
    views = load_embedding_cache(embedding_path, embedding_index_path, canonical)
    graphs = build_graphs(canonical, upstream_modules(repository))
    base_tables, checkpoint_hashes, maximum_difference = predict_outer_isolated_base(
        canonical,
        graphs,
        left_run_dir,
        left_mode,
        right_run_dir,
        right_mode,
        ensemble_run_dir,
        batch_size,
        device,
    )
    outputs, selections = [], []
    fold_values = canonical["development_fold"].to_numpy()
    for outer_fold in range(5):
        inner_fold = (outer_fold + 1) % 5
        train_mask = ~np.isin(fold_values, [outer_fold, inner_fold])
        view_names = ("target",) if representation == "target_only" else VIEWS
        projected, explained = project_views(
            views,
            train_mask,
            components_per_view,
            seed + 1000 * outer_fold,
            view_names,
        )
        base = base_tables[outer_fold]
        base_logit = base["base_logit"].to_numpy()[:, None]
        selected = (
            projected["target"]
            if representation == "target_only"
            else np.concatenate([projected[name] for name in VIEWS], axis=1)
        )
        shuffled = shuffle_within_folds(selected, fold_values, seed + 10_000 + outer_fold)
        random = np.random.default_rng(seed + 20_000 + outer_fold).normal(
            size=selected.shape
        )
        feature_sets: dict[str, np.ndarray] = {
            "base_recalibrated": base_logit,
            "foundation_target": np.column_stack([base_logit, projected["target"]]),
            "foundation_shuffled_control": np.column_stack([base_logit, shuffled]),
            "foundation_random_control": np.column_stack([base_logit, random]),
        }
        if representation == "all_views":
            feature_sets.update(
                {
                    "foundation_local": np.column_stack(
                        [base_logit, projected["local_mean"]]
                    ),
                    "foundation_global": np.column_stack(
                        [base_logit, projected["global_mean"]]
                    ),
                    "foundation_combined": np.column_stack([base_logit, selected]),
                }
            )
        for model_name, features in feature_sets.items():
            outer, selection = fit_gate(
                base, features, model_name, outer_fold, seed
            )
            outputs.append(outer)
            selection.update(
                {
                    "pca_components_per_view": components_per_view,
                    **{
                        f"{name}_explained_variance": value
                        for name, value in explained.items()
                    },
                }
            )
            selections.append(selection)
        raw = base[base["development_fold"] == outer_fold].copy()
        raw["model"] = "base_raw"
        raw["probability"] = raw["base_probability"]
        raw["prediction"] = raw["base_prediction"]
        raw["threshold"] = np.nan
        raw["matched_recall_threshold"] = np.nan
        raw["matched_recall_prediction"] = raw["base_prediction"]
        outputs.append(raw)
    predictions = pd.concat(outputs, ignore_index=True)
    reference = predictions[predictions["model"] == "base_recalibrated"].copy()
    raw_reference = predictions[predictions["model"] == "base_raw"].copy()
    metric_rows, bootstrap_rows = [], []
    for model_name, group in predictions.groupby("model", sort=False):
        row = metrics(group["label"], group["probability"], group["prediction"])
        raw_predictions = raw_reference[["example_id", "prediction"]].rename(
            columns={"prediction": "baseline_oof_prediction"}
        )
        rescue = group.merge(raw_predictions, on="example_id", validate="one_to_one")
        row.update(rescue_metrics(rescue, "prediction"))
        row["model"] = model_name
        metric_rows.append(row)
        if model_name != "base_raw":
            bootstrap = bootstrap_differences(
                group, reference, bootstrap_replicates, seed + 30_000
            )
            bootstrap["candidate"] = model_name
            bootstrap["reference"] = "base_recalibrated"
            bootstrap_rows.append(bootstrap)
    primary_name = (
        "foundation_target"
        if representation == "target_only"
        else "foundation_combined"
    )
    primary_group = predictions[predictions["model"] == primary_name]
    control_names = ["foundation_shuffled_control", "foundation_random_control"]
    if representation == "all_views":
        control_names = ["foundation_target", "foundation_local", "foundation_global", *control_names]
    for control_name in control_names:
        control = predictions[predictions["model"] == control_name]
        bootstrap = bootstrap_differences(
            primary_group, control, bootstrap_replicates, seed + 40_000
        )
        bootstrap["candidate"] = primary_name
        bootstrap["reference"] = control_name
        bootstrap_rows.append(bootstrap)
    metric_table = pd.DataFrame(metric_rows)
    bootstrap = pd.concat(bootstrap_rows, ignore_index=True)
    confidence = (
        bootstrap.groupby(["candidate", "reference", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(output_dir / "fold_selections.csv", index=False)
    metric_table.to_csv(output_dir / "metrics.csv", index=False)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(output_dir / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "outer-isolated development gate; no outer pair enters either neural component, PCA or gate fitting",
        "context": context,
        "foundation_name": foundation_name,
        "representation": representation,
        "rows": len(canonical),
        "pairs": int(canonical["pair_id"].nunique()),
        "base_components": [left_mode, right_mode],
        "components_per_foundation_view": components_per_view,
        "maximum_recomputed_outer_difference": maximum_difference,
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
                f"foundation_residual_{foundation_name}_{representation}_"
                f"{context}_seed{seed}"
            ),
            "experiment_type": "outer_isolated_frozen_rinalmo_residual",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "embedding_array": sha256_file(embedding_path),
                "embedding_index": sha256_file(embedding_index_path),
                "ensemble_selections": sha256_file(
                    ensemble_run_dir / "fold_selections.csv"
                ),
                "base_checkpoints": checkpoint_hashes,
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": (
                "base ensemble logit plus fold-fitted PCA of frozen RiNALMo "
                + ("target token" if representation == "target_only" else "target/local/global views")
            ),
            "hyperparameters": {
                "components_per_view": components_per_view,
                "representation": representation,
                "C_grid": list(CS),
                "selection": "inner-fold F1; threshold on same inner fold",
                "bootstrap_replicates": bootstrap_replicates,
            },
            "artifact": str((output_dir / "oof_predictions.parquet").relative_to(repository)),
            "metrics": metric_table.to_dict(orient="records"),
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument(
        "--embedding-array",
        type=Path,
        default=Path(
            "data/derived/foundation/rinalmo_micro/MuscleSkeletal/embeddings.npz"
        ),
    )
    parser.add_argument(
        "--embedding-index",
        type=Path,
        default=Path(
            "data/derived/foundation/rinalmo_micro/MuscleSkeletal/index.parquet"
        ),
    )
    parser.add_argument("--left-run-dir", type=Path, required=True)
    parser.add_argument("--left-mode", required=True)
    parser.add_argument("--right-run-dir", type=Path, required=True)
    parser.add_argument("--right-mode", required=True)
    parser.add_argument("--ensemble-run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/foundation_residual/rinalmo_micro/MuscleSkeletal"),
    )
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument("--foundation-name", default="rinalmo_micro")
    parser.add_argument(
        "--representation",
        choices=("all_views", "target_only"),
        default="all_views",
    )
    parser.add_argument("--components-per-view", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(
        json.dumps(
            run(
                root,
                resolve(args.site_master),
                resolve(args.embedding_array),
                resolve(args.embedding_index),
                resolve(args.left_run_dir),
                args.left_mode,
                resolve(args.right_run_dir),
                args.right_mode,
                resolve(args.ensemble_run_dir),
                resolve(args.output_dir),
                args.context,
                args.foundation_name,
                args.representation,
                args.components_per_view,
                args.batch_size,
                args.bootstrap_replicates,
                args.seed,
                device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
