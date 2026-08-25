"""Outer-isolated gate combining expanded GAT and predicted pair engagement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from adarpro.models.baseline_oof import build_graphs, torch_load, upstream_modules
from adarpro.models.multitask_oof import MultiTaskGAT, predict_binary
from adarpro.models.pair_engagement import make_model, prepare_pair_table
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


def select_pair_model(
    pair_table: pd.DataFrame,
    train_folds: list[int],
    columns: list[str],
    seed: int,
    shuffled: bool,
):
    parameters = (0.1, 1.0, 10.0, 100.0)
    losses = []
    training = pair_table[pair_table["development_fold"].isin(train_folds)].copy()
    target = training["edited_fraction"].to_numpy().copy()
    if shuffled:
        np.random.default_rng(seed).shuffle(target)
    training["model_target"] = target
    for parameter in parameters:
        fold_losses = []
        for held_fold in train_folds:
            fit = training[training["development_fold"] != held_fold]
            held = training[training["development_fold"] == held_fold]
            model = make_model("ridge", parameter, seed + held_fold)
            model.fit(fit[columns], fit["model_target"])
            prediction = np.clip(model.predict(held[columns]), 0, 1)
            fold_losses.append(mean_squared_error(held["model_target"], prediction))
        losses.append((float(np.mean(fold_losses)), parameter))
    _, selected = min(losses)
    model = make_model("ridge", selected, seed)
    model.fit(training[columns], training["model_target"])
    return model, selected


def fit_gate(
    table: pd.DataFrame,
    columns: list[str],
    outer_fold: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    inner_fold = (outer_fold + 1) % 5
    train = table[~table["development_fold"].isin([outer_fold, inner_fold])]
    inner = table[table["development_fold"] == inner_fold]
    outer = table[table["development_fold"] == outer_fold].copy()
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
        model.fit(train[columns], train["label"])
        probability = model.predict_proba(inner[columns])[:, 1]
        threshold, f1 = best_f1_threshold(inner["label"].to_numpy(), probability)
        candidate = (f1, -c_value, model, probability, threshold)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    _, negative_c, model, inner_probability, threshold = best
    inner_reference_recall = float(
        np.mean(
            inner.loc[inner["label"] == 1, "expanded_reference_prediction"] == 1
        )
    )
    recall_threshold = matched_recall_threshold(
        inner["label"].to_numpy(), inner_probability, inner_reference_recall
    )
    probability = model.predict_proba(outer[columns])[:, 1]
    outer["probability"] = probability
    outer["threshold"] = threshold
    outer["prediction"] = (probability >= threshold).astype(int)
    outer["matched_recall_threshold"] = recall_threshold
    outer["matched_recall_prediction"] = (probability >= recall_threshold).astype(int)
    return outer, {
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "C": -negative_c,
        "threshold": threshold,
        "matched_recall_threshold": recall_threshold,
        "inner_f1": best[0],
        "feature_count": len(columns),
    }


def run(
    repository: Path,
    site_master_path: Path,
    raw_path: Path,
    genomic_path: Path,
    base_run_dir: Path,
    context: str,
    mode: str,
    output_dir: Path,
    batch_size: int,
    bootstrap_replicates: int,
    seed: int,
    device: torch.device,
) -> dict:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("hierarchical gate received a sealed split")
    canonical = master[master["context"] == context].copy().reset_index(drop=True)
    raw = pd.read_parquet(raw_path)
    raw = raw[raw["context"] == context].copy()
    genomic = pd.read_parquet(genomic_path)
    pair_table, combined, _ = prepare_pair_table(raw, genomic, context)
    graph_intrinsic = [column for column in combined if column.startswith("intrinsic__")]
    baseline = upstream_modules(repository)
    graphs = build_graphs(canonical, baseline)
    outputs, selections, checkpoint_hashes = [], [], {}
    maximum_outer_difference = 0.0
    for outer_fold in range(5):
        inner_fold = (outer_fold + 1) % 5
        train_folds = sorted(set(range(5)) - {outer_fold, inner_fold})
        checkpoint = base_run_dir / f"outer_{outer_fold}" / "best.pth"
        checkpoint_hashes[f"outer_{outer_fold}"] = sha256_file(checkpoint)
        saved = torch_load(checkpoint, device)
        base_model = MultiTaskGAT().to(device)
        base_model.load_state_dict(saved["model_state"])
        probability, labels = predict_binary(
            base_model,
            DataLoader(graphs, batch_size=batch_size, shuffle=False),
            device,
        )
        if not np.array_equal(labels, canonical["label"].to_numpy()):
            raise RuntimeError("base-model prediction order mismatch")
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
        clipped = np.clip(probability, 1e-7, 1 - 1e-7)
        table["expanded_logit"] = np.log(clipped / (1 - clipped))
        table["expanded_probability"] = probability
        table["expanded_reference_prediction"] = (
            probability >= float(saved["best"]["threshold"])
        ).astype(int)
        saved_outer = pd.read_parquet(
            base_run_dir / f"outer_{outer_fold}" / "outer_predictions.parquet"
        ).sort_values("example_id")
        recomputed_outer = table[table["development_fold"] == outer_fold].sort_values(
            "example_id"
        )
        maximum_outer_difference = max(
            maximum_outer_difference,
            float(
                np.max(
                    np.abs(
                        saved_outer["probability"].to_numpy()
                        - recomputed_outer["expanded_probability"].to_numpy()
                    )
                )
            ),
        )
        score_specs = {
            "graph_engagement": (graph_intrinsic, False),
            "combined_engagement": (combined, False),
            "shuffled_engagement": (combined, True),
        }
        pair_scores = pair_table[["pair_id", *combined]].copy()
        for score_name, (columns, shuffled) in score_specs.items():
            pair_model, parameter = select_pair_model(
                pair_table,
                train_folds,
                columns,
                seed + 10_000 * outer_fold,
                shuffled,
            )
            pair_scores[score_name] = np.clip(
                pair_model.predict(pair_table[columns]), 0, 1
            )
            selections.append(
                {
                    "model": score_name,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "pair_ridge_alpha": parameter,
                    "pair_feature_count": len(columns),
                }
            )
        table = table.merge(
            pair_scores[
                [
                    "pair_id",
                    "graph_engagement",
                    "combined_engagement",
                    "shuffled_engagement",
                ]
            ],
            on="pair_id",
            validate="many_to_one",
        )
        gate_sets = {
            "expanded_recalibrated": ["expanded_logit"],
            "expanded_plus_graph_engagement": [
                "expanded_logit",
                "graph_engagement",
            ],
            "expanded_plus_combined_engagement": [
                "expanded_logit",
                "combined_engagement",
            ],
            "expanded_plus_shuffled_engagement": [
                "expanded_logit",
                "shuffled_engagement",
            ],
        }
        for name, columns in gate_sets.items():
            outer, selection = fit_gate(table, columns, outer_fold, seed)
            outer["model"] = name
            outputs.append(outer)
            selection["model"] = name
            selections.append(selection)
        reference = table[table["development_fold"] == outer_fold].copy()
        reference["model"] = "expanded_reference"
        reference["probability"] = reference["expanded_probability"]
        reference["prediction"] = reference["expanded_reference_prediction"]
        reference["threshold"] = float(saved["best"]["threshold"])
        reference["matched_recall_threshold"] = np.nan
        reference["matched_recall_prediction"] = reference["prediction"]
        outputs.append(reference)
    if maximum_outer_difference > 5e-5:
        raise RuntimeError(
            f"recomputed expanded outer probability mismatch: {maximum_outer_difference}"
        )
    predictions = pd.concat(outputs, ignore_index=True)
    reference = predictions[predictions["model"] == "expanded_reference"].copy()
    metric_rows, bootstraps = [], []
    for name, group in predictions.groupby("model", sort=False):
        row = metrics(group["label"], group["probability"], group["prediction"])
        reference_view = reference if name != "expanded_reference" else group
        group_for_rescue = group.merge(
            reference_view[["example_id", "prediction"]].rename(
                columns={"prediction": "baseline_oof_prediction"}
            ),
            on="example_id",
            validate="one_to_one",
        )
        row.update(rescue_metrics(group_for_rescue, "prediction"))
        row["model"] = name
        metric_rows.append(row)
        values = bootstrap_differences(group, reference, bootstrap_replicates, seed + 4000)
        values["model"] = name
        bootstraps.append(values)
    metric_table = pd.DataFrame(metric_rows)
    bootstrap = pd.concat(bootstraps, ignore_index=True)
    confidence = (
        bootstrap.groupby(["model", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    destination = output_dir / mode / context
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(destination / "oof_predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(destination / "fold_selections.csv", index=False)
    metric_table.to_csv(destination / "metrics.csv", index=False)
    bootstrap.to_parquet(destination / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(destination / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "outer-isolated development gate; no outer pair enters either stage",
        "context": context,
        "base_mode": mode,
        "rows": len(reference),
        "pairs": int(reference["pair_id"].nunique()),
        "maximum_recomputed_outer_difference": maximum_outer_difference,
        "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"hierarchical_gate_{mode}_{context}_seed{seed}",
            "experiment_type": "outer_isolated_expanded_plus_pair_engagement_gate",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "raw_development_sites": sha256_file(raw_path),
                "genomic_context": sha256_file(genomic_path),
                "base_checkpoints": checkpoint_hashes,
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": "outer-specific expanded GAT logit plus predicted pair engagement",
            "hyperparameters": {
                "pair_model": "ridge; alpha selected by CV within three training folds",
                "gate": "L2 logistic; C/threshold selected on inner fold",
                "bootstrap_replicates": bootstrap_replicates,
            },
            "artifact": str(destination / "oof_predictions.parquet"),
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
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument("--mode", default="binary_natural")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/runs/hierarchical_gate")
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(
        json.dumps(
            run(
                root,
                resolve(args.site_master),
                resolve(args.raw_sites),
                resolve(args.genomic_features),
                resolve(args.base_run_dir),
                args.context,
                args.mode,
                resolve(args.output_dir),
                args.batch_size,
                args.bootstrap_replicates,
                args.seed,
                torch.device(args.device),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
