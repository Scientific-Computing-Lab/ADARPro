"""Leakage-free matched-recall operating points for completed neural OOF runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import recall_score
from torch_geometric.loader import DataLoader

from adarpro.models.baseline_oof import (
    build_graphs,
    metric_row,
    torch_load,
    upstream_modules,
)
from adarpro.models.multitask_oof import (
    MultiTaskGAT,
    attach_local_windows,
    attach_pair_context,
    bootstrap_delta,
    pair_context_table,
    predict_binary,
)
from adarpro.models.residual import matched_recall_threshold, rescue_metrics
from adarpro.quarantine import sha256_file
from adarpro.registry import register


def run(
    repository: Path,
    site_master_path: Path,
    raw_path: Path,
    run_dir: Path,
    mode: str,
    context: str,
    output_dir: Path,
    batch_size: int,
    bootstrap_replicates: int,
    seed: int,
    device: torch.device,
) -> dict:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("matched-recall audit received a sealed split")
    canonical = master[master["context"] == context].copy().reset_index(drop=True)
    if canonical["baseline_oof_probability"].isna().any():
        raise RuntimeError(f"development reference is incomplete for {context}")
    baseline = upstream_modules(repository)
    graphs = build_graphs(canonical, baseline)
    if mode.endswith("_local"):
        attach_local_windows(graphs, canonical)
    if "_context" in mode:
        raw = pd.read_parquet(raw_path)
        raw = raw[raw["context"] == context].copy()
        attach_pair_context(graphs, canonical, pair_context_table(raw))
    prediction_tables, thresholds, checkpoint_hashes = [], [], {}
    for outer_fold in range(5):
        inner_fold = (outer_fold + 1) % 5
        inner_indices = canonical.index[
            canonical["development_fold"] == inner_fold
        ].to_numpy()
        inner_loader = DataLoader(
            [graphs[index] for index in inner_indices],
            batch_size=batch_size,
            shuffle=False,
        )
        checkpoint_path = run_dir / f"outer_{outer_fold}" / "best.pth"
        checkpoint_hashes[f"outer_{outer_fold}"] = sha256_file(checkpoint_path)
        saved = torch_load(checkpoint_path, device)
        model = MultiTaskGAT(
            use_local=mode.endswith("_local"),
            use_target_pool=mode.endswith("_target_pool"),
            use_anchor_pool=mode.endswith("_anchor_pool"),
            use_duplicate_pool=mode.endswith("_duplicate_pool"),
            use_pair_context="_context" in mode,
        ).to(device)
        model.load_state_dict(saved["model_state"])
        probability, labels = predict_binary(model, inner_loader, device)
        baseline_recall = recall_score(
            labels,
            canonical.loc[inner_indices, "baseline_oof_prediction"],
            zero_division=0,
        )
        threshold = matched_recall_threshold(labels, probability, baseline_recall)
        outer_path = run_dir / f"outer_{outer_fold}" / "outer_predictions.parquet"
        outer = pd.read_parquet(outer_path)
        if set(outer["development_fold"]) != {outer_fold}:
            raise RuntimeError(f"outer prediction fold mismatch: {outer_path}")
        outer["matched_recall_threshold"] = threshold
        outer["matched_recall_prediction"] = (outer["probability"] >= threshold).astype(
            int
        )
        outer["inner_baseline_recall"] = baseline_recall
        prediction_tables.append(outer)
        thresholds.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "matched_recall_threshold": threshold,
                "inner_baseline_recall": baseline_recall,
            }
        )
    predictions = pd.concat(prediction_tables, ignore_index=True)
    matched = predictions.copy()
    matched["prediction"] = matched["matched_recall_prediction"]
    candidate_metrics = metric_row(
        matched["label"], matched["probability"], matched["prediction"]
    )
    candidate_metrics.update(rescue_metrics(matched, "prediction"))
    reference_metrics = metric_row(
        matched["label"],
        matched["baseline_oof_probability"],
        matched["baseline_oof_prediction"],
    )
    bootstrap = bootstrap_delta(matched, bootstrap_replicates, seed)
    confidence = (
        bootstrap.groupby("metric")["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame(thresholds).to_csv(output_dir / "fold_thresholds.csv", index=False)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(output_dir / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "canonical train+valid grouped OOF only",
        "mode": mode,
        "context": context,
        "selection": "threshold chosen on inner fold to meet development-reference recall",
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
            "run_id": f"matched_recall_{mode}_{context}_seed{seed}",
            "experiment_type": "neural_oof_matched_recall_audit",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "checkpoints": checkpoint_hashes,
            },
            "fold_definition": "five global pair folds; threshold from inner (k+1)%5",
            "seed": seed,
            "features": f"completed {mode} OOF probability",
            "hyperparameters": {
                "batch_size": batch_size,
                "bootstrap_replicates": bootstrap_replicates,
            },
            "artifact": str(output_dir / "oof_predictions.parquet"),
            "metrics": summary,
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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument("--output-dir", type=Path, required=True)
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
                resolve(args.raw_sites),
                resolve(args.run_dir),
                args.mode,
                args.context,
                resolve(args.output_dir),
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
