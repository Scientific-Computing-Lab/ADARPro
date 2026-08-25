"""Evaluate grouped-OOF baseline checkpoints at natural development prevalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from adarpro.models.baseline_oof import (
    build_graphs,
    create_model,
    metric_row,
    predict,
    torch_load,
    upstream_modules,
)


def clustered_bootstrap(
    predictions: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    pairs = predictions["pair_id"].unique()
    pair_values = predictions["pair_id"].to_numpy()
    indices = {pair_id: np.flatnonzero(pair_values == pair_id) for pair_id in pairs}
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        subset = predictions.iloc[
            np.concatenate([indices[pair_id] for pair_id in sampled])
        ]
        metrics = metric_row(
            subset["label"], subset["probability"], subset["prediction"]
        )
        rows.append({"replicate": replicate, **metrics})
    return pd.DataFrame(rows)


def upsert_context_metrics(path: Path, rows: list[dict]) -> None:
    """Preserve previously evaluated contexts when a subset is requested."""

    update = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[~existing["context"].isin(update["context"])]
        update = pd.concat([existing, update], ignore_index=True)
    update.sort_values("context").to_csv(path, index=False)


def run_context(
    repository: Path,
    raw_path: Path,
    checkpoint_root: Path,
    context: str,
    output_dir: Path,
    batch_size: int,
    bootstrap_replicates: int,
    seed: int,
    device: torch.device,
) -> dict:
    raw = pd.read_parquet(raw_path)
    if set(raw["development_fold"].unique()) != set(range(5)):
        raise RuntimeError("natural-prevalence input lacks all development folds")
    table = raw[(raw["context"] == context) & raw["binary_label"].notna()].copy()
    table["label"] = table["binary_label"].astype(int)
    table = table.reset_index(drop=True)
    if table.empty:
        raise RuntimeError(f"no binary natural-prevalence rows for {context}")
    baseline = upstream_modules(repository)
    print(f"[{context}] building {len(table)} natural-prevalence graphs", flush=True)
    graphs = build_graphs(table, baseline)
    folds = []
    for outer_fold in range(5):
        indices = np.flatnonzero(table["development_fold"].to_numpy() == outer_fold)
        loader = DataLoader(
            [graphs[index] for index in indices],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        checkpoint_path = checkpoint_root / context / f"outer_{outer_fold}" / "best.pth"
        if not checkpoint_path.exists():
            raise RuntimeError(f"missing grouped-OOF checkpoint: {checkpoint_path}")
        model, _, _ = create_model(baseline, device)
        saved = torch_load(checkpoint_path, device)
        model.load_state_dict(saved["model_state"])
        probabilities, labels = predict(model, loader, device)
        subset = table.iloc[indices][
            [
                "example_id",
                "context",
                "pair_id",
                "pair_local_key",
                "site_id",
                "editing_index",
                "development_fold",
                "label",
            ]
        ].copy()
        if not np.array_equal(subset["label"].to_numpy(), labels):
            raise RuntimeError("natural-prevalence graph label order mismatch")
        threshold = float(saved["best"]["threshold"])
        subset["probability"] = probabilities
        subset["threshold"] = threshold
        subset["prediction"] = (probabilities >= threshold).astype(int)
        folds.append(subset)
    predictions = pd.concat(folds, ignore_index=True)
    metrics = metric_row(
        predictions["label"], predictions["probability"], predictions["prediction"]
    )
    metrics.update(
        {
            "context": context,
            "pairs": int(predictions["pair_id"].nunique()),
            "positive_prevalence": float(predictions["label"].mean()),
            "scope": "natural-prevalence train+valid pairs, grouped OOF",
            "canonical_test_access": False,
        }
    )
    context_dir = output_dir / context
    context_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(context_dir / "oof_predictions.parquet", index=False)
    bootstrap = clustered_bootstrap(predictions, bootstrap_replicates, seed)
    bootstrap.to_parquet(context_dir / "pair_bootstrap.parquet", index=False)
    ci = (
        bootstrap.drop(columns="replicate")
        .quantile([0.025, 0.5, 0.975])
        .transpose()
        .reset_index(names="metric")
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    ci.to_csv(context_dir / "pair_bootstrap_ci.csv", index=False)
    (context_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


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
        "--checkpoint-root",
        type=Path,
        default=Path("experiments/runs/baseline_oof"),
    )
    parser.add_argument("--contexts", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/natural_prevalence"),
    )
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
    rows = [
        run_context(
            root,
            resolve(args.raw_sites),
            resolve(args.checkpoint_root),
            context,
            resolve(args.output_dir),
            args.batch_size,
            args.bootstrap_replicates,
            args.seed,
            device,
        )
        for context in args.contexts
    ]
    output = resolve(args.output_dir)
    upsert_context_metrics(output / "metrics.csv", rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
