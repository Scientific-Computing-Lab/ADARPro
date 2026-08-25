"""Nested pair-grouped out-of-fold predictions from the exact Baseline GAT."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch_geometric.loader import DataLoader

from adarpro.quarantine import sha256_file
from adarpro.registry import register

CONTEXTS = ("Artery", "Brain", "Liver", "MuscleSkeletal", "Combined")
THRESHOLDS = np.linspace(0.1, 0.9, 33)


def upstream_modules(repository: Path):
    code_dir = repository / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import gnnadar_verb_compact as baseline

    return baseline


def configure(seed: int, threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, threads)))
    except RuntimeError:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def pair_hash(values: pd.Series) -> str:
    payload = ",".join(str(value) for value in sorted(set(values))).encode()
    return hashlib.sha256(payload).hexdigest()


def build_graphs(table: pd.DataFrame, baseline) -> list:
    graphs = []
    for row in table.itertuples(index=False):
        graph = baseline.create_rna_graph(
            row.sequence, row.structure, int(row.local_position), int(row.label)
        )
        if graph is None:
            raise RuntimeError(f"unbuildable graph: {row.example_id}")
        graphs.append(graph)
    return graphs


def create_model(baseline, device: torch.device):
    model = baseline.RNAEditingGNN(8, 32, 1, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return model, optimizer, nn.BCELoss()


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        output, _ = model(batch)
        probabilities.extend(output.detach().cpu().view(-1).tolist())
        labels.extend(batch.y.detach().cpu().view(-1).int().tolist())
    return np.asarray(probabilities), np.asarray(labels)


def train_epoch(model, loader, optimizer, loss_function, device) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output, _ = model(batch)
        loss = loss_function(output.view(-1), batch.y.view(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {float(loss)}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    scores = [
        f1_score(labels, probabilities >= threshold, zero_division=0)
        for threshold in THRESHOLDS
    ]
    index = int(np.argmax(scores))
    return float(THRESHOLDS[index]), float(scores[index])


def metric_row(labels, probabilities, predictions) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return {
        "n": len(labels),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(recall_score(labels, predictions, pos_label=0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def append_history(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_fold(
    table: pd.DataFrame,
    graphs: list,
    baseline,
    context: str,
    outer_fold: int,
    epochs: int,
    batch_size: int,
    checkpoint_every: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    inner_fold = (outer_fold + 1) % 5
    outer_mask = table["development_fold"].to_numpy() == outer_fold
    inner_mask = table["development_fold"].to_numpy() == inner_fold
    train_mask = ~(outer_mask | inner_mask)
    partitions = {
        "train": np.flatnonzero(train_mask),
        "selection": np.flatnonzero(inner_mask),
        "outer": np.flatnonzero(outer_mask),
    }
    pair_sets = {
        name: set(table.iloc[indices]["pair_id"]) for name, indices in partitions.items()
    }
    if any(
        pair_sets[left] & pair_sets[right]
        for left, right in (("train", "selection"), ("train", "outer"), ("selection", "outer"))
    ):
        raise RuntimeError("pair leakage across nested OOF partitions")
    fold_dir = output_dir / context / f"outer_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "context": context,
        "outer_fold": outer_fold,
        "inner_selection_fold": inner_fold,
        "training_folds": sorted(set(range(5)) - {outer_fold, inner_fold}),
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "architecture": "upstream Baseline GAT: RNAEditingGNN(8,32,1,4)",
        "optimizer": "Adam(lr=1e-3,weight_decay=1e-4)",
        "selection": "maximum inner-fold F1; 33 thresholds from 0.1 to 0.9",
        "partition_rows": {name: len(indices) for name, indices in partitions.items()},
        "partition_pairs": {name: len(pair_sets[name]) for name in partitions},
        "partition_pair_sha256": {
            name: pair_hash(table.iloc[indices]["pair_id"])
            for name, indices in partitions.items()
        },
        "canonical_test_access": False,
    }
    config_path = fold_dir / "config.json"
    expected_config = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if config_path.exists() and config_path.read_text() != expected_config:
        raise RuntimeError(f"refusing incompatible resume: {config_path}")
    config_path.write_text(expected_config)
    loaders = {
        "train": DataLoader(
            [graphs[index] for index in partitions["train"]],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        ),
        "selection": DataLoader(
            [graphs[index] for index in partitions["selection"]],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "outer": DataLoader(
            [graphs[index] for index in partitions["outer"]],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }
    fold_seed = seed + 1000 * list(CONTEXTS).index(context) + outer_fold
    configure(fold_seed, threads=8)
    model, optimizer, loss_function = create_model(baseline, device)
    best_path = fold_dir / "best.pth"
    last_path = fold_dir / "last.pth"
    history_path = fold_dir / "history.csv"
    start_epoch = 1
    best = {"f1": -1.0, "epoch": 0, "threshold": None}
    if last_path.exists():
        saved = torch_load(last_path, device)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng_state(saved["rng_state"])
        best = saved["best"]
        start_epoch = int(saved["epoch"]) + 1
    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        loss = train_epoch(model, loaders["train"], optimizer, loss_function, device)
        probabilities, labels = predict(model, loaders["selection"], device)
        threshold, selection_f1 = choose_threshold(labels, probabilities)
        if selection_f1 > best["f1"] + 1e-6:
            best = {"f1": selection_f1, "epoch": epoch, "threshold": threshold}
            atomic_save(
                {"model_state": model.state_dict(), "best": best, "config": config},
                best_path,
            )
        append_history(
            history_path,
            {
                "epoch": epoch,
                "train_loss": loss,
                "selection_f1": selection_f1,
                "selection_auroc": roc_auc_score(labels, probabilities),
                "selection_auprc": average_precision_score(labels, probabilities),
                "selection_threshold": threshold,
                "best_epoch": best["epoch"],
                "elapsed_sec": time.time() - started,
            },
        )
        if epoch % checkpoint_every == 0 or epoch == epochs:
            atomic_save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "rng_state": rng_state(),
                    "best": best,
                    "config": config,
                },
                last_path,
            )
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"[{context} outer={outer_fold}] epoch={epoch}/{epochs} "
                f"loss={loss:.5f} selection_f1={selection_f1:.5f} "
                f"best={best['epoch']}",
                flush=True,
            )
    saved = torch_load(best_path, device)
    model.load_state_dict(saved["model_state"])
    probabilities, labels = predict(model, loaders["outer"], device)
    threshold = float(saved["best"]["threshold"])
    outer = table.iloc[partitions["outer"]][
        [
            "example_id",
            "context",
            "pair_id",
            "pair_local_key",
            "site_id",
            "local_position",
            "label",
            "development_fold",
        ]
    ].copy()
    if not np.array_equal(outer["label"].to_numpy(), labels):
        raise RuntimeError("outer prediction order/label mismatch")
    outer["baseline_oof_probability"] = probabilities
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    outer["baseline_oof_logit"] = np.log(clipped / (1 - clipped))
    outer["baseline_oof_threshold"] = threshold
    outer["baseline_oof_prediction"] = (probabilities >= threshold).astype(int)
    outer["baseline_oof_status"] = np.select(
        [
            (outer["label"] == 1) & (outer["baseline_oof_prediction"] == 1),
            (outer["label"] == 0) & (outer["baseline_oof_prediction"] == 1),
            (outer["label"] == 0) & (outer["baseline_oof_prediction"] == 0),
        ],
        ["TP", "FP", "TN"],
        default="FN",
    )
    outer["baseline_best_epoch"] = int(saved["best"]["epoch"])
    outer["baseline_inner_fold"] = inner_fold
    outer["baseline_fold_seed"] = fold_seed
    outer_path = fold_dir / "outer_predictions.parquet"
    outer.to_parquet(outer_path, index=False)
    return outer


def train_context(
    repository: Path,
    site_master_path: Path,
    context: str,
    epochs: int,
    batch_size: int,
    checkpoint_every: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    baseline = upstream_modules(repository)
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("site master contains held-out examples")
    table = master[master["context"] == context].copy().reset_index(drop=True)
    if set(table["development_fold"].unique()) != set(range(5)):
        raise RuntimeError(f"incomplete folds for {context}")
    print(f"[{context}] building {len(table)} graphs", flush=True)
    graphs = build_graphs(table, baseline)
    predictions = [
        run_fold(
            table,
            graphs,
            baseline,
            context,
            outer_fold,
            epochs,
            batch_size,
            checkpoint_every,
            seed,
            device,
            output_dir,
        )
        for outer_fold in range(5)
    ]
    oof = pd.concat(predictions, ignore_index=True)
    if len(oof) != len(table) or oof["example_id"].duplicated().any():
        raise RuntimeError(f"OOF coverage failure for {context}")
    metrics = metric_row(
        oof["label"], oof["baseline_oof_probability"], oof["baseline_oof_prediction"]
    )
    metrics.update(
        {
            "context": context,
            "unique_pairs": int(oof["pair_id"].nunique()),
            "site_master_sha256": sha256_file(site_master_path),
        }
    )
    context_dir = output_dir / context
    oof_path = context_dir / "oof_predictions.parquet"
    metrics_path = context_dir / "oof_metrics.json"
    oof.to_parquet(oof_path, index=False)
    metrics["oof_predictions_sha256"] = sha256_file(oof_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return oof, metrics


def update_site_master(site_master_path: Path, predictions: list[pd.DataFrame]) -> None:
    lock_path = site_master_path.with_suffix(f"{site_master_path.suffix}.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        master = pd.read_parquet(site_master_path)
        columns = [
            "baseline_oof_probability",
            "baseline_oof_logit",
            "baseline_oof_threshold",
            "baseline_oof_prediction",
            "baseline_oof_status",
            "baseline_best_epoch",
            "baseline_inner_fold",
            "baseline_fold_seed",
        ]
        combined = pd.concat(predictions, ignore_index=True)[["example_id", *columns]]
        if combined["example_id"].duplicated().any():
            raise RuntimeError("duplicate examples while updating the site master")
        master = master.set_index("example_id")
        combined = combined.set_index("example_id")
        missing = combined.index.difference(master.index)
        if len(missing):
            raise RuntimeError(f"OOF examples absent from site master: {missing[:5].tolist()}")
        for column in columns:
            if column not in master.columns:
                master[column] = pd.NA
            master.loc[combined.index, column] = combined[column]
        master = master.reset_index()
        temporary = site_master_path.with_name(f".{site_master_path.name}.tmp-{os.getpid()}")
        master.to_parquet(temporary, index=False)
        os.replace(temporary, site_master_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument("--contexts", nargs="+", choices=CONTEXTS, default=list(CONTEXTS))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/runs/baseline_oof")
    )
    args = parser.parse_args()
    repository = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repository / path

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    site_master_path = resolve(args.site_master)
    output_dir = resolve(args.output_dir)
    all_predictions, all_metrics = [], []
    for context in args.contexts:
        predictions, metrics = train_context(
            repository,
            site_master_path,
            context,
            args.epochs,
            args.batch_size,
            args.checkpoint_every,
            args.seed,
            device,
            output_dir,
        )
        all_predictions.append(predictions)
        all_metrics.append(metrics)
        register(
            repository,
            {
                "run_id": f"baseline_oof_{context}_seed{args.seed}_epochs{args.epochs}",
                "experiment_type": "baseline_grouped_oof",
                "context": context,
                "input_hashes": {"site_master": metrics["site_master_sha256"]},
                "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
                "seed": args.seed,
                "features": "upstream compact Baseline GAT node/edge representation",
                "hyperparameters": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "optimizer": "Adam(lr=1e-3,weight_decay=1e-4)",
                },
                "artifact": str(
                    (output_dir / context / "oof_predictions.parquet").relative_to(
                        repository
                    )
                ),
                "metrics": metrics,
                "canonical_test_access": False,
            },
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
    update_site_master(site_master_path, all_predictions)
    metrics_table = pd.DataFrame(all_metrics)
    metrics_table.to_csv(output_dir / "oof_metrics.csv", index=False)


if __name__ == "__main__":
    main()
