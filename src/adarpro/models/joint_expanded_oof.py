"""Cross-tissue shared GAT trained on expanded high/low development sites."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from adarpro.models.baseline_oof import (
    atomic_save,
    build_graphs,
    choose_threshold,
    configure,
    metric_row,
    restore_rng_state,
    rng_state,
    torch_load,
    upstream_modules,
)
from adarpro.models.multitask_oof import bootstrap_delta
from adarpro.models.residual import matched_recall_threshold
from adarpro.quarantine import sha256_file
from adarpro.registry import register

TISSUES = ("Artery", "Brain", "Liver", "MuscleSkeletal")
MODES = ("joint_binary_expanded", "joint_binary_pair_balanced")
BUDGET_POLICIES = ("muscle_total", "canonical_per_tissue")


class JointContextGAT(nn.Module):
    """Checkpoint-independent compact GAT with one output per real tissue."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, heads: int = 4):
        super().__init__()
        self.gat_layers = nn.ModuleList(
            [
                GATConv(input_dim, hidden_dim, heads=heads, concat=True),
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True),
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True),
            ]
        )
        self.batch_norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim * heads) for _ in range(3)]
        )
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden_dim * heads, len(TISSUES))

    def forward(self, data) -> torch.Tensor:
        nodes = data.x
        for layer, normalization in zip(self.gat_layers, self.batch_norm_layers):
            nodes = self.dropout(normalization(F.relu(layer(nodes, data.edge_index))))
        logits = self.output(global_mean_pool(nodes, data.batch))
        return logits.gather(1, data.context_index.view(-1, 1)).view(-1)


class ContextRowDataset(Dataset):
    def __init__(
        self,
        graphs: list,
        graph_indices: np.ndarray,
        labels: np.ndarray,
        context_indices: np.ndarray,
    ):
        self.graphs = graphs
        self.graph_indices = graph_indices
        self.labels = labels
        self.context_indices = context_indices

    def __len__(self) -> int:
        return len(self.graph_indices)

    def __getitem__(self, index: int):
        graph = self.graphs[int(self.graph_indices[index])].clone()
        graph.y = torch.tensor([float(self.labels[index])], dtype=torch.float32)
        graph.context_index = torch.tensor(
            [int(self.context_indices[index])], dtype=torch.long
        )
        return graph


def build_raw_dataset(
    raw: pd.DataFrame, baseline
) -> tuple[ContextRowDataset, pd.DataFrame]:
    binary = raw[(raw["editing_index"] < 1) | (raw["editing_index"] >= 15)].copy()
    identity = ["pair_local_key", "sequence", "structure", "local_position"]
    uniqueness = binary.groupby("pair_local_key")[identity[1:]].nunique()
    if (uniqueness > 1).any().any():
        raise RuntimeError(
            "cross-tissue site has inconsistent intrinsic representation"
        )
    sites = binary.drop_duplicates("pair_local_key").reset_index(drop=True)
    graphs = []
    for row in sites.itertuples(index=False):
        graph = baseline.create_rna_graph(
            row.sequence, row.structure, int(row.local_position), 0
        )
        if graph is None:
            raise RuntimeError(f"unbuildable raw joint graph: {row.pair_local_key}")
        graphs.append(graph)
    graph_lookup = pd.Series(sites.index, index=sites["pair_local_key"])
    graph_indices = binary["pair_local_key"].map(graph_lookup).to_numpy(dtype=np.int64)
    labels = (binary["editing_index"].to_numpy() >= 15).astype(np.float32)
    context_lookup = {name: index for index, name in enumerate(TISSUES)}
    context_indices = binary["context"].map(context_lookup).to_numpy(dtype=np.int64)
    return ContextRowDataset(
        graphs, graph_indices, labels, context_indices
    ), binary.reset_index(drop=True)


def attach_context(graphs: list, table: pd.DataFrame) -> None:
    lookup = {name: index for index, name in enumerate(TISSUES)}
    for graph, context in zip(graphs, table["context"]):
        graph.context_index = torch.tensor([lookup[context]], dtype=torch.long)


def sample_indices(
    table: pd.DataFrame,
    eligible: np.ndarray,
    per_tissue_samples: int | dict[str, int],
    pair_balanced: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = []
    states = table["editing_index"].to_numpy() >= 15
    contexts = table["context"].to_numpy()
    pairs = table["pair_id"].to_numpy()
    for context in TISSUES:
        context_samples = (
            per_tissue_samples[context]
            if isinstance(per_tissue_samples, dict)
            else per_tissue_samples
        )
        counts = [context_samples // 2, context_samples - context_samples // 2]
        for state, count in zip((False, True), counts):
            available = eligible[
                (contexts[eligible] == context) & (states[eligible] == state)
            ]
            if not len(available):
                raise RuntimeError(f"empty joint sampling stratum: {context}/{state}")
            if not pair_balanced:
                selected.append(
                    rng.choice(available, size=count, replace=count > len(available))
                )
                continue
            available_pairs = pairs[available]
            unique_pairs = np.unique(available_pairs)
            sampled_pairs = rng.choice(unique_pairs, size=count, replace=True)
            by_pair = {
                pair: available[available_pairs == pair] for pair in unique_pairs
            }
            selected.append(
                np.asarray([rng.choice(by_pair[pair]) for pair in sampled_pairs])
            )
    return rng.permutation(np.concatenate(selected))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        probabilities.extend(torch.sigmoid(model(batch)).cpu().tolist())
        labels.extend(batch.y.view(-1).int().cpu().tolist())
    return np.asarray(probabilities), np.asarray(labels)


def append_history(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_fold(
    canonical: pd.DataFrame,
    canonical_all: pd.DataFrame,
    canonical_graphs: list,
    raw: pd.DataFrame,
    raw_dataset: ContextRowDataset,
    mode: str,
    budget_policy: str,
    outer_fold: int,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    checkpoint_every: int,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    inner_fold = (outer_fold + 1) % 5
    train_pair_mask = ~raw["development_fold"].isin([outer_fold, inner_fold])
    eligible = np.flatnonzero(train_pair_mask.to_numpy())
    selection_mask = canonical["development_fold"] == inner_fold
    outer_mask = canonical["development_fold"] == outer_fold
    train_pairs = set(raw.loc[train_pair_mask, "pair_id"])
    selection_pairs = set(canonical.loc[selection_mask, "pair_id"])
    outer_pairs = set(canonical.loc[outer_mask, "pair_id"])
    if (
        train_pairs & selection_pairs
        or train_pairs & outer_pairs
        or selection_pairs & outer_pairs
    ):
        raise RuntimeError("pair leakage in cross-tissue shared fold")
    selection_indices = np.flatnonzero(selection_mask.to_numpy())
    outer_indices = np.flatnonzero(outer_mask.to_numpy())
    muscle_epoch_samples = int(
        (~canonical["development_fold"].isin([outer_fold, inner_fold])).sum()
    )
    if budget_policy == "muscle_total":
        per_tissue_samples: int | dict[str, int] = muscle_epoch_samples // len(TISSUES)
        per_tissue_sample_counts = {context: per_tissue_samples for context in TISSUES}
    elif budget_policy == "canonical_per_tissue":
        train_canonical = canonical_all[
            ~canonical_all["development_fold"].isin([outer_fold, inner_fold])
        ]
        per_tissue_sample_counts = {
            context: int((train_canonical["context"] == context).sum())
            for context in TISSUES
        }
        per_tissue_samples = per_tissue_sample_counts
    else:
        raise ValueError(f"unknown budget policy: {budget_policy}")
    total_epoch_samples = sum(per_tissue_sample_counts.values())
    run_name = f"{mode}_{budget_policy}"
    fold_dir = output_dir / run_name / "MuscleSkeletal" / f"outer_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": mode,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "epochs": epochs,
        "batch_size": batch_size,
        "budget_policy": budget_policy,
        "per_tissue_samples": per_tissue_sample_counts,
        "total_epoch_samples": total_epoch_samples,
        "tissues": TISSUES,
        "selection": "Muscle canonical inner-fold F1 and threshold",
        "sampler": (
            "equal tissue and binary state; uniform pair then site"
            if mode.endswith("pair_balanced")
            else "equal tissue and binary state; uniform site"
        ),
        "canonical_test_access": False,
    }
    config_path = fold_dir / "config.json"
    serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if config_path.exists() and config_path.read_text() != serialized:
        raise RuntimeError(f"refusing incompatible resume: {config_path}")
    config_path.write_text(serialized)
    selection_loader = DataLoader(
        [canonical_graphs[index] for index in selection_indices],
        batch_size=batch_size,
        shuffle=False,
    )
    outer_loader = DataLoader(
        [canonical_graphs[index] for index in outer_indices],
        batch_size=batch_size,
        shuffle=False,
    )
    # Keep encoder initialization comparable across site- and pair-balanced runs.
    fold_seed = seed + outer_fold
    configure(fold_seed, threads=8)
    sample_rng = np.random.default_rng(fold_seed + 700_000)
    model = JointContextGAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_path = fold_dir / "best.pth"
    last_path = fold_dir / "last.pth"
    history_path = fold_dir / "history.csv"
    best = {"f1": -1.0, "epoch": 0, "threshold": None}
    start_epoch = 1
    if last_path.exists():
        saved = torch_load(last_path, device)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng_state(saved["rng_state"])
        sample_rng.bit_generator.state = saved["sample_rng_state"]
        best = saved["best"]
        start_epoch = int(saved["epoch"]) + 1
    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        sampled = sample_indices(
            raw,
            eligible,
            per_tissue_samples,
            mode.endswith("pair_balanced"),
            sample_rng,
        )
        loader = DataLoader(
            Subset(raw_dataset, sampled.tolist()),
            batch_size=batch_size,
            shuffle=False,
        )
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probability, labels = predict(model, selection_loader, device)
        threshold, selection_f1 = choose_threshold(labels, probability)
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
                "loss": float(np.mean(losses)),
                "selection_f1": selection_f1,
                "selection_auroc": roc_auc_score(labels, probability),
                "selection_auprc": average_precision_score(labels, probability),
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
                    "sample_rng_state": sample_rng.bit_generator.state,
                    "best": best,
                    "config": config,
                },
                last_path,
            )
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"[{mode} MuscleSkeletal outer={outer_fold}] {epoch}/{epochs} "
                f"loss={np.mean(losses):.5f} selection_f1={selection_f1:.5f} "
                f"best={best['epoch']}",
                flush=True,
            )
    saved = torch_load(best_path, device)
    model.load_state_dict(saved["model_state"])
    selection_probability, selection_labels = predict(model, selection_loader, device)
    baseline_selection_recall = recall_score(
        selection_labels,
        canonical.loc[selection_mask, "baseline_oof_prediction"],
        zero_division=0,
    )
    recall_threshold = matched_recall_threshold(
        selection_labels, selection_probability, baseline_selection_recall
    )
    probability, labels = predict(model, outer_loader, device)
    threshold = float(saved["best"]["threshold"])
    outer = canonical.loc[
        outer_mask,
        [
            "example_id",
            "context",
            "pair_id",
            "pair_local_key",
            "label",
            "development_fold",
            "baseline_oof_probability",
            "baseline_oof_prediction",
        ],
    ].copy()
    if not np.array_equal(outer["label"].to_numpy(), labels):
        raise RuntimeError("joint outer prediction order mismatch")
    outer["mode"] = mode
    outer["probability"] = probability
    outer["threshold"] = threshold
    outer["prediction"] = (probability >= threshold).astype(int)
    outer["matched_recall_threshold"] = recall_threshold
    outer["matched_recall_prediction"] = (probability >= recall_threshold).astype(int)
    outer["inner_baseline_recall"] = baseline_selection_recall
    outer["best_epoch"] = int(saved["best"]["epoch"])
    outer.to_parquet(fold_dir / "outer_predictions.parquet", index=False)
    return outer


def run(
    repository: Path,
    site_master_path: Path,
    raw_path: Path,
    mode: str,
    budget_policy: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    checkpoint_every: int,
    bootstrap_replicates: int,
    seed: int,
    device: torch.device,
) -> dict:
    baseline = upstream_modules(repository)
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("joint-expanded site master contains a sealed split")
    canonical_all = (
        master[master["context"].isin(TISSUES)].copy().reset_index(drop=True)
    )
    canonical = (
        canonical_all[canonical_all["context"] == "MuscleSkeletal"]
        .copy()
        .reset_index(drop=True)
    )
    if canonical["baseline_oof_probability"].isna().any():
        raise RuntimeError("Muscle development reference is incomplete")
    canonical_graphs = build_graphs(canonical, baseline)
    attach_context(canonical_graphs, canonical)
    raw = pd.read_parquet(raw_path)
    raw = raw[raw["context"].isin(TISSUES)].copy().reset_index(drop=True)
    if set(raw["development_fold"].unique()) != set(range(5)):
        raise RuntimeError("raw cross-tissue table lacks five development folds")
    raw_dataset, binary = build_raw_dataset(raw, baseline)
    print(
        f"[{mode}] {len(binary)} high/low tissue rows; "
        f"{len(raw_dataset.graphs)} unique site graphs",
        flush=True,
    )
    predictions = [
        run_fold(
            canonical,
            canonical_all,
            canonical_graphs,
            binary,
            raw_dataset,
            mode,
            budget_policy,
            fold,
            output_dir,
            epochs,
            batch_size,
            checkpoint_every,
            seed,
            device,
        )
        for fold in range(5)
    ]
    oof = pd.concat(predictions, ignore_index=True)
    run_name = f"{mode}_{budget_policy}"
    destination = output_dir / run_name / "MuscleSkeletal"
    oof_path = destination / "oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)
    candidate_metrics = metric_row(oof["label"], oof["probability"], oof["prediction"])
    baseline_metrics = metric_row(
        oof["label"], oof["baseline_oof_probability"], oof["baseline_oof_prediction"]
    )
    bootstrap = bootstrap_delta(oof, bootstrap_replicates, seed + 2000)
    bootstrap.to_parquet(destination / "pair_bootstrap.parquet", index=False)
    confidence = (
        bootstrap.groupby("metric")["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    confidence.to_csv(destination / "pair_bootstrap_ci.csv", index=False)
    matched_oof = oof.copy()
    matched_oof["prediction"] = matched_oof["matched_recall_prediction"]
    matched_bootstrap = bootstrap_delta(matched_oof, bootstrap_replicates, seed + 3000)
    matched_bootstrap.to_parquet(
        destination / "pair_bootstrap_matched_recall.parquet", index=False
    )
    matched_confidence = (
        matched_bootstrap.groupby("metric")["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    matched_confidence.to_csv(
        destination / "pair_bootstrap_matched_recall_ci.csv", index=False
    )
    summary = {
        "scope": "Muscle canonical train+valid OOF; raw high/low training from four individual tissues",
        "mode": mode,
        "budget_policy": budget_policy,
        "canonical_rows": len(canonical),
        "raw_training_pool_rows": len(binary),
        "unique_training_site_graphs": len(raw_dataset.graphs),
        "pairs": int(canonical["pair_id"].nunique()),
        "epochs": epochs,
        "candidate_metrics": candidate_metrics,
        "candidate_matched_recall_metrics": metric_row(
            oof["label"], oof["probability"], oof["matched_recall_prediction"]
        ),
        "development_reference_metrics": baseline_metrics,
        "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"{mode}_{budget_policy}_oof_MuscleSkeletal_seed{seed}_epochs{epochs}",
            "experiment_type": "expanded_cross_tissue_shared_grouped_oof",
            "context": "MuscleSkeletal",
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "raw_development_sites": sha256_file(raw_path),
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": "exact compact GAT graph with four tissue-specific output heads",
            "hyperparameters": {
                "mode": mode,
                "budget_policy": budget_policy,
                "epochs": epochs,
                "batch_size": batch_size,
                "total_update_budget": (
                    "one canonical-training-row budget per tissue"
                    if budget_policy == "canonical_per_tissue"
                    else "Muscle canonical training rows, divided equally across tissues"
                ),
            },
            "artifact": str(oof_path.relative_to(repository)),
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
    parser.add_argument("--mode", choices=MODES, default="joint_binary_expanded")
    parser.add_argument(
        "--budget-policy", choices=BUDGET_POLICIES, default="muscle_total"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/runs/richer_supervision")
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=25)
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
                args.mode,
                args.budget_policy,
                resolve(args.output_dir),
                args.epochs,
                args.batch_size,
                args.checkpoint_every,
                args.bootstrap_replicates,
                args.seed,
                device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
