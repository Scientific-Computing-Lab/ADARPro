"""Pair-balanced expanded-supervision multi-label tissue GAT."""

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
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from adarpro.models.baseline_oof import (
    atomic_save,
    choose_threshold,
    configure,
    metric_row,
    restore_rng_state,
    rng_state,
    torch_load,
    upstream_modules,
)
from adarpro.models.joint_expanded_oof import TISSUES, sample_indices
from adarpro.quarantine import sha256_file
from adarpro.registry import register

MODE = "joint_multilabel_pair_balanced"


class JointMultiLabelGAT(nn.Module):
    """Compact AdarEdit encoder with one output for each real tissue."""

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
        return self.output(global_mean_pool(nodes, data.batch))


class WeightedSiteDataset(Dataset):
    """Unique graph views with tissue-specific sampled loss weights."""

    def __init__(
        self,
        graphs: list,
        graph_indices: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
    ):
        self.graphs = graphs
        self.graph_indices = graph_indices
        self.labels = labels
        self.weights = weights

    def __len__(self) -> int:
        return len(self.graph_indices)

    def __getitem__(self, index: int):
        graph_index = int(self.graph_indices[index])
        graph = self.graphs[graph_index].clone()
        graph.y_multilabel = torch.tensor(
            self.labels[graph_index][None, :], dtype=torch.float32
        )
        graph.y_weight = torch.tensor(self.weights[index][None, :], dtype=torch.float32)
        return graph


def validate_site_identity(table: pd.DataFrame) -> None:
    identity = ["sequence", "structure", "local_position", "pair_id"]
    counts = table.groupby("pair_local_key")[identity].nunique()
    if (counts > 1).any().any():
        raise RuntimeError("cross-tissue site has inconsistent intrinsic identity")


def build_multilabel_training_table(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Return unique sites, binary rows, labels, and row-to-site/context maps."""

    binary = raw[(raw["editing_index"] < 1) | (raw["editing_index"] >= 15)].copy()
    binary = binary.reset_index(drop=True)
    validate_site_identity(binary)
    if binary.duplicated(["pair_local_key", "context"]).any():
        raise RuntimeError("duplicate site/context binary observation")
    sites = binary.drop_duplicates("pair_local_key").reset_index(drop=True)
    site_lookup = pd.Series(sites.index, index=sites["pair_local_key"])
    context_lookup = {name: index for index, name in enumerate(TISSUES)}
    row_graph_indices = binary["pair_local_key"].map(site_lookup).to_numpy(np.int64)
    row_context_indices = binary["context"].map(context_lookup).to_numpy(np.int64)
    labels = np.zeros((len(sites), len(TISSUES)), dtype=np.float32)
    labels[row_graph_indices, row_context_indices] = (
        binary["editing_index"].to_numpy() >= 15
    ).astype(np.float32)
    return sites, binary, labels, row_graph_indices, row_context_indices


def aggregate_sample_weights(
    selected_rows: np.ndarray,
    row_graph_indices: np.ndarray,
    row_context_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse repeated sampled site/context rows into exact loss weights."""

    graph_indices, inverse = np.unique(
        row_graph_indices[selected_rows], return_inverse=True
    )
    weights = np.zeros((len(graph_indices), len(TISSUES)), dtype=np.float32)
    np.add.at(weights, (inverse, row_context_indices[selected_rows]), 1.0)
    if not np.isclose(weights.sum(), len(selected_rows)):
        raise RuntimeError("multi-label sample aggregation changed update weight")
    return graph_indices, weights


def build_graphs(sites: pd.DataFrame, baseline) -> list:
    graphs = []
    for row in sites.itertuples(index=False):
        graph = baseline.create_rna_graph(
            row.sequence, row.structure, int(row.local_position), 0
        )
        if graph is None:
            raise RuntimeError(f"unbuildable joint site: {row.pair_local_key}")
        graphs.append(graph)
    return graphs


def append_history(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_epoch(
    model: JointMultiLabelGAT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        labels = batch.y_multilabel.view(-1, len(TISSUES))
        weights = batch.y_weight.view(-1, len(TISSUES))
        elementwise = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        loss = (elementwise * weights).sum() / weights.sum().clamp_min(1.0)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite joint multi-label loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict_graphs(
    model: JointMultiLabelGAT,
    graphs: list,
    graph_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = []
    loader = DataLoader(
        [graphs[int(index)] for index in graph_indices],
        batch_size=batch_size,
        shuffle=False,
    )
    for batch in loader:
        outputs.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(outputs)


def row_probabilities(
    table: pd.DataFrame,
    graph_lookup: pd.Series,
    graph_indices: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    local_lookup = pd.Series(np.arange(len(graph_indices)), index=graph_indices)
    row_graph = table["pair_local_key"].map(graph_lookup).to_numpy(np.int64)
    local_graph = pd.Series(row_graph).map(local_lookup).to_numpy()
    if pd.isna(local_graph).any():
        raise RuntimeError("canonical row missing from predicted graph subset")
    context_lookup = {name: index for index, name in enumerate(TISSUES)}
    row_context = table["context"].map(context_lookup).to_numpy(np.int64)
    return probabilities[local_graph.astype(np.int64), row_context]


def select_thresholds(
    labels: np.ndarray, probabilities: np.ndarray, contexts: pd.Series
) -> tuple[dict[str, float], float]:
    thresholds, scores = {}, []
    for context in TISSUES:
        mask = contexts.to_numpy() == context
        threshold, score = choose_threshold(labels[mask], probabilities[mask])
        thresholds[context] = threshold
        scores.append(score)
    return thresholds, float(np.mean(scores))


def run_fold(
    canonical: pd.DataFrame,
    raw_binary: pd.DataFrame,
    graphs: list,
    labels: np.ndarray,
    raw_graph_indices: np.ndarray,
    raw_context_indices: np.ndarray,
    graph_lookup: pd.Series,
    outer_fold: int,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    checkpoint_every: int,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    inner_fold = (outer_fold + 1) % 5
    train_mask = ~raw_binary["development_fold"].isin([outer_fold, inner_fold])
    eligible = np.flatnonzero(train_mask.to_numpy())
    selection = canonical[canonical["development_fold"] == inner_fold].copy()
    outer = canonical[canonical["development_fold"] == outer_fold].copy()
    train_pairs = set(raw_binary.loc[train_mask, "pair_id"])
    if train_pairs & set(selection["pair_id"]) or train_pairs & set(outer["pair_id"]):
        raise RuntimeError("pair leakage in joint multi-label fold")

    canonical_train = canonical[
        ~canonical["development_fold"].isin([outer_fold, inner_fold])
    ]
    per_tissue_samples = {
        context: int((canonical_train["context"] == context).sum())
        for context in TISSUES
    }
    selection_graph_indices = np.sort(
        selection["pair_local_key"].map(graph_lookup).unique().astype(np.int64)
    )
    outer_graph_indices = np.sort(
        outer["pair_local_key"].map(graph_lookup).unique().astype(np.int64)
    )
    fold_dir = output_dir / MODE / f"outer_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": MODE,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "epochs": epochs,
        "batch_size": batch_size,
        "per_tissue_samples": per_tissue_samples,
        "total_label_weight_per_epoch": sum(per_tissue_samples.values()),
        "sampler": "per tissue/state: uniform pair then uniform site; repeated graphs collapsed to exact loss weights",
        "selection": "macro F1 across four individual tissues on inner fold",
        "canonical_test_access": False,
    }
    config_path = fold_dir / "config.json"
    serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if config_path.exists() and config_path.read_text() != serialized:
        raise RuntimeError(f"refusing incompatible resume: {config_path}")
    config_path.write_text(serialized)

    fold_seed = seed + outer_fold
    configure(fold_seed, threads=8)
    sample_rng = np.random.default_rng(fold_seed + 700_000)
    model = JointMultiLabelGAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_path = fold_dir / "best.pth"
    last_path = fold_dir / "last.pth"
    history_path = fold_dir / "history.csv"
    best = {"macro_f1": -1.0, "epoch": 0, "thresholds": None}
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
        selected_rows = sample_indices(
            raw_binary,
            eligible,
            per_tissue_samples,
            True,
            sample_rng,
        )
        graph_indices, weights = aggregate_sample_weights(
            selected_rows, raw_graph_indices, raw_context_indices
        )
        loader = DataLoader(
            WeightedSiteDataset(graphs, graph_indices, labels, weights),
            batch_size=batch_size,
            shuffle=False,
        )
        loss = train_epoch(model, loader, optimizer, device)
        selection_matrix = predict_graphs(
            model, graphs, selection_graph_indices, batch_size, device
        )
        selection_probability = row_probabilities(
            selection,
            graph_lookup,
            selection_graph_indices,
            selection_matrix,
        )
        thresholds, macro_f1 = select_thresholds(
            selection["label"].to_numpy(),
            selection_probability,
            selection["context"],
        )
        if macro_f1 > best["macro_f1"] + 1e-6:
            best = {
                "macro_f1": macro_f1,
                "epoch": epoch,
                "thresholds": thresholds,
            }
            atomic_save(
                {"model_state": model.state_dict(), "best": best, "config": config},
                best_path,
            )
        append_history(
            history_path,
            {
                "epoch": epoch,
                "loss": loss,
                "selection_macro_f1": macro_f1,
                "sampled_unique_graphs": len(graph_indices),
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
                f"[{MODE} outer={outer_fold}] {epoch}/{epochs} loss={loss:.5f} "
                f"macro_f1={macro_f1:.5f} unique_graphs={len(graph_indices)} "
                f"best={best['epoch']}",
                flush=True,
            )

    saved = torch_load(best_path, device)
    model.load_state_dict(saved["model_state"])
    outer_matrix = predict_graphs(
        model, graphs, outer_graph_indices, batch_size, device
    )
    outer_probability = row_probabilities(
        outer, graph_lookup, outer_graph_indices, outer_matrix
    )
    outer["mode"] = MODE
    outer["probability"] = outer_probability
    outer["threshold"] = outer["context"].map(saved["best"]["thresholds"])
    outer["prediction"] = (outer["probability"] >= outer["threshold"]).astype(int)
    outer["best_epoch"] = int(saved["best"]["epoch"])
    columns = [
        "example_id",
        "context",
        "pair_id",
        "pair_local_key",
        "label",
        "development_fold",
        "baseline_oof_probability",
        "baseline_oof_prediction",
        "mode",
        "probability",
        "threshold",
        "prediction",
        "best_epoch",
    ]
    outer = outer[columns].sort_values("example_id").reset_index(drop=True)
    outer.to_parquet(fold_dir / "outer_predictions.parquet", index=False)
    return outer


def run(
    repository: Path,
    site_master_path: Path,
    raw_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    checkpoint_every: int,
    seed: int,
    device: torch.device,
    folds: tuple[int, ...] = tuple(range(5)),
) -> dict:
    if not folds or len(set(folds)) != len(folds) or set(folds) - set(range(5)):
        raise ValueError("folds must be a nonempty unique subset of 0..4")
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("joint multi-label site master contains a sealed split")
    canonical = master[master["context"].isin(TISSUES)].copy().reset_index(drop=True)
    raw = pd.read_parquet(raw_path)
    raw = raw[raw["context"].isin(TISSUES)].copy().reset_index(drop=True)
    if set(raw["development_fold"].unique()) != set(range(5)):
        raise RuntimeError("raw joint table lacks five development folds")
    sites, binary, labels, raw_graph_indices, raw_context_indices = (
        build_multilabel_training_table(raw)
    )
    canonical_missing = set(canonical["pair_local_key"]) - set(sites["pair_local_key"])
    if canonical_missing:
        raise RuntimeError(
            f"{len(canonical_missing)} canonical sites absent from binary raw pool"
        )
    graph_lookup = pd.Series(sites.index, index=sites["pair_local_key"])
    print(
        f"[{MODE}] building {len(sites)} unique graphs for {len(binary)} "
        f"high/low tissue observations",
        flush=True,
    )
    graphs = build_graphs(sites, upstream_modules(repository))
    outputs = [
        run_fold(
            canonical,
            binary,
            graphs,
            labels,
            raw_graph_indices,
            raw_context_indices,
            graph_lookup,
            fold,
            output_dir,
            epochs,
            batch_size,
            checkpoint_every,
            seed,
            device,
        )
        for fold in folds
    ]
    oof = pd.concat(outputs, ignore_index=True)
    destination = output_dir / MODE
    destination.mkdir(parents=True, exist_ok=True)
    if set(folds) != set(range(5)):
        partial_path = destination / "partial_oof_predictions.parquet"
        oof.to_parquet(partial_path, index=False)
        summary = {
            "scope": "partial development smoke/futility run",
            "completed_folds": list(folds),
            "canonical_rows": len(oof),
            "epochs": epochs,
            "canonical_test_access": False,
        }
        (destination / "partial_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return summary
    oof_path = destination / "oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)
    by_tissue = {}
    for context in TISSUES:
        subset = oof[oof["context"] == context]
        by_tissue[context] = {
            "candidate": metric_row(
                subset["label"], subset["probability"], subset["prediction"]
            )
        }
        if subset["baseline_oof_probability"].notna().all():
            by_tissue[context]["development_reference"] = metric_row(
                subset["label"],
                subset["baseline_oof_probability"],
                subset["baseline_oof_prediction"],
            )
    macro = {
        metric: float(
            np.mean([by_tissue[context]["candidate"][metric] for context in TISSUES])
        )
        for metric in ("auroc", "auprc", "f1")
    }
    summary = {
        "scope": "four individual tissues; expanded high/low grouped OOF",
        "mode": MODE,
        "canonical_rows": len(oof),
        "raw_binary_rows": len(binary),
        "unique_graphs": len(graphs),
        "pairs": int(canonical["pair_id"].nunique()),
        "epochs": epochs,
        "candidate_macro": macro,
        "by_tissue": by_tissue,
        "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"{MODE}_oof_seed{seed}_epochs{epochs}",
            "experiment_type": "expanded_pair_balanced_joint_multilabel_grouped_oof",
            "context": "four_individual_tissues",
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "raw_development_sites": sha256_file(raw_path),
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": "exact compact GAT graph; four masked tissue outputs",
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "sampling": "pair-balanced within every tissue/high-low stratum; duplicate graph draws collapsed to exact loss weights",
                "selection": "four-tissue inner-fold macro F1",
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/joint_multilabel"),
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
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
                resolve(args.output_dir),
                args.epochs,
                args.batch_size,
                args.checkpoint_every,
                args.seed,
                device,
                tuple(args.folds),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
