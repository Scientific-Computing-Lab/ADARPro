"""Pair-grouped OOF substrate-wide and hierarchical editing models."""

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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from adarpro.models.baseline_oof import (
    CONTEXTS,
    atomic_save,
    choose_threshold,
    configure,
    metric_row,
    restore_rng_state,
    rng_state,
    torch_load,
    upstream_modules,
)
from adarpro.models.multitask_oof import bootstrap_delta
from adarpro.quarantine import sha256_file
from adarpro.registry import register

MODES = ("joint_binary", "joint_multitask", "hierarchical_multitask")


def paired_positions(structure: str) -> np.ndarray:
    partners = np.full(len(structure), -1, dtype=np.int64)
    stack = []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")" and stack:
            partner = stack.pop()
            partners[index] = partner
            partners[partner] = index
    return partners


def substrate_node_features(sequence: str, paired: np.ndarray) -> torch.Tensor:
    base = {
        "A": [1, 0, 0, 0, 0],
        "G": [0, 1, 0, 0, 0],
        "C": [0, 0, 1, 0, 0],
        "T": [0, 0, 0, 1, 0],
        "U": [0, 0, 0, 1, 0],
        "N": [0, 0, 0, 0, 1],
    }
    length = len(sequence)
    normalized = np.linspace(-1.0, 1.0, length) if length > 1 else np.zeros(1)
    separator = np.flatnonzero(np.asarray(list(sequence)) == "N")
    arm = np.zeros(length, dtype=float)
    if len(separator):
        arm[: separator[0]] = -1.0
        arm[separator[-1] + 1 :] = 1.0
    values = [
        base.get(symbol.upper(), base["N"])
        + [float(paired[index] >= 0), float(normalized[index]), float(arm[index])]
        for index, symbol in enumerate(sequence)
    ]
    return torch.tensor(values, dtype=torch.float32)


def build_substrate_graphs(
    table: pd.DataFrame, baseline, *, raw_labels: bool
) -> list[Data]:
    graphs = []
    for pair_id, group in table.groupby("pair_id", sort=True):
        if group["sequence"].nunique() != 1 or group["structure"].nunique() != 1:
            raise RuntimeError(f"non-unique substrate representation for pair {pair_id}")
        sequence = group["sequence"].iloc[0]
        structure = group["structure"].iloc[0]
        template = baseline.create_rna_graph(sequence, structure, 0, 0)
        if template is None:
            raise RuntimeError(f"unbuildable substrate graph for pair {pair_id}")
        partner = paired_positions(structure)
        site_index = group["local_position"].to_numpy(dtype=np.int64)
        if np.any(np.asarray(list(sequence))[site_index] != "A"):
            raise RuntimeError(f"non-A target in substrate {pair_id}")
        partner_index = partner[site_index].copy()
        partner_index[partner_index < 0] = site_index[partner_index < 0]
        graph = Data(
            x=substrate_node_features(sequence, partner),
            edge_index=template.edge_index,
            site_index=torch.tensor(site_index, dtype=torch.long),
            partner_index=torch.tensor(partner_index, dtype=torch.long),
            site_paired=torch.tensor(partner[site_index] >= 0, dtype=torch.float32),
            pair_id_tensor=torch.tensor([int(pair_id)], dtype=torch.long),
        )
        if raw_labels:
            editing = group["editing_index"].to_numpy(dtype=float)
            ordinal = np.where(editing < 1, 0, np.where(editing >= 15, 2, 1))
            graph.ordinal_label = torch.tensor(ordinal, dtype=torch.long)
            graph.editing_fraction = torch.tensor(editing / 100, dtype=torch.float32)
            graph.binary_mask = torch.tensor(ordinal != 1, dtype=torch.bool)
            graph.binary_label = torch.tensor(ordinal == 2, dtype=torch.float32)
            graph.engaged_label = torch.tensor(
                [float(np.any(ordinal == 2))], dtype=torch.float32
            )
        else:
            graph.binary_mask = torch.ones(len(group), dtype=torch.bool)
            graph.binary_label = torch.tensor(
                group["label"].to_numpy(dtype=float), dtype=torch.float32
            )
            graph.ordinal_label = torch.tensor(
                2 * group["label"].to_numpy(dtype=np.int64), dtype=torch.long
            )
            graph.editing_fraction = graph.binary_label.clone()
            graph.engaged_label = torch.tensor(
                [float(group["label"].max())], dtype=torch.float32
            )
            graph.row_id = torch.tensor(group.index.to_numpy(), dtype=torch.long)
        graphs.append(graph)
    return graphs


class SubstrateGAT(nn.Module):
    def __init__(self, hierarchical: bool):
        super().__init__()
        self.hierarchical = hierarchical
        hidden, heads = 32, 4
        self.layers = nn.ModuleList(
            [
                GATConv(8, hidden, heads=heads),
                GATConv(hidden * heads, hidden, heads=heads),
                GATConv(hidden * heads, hidden, heads=heads),
            ]
        )
        self.normalization = nn.ModuleList(
            [nn.BatchNorm1d(hidden * heads) for _ in range(3)]
        )
        self.dropout = nn.Dropout(0.2)
        representation = 3 * hidden * heads + 1
        self.site_projection = nn.Sequential(
            nn.Linear(representation, 128), nn.ReLU(), nn.Dropout(0.2)
        )
        self.chemical_head = nn.Linear(128, 1)
        self.ordinal_head = nn.Linear(128, 3)
        self.continuous_head = nn.Linear(128, 1)
        self.engagement_head = nn.Sequential(
            nn.Linear(hidden * heads, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, data):
        nodes = data.x
        for layer, normalization in zip(self.layers, self.normalization):
            nodes = self.dropout(normalization(F.relu(layer(nodes, data.edge_index))))
        substrate = global_mean_pool(nodes, data.batch)
        site_batch = data.batch[data.site_index]
        representation = torch.cat(
            [
                nodes[data.site_index],
                nodes[data.partner_index],
                substrate[site_batch],
                data.site_paired.view(-1, 1),
            ],
            dim=1,
        )
        site = self.site_projection(representation)
        chemical_logit = self.chemical_head(site).view(-1)
        engagement_logit = self.engagement_head(substrate).view(-1)
        if self.hierarchical:
            probability = torch.sigmoid(chemical_logit) * torch.sigmoid(
                engagement_logit[site_batch]
            )
        else:
            probability = torch.sigmoid(chemical_logit)
        return {
            "probability": probability,
            "chemical_logit": chemical_logit,
            "ordinal_logit": self.ordinal_head(site),
            "continuous": torch.sigmoid(self.continuous_head(site).view(-1)),
            "engagement_logit": engagement_logit,
        }


def class_weights(graphs: list[Data]) -> dict[str, torch.Tensor | float]:
    binary = torch.cat(
        [graph.binary_label[graph.binary_mask] for graph in graphs]
    ).numpy()
    ordinal = torch.cat([graph.ordinal_label for graph in graphs]).numpy()
    negative, positive = (binary == 0).sum(), (binary == 1).sum()
    ordinal_counts = np.bincount(ordinal, minlength=3)
    ordinal_weight = len(ordinal) / (3 * np.maximum(ordinal_counts, 1))
    engagement = torch.cat([graph.engaged_label for graph in graphs]).numpy()
    engagement_negative = (engagement == 0).sum()
    engagement_positive = (engagement == 1).sum()
    return {
        "binary_positive": float(negative / max(positive, 1)),
        "ordinal": torch.tensor(ordinal_weight, dtype=torch.float32),
        "engagement_positive": float(
            engagement_negative / max(engagement_positive, 1)
        ),
    }


def train_epoch(
    model: SubstrateGAT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mode: str,
    weights: dict,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {name: [] for name in ("loss", "binary", "ordinal", "continuous", "engagement")}
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        mask = batch.binary_mask.view(-1).bool()
        labels = batch.binary_label.view(-1)[mask]
        binary_weight = torch.where(
            labels > 0,
            torch.as_tensor(weights["binary_positive"], device=device),
            torch.ones((), device=device),
        )
        binary_loss = F.binary_cross_entropy(
            output["probability"][mask], labels, weight=binary_weight
        )
        zero = torch.zeros((), device=device)
        ordinal_loss, continuous_loss, engagement_loss = zero, zero, zero
        loss = binary_loss
        if mode != "joint_binary":
            ordinal_weight = weights["ordinal"].to(device)
            ordinal_loss = F.cross_entropy(
                output["ordinal_logit"],
                batch.ordinal_label.view(-1),
                weight=ordinal_weight,
            )
            continuous_loss = F.huber_loss(
                output["probability"],
                batch.editing_fraction.view(-1),
                delta=0.1,
            )
            loss = loss + 0.5 * ordinal_loss + 0.5 * continuous_loss
        if mode == "hierarchical_multitask":
            engaged = batch.engaged_label.view(-1)
            engagement_loss = F.binary_cross_entropy_with_logits(
                output["engagement_logit"],
                engaged,
                pos_weight=torch.as_tensor(
                    weights["engagement_positive"], device=device
                ),
            )
            loss = loss + 0.25 * engagement_loss
        loss.backward()
        optimizer.step()
        for name, value in (
            ("loss", loss),
            ("binary", binary_loss),
            ("ordinal", ordinal_loss),
            ("continuous", continuous_loss),
            ("engagement", engagement_loss),
        ):
            totals[name].append(float(value.detach().cpu()))
    return {name: float(np.mean(values)) for name, values in totals.items()}


@torch.no_grad()
def predict(model: SubstrateGAT, loader: DataLoader, device: torch.device):
    model.eval()
    rows, probabilities, labels = [], [], []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        rows.extend(batch.row_id.cpu().tolist())
        probabilities.extend(output["probability"].cpu().tolist())
        labels.extend(batch.binary_label.view(-1).int().cpu().tolist())
    return np.asarray(rows), np.asarray(probabilities), np.asarray(labels)


def append_history(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_fold(
    canonical: pd.DataFrame,
    canonical_graphs: dict[int, Data],
    training: pd.DataFrame,
    training_graphs: dict[int, Data],
    context: str,
    mode: str,
    outer_fold: int,
    epochs: int,
    graph_batch_size: int,
    checkpoint_every: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    inner_fold = (outer_fold + 1) % 5
    train_pairs = set(
        training.loc[
            ~training["development_fold"].isin([outer_fold, inner_fold]), "pair_id"
        ]
    )
    selection_pairs = set(
        canonical.loc[canonical["development_fold"] == inner_fold, "pair_id"]
    )
    outer_pairs = set(
        canonical.loc[canonical["development_fold"] == outer_fold, "pair_id"]
    )
    if train_pairs & selection_pairs or train_pairs & outer_pairs or selection_pairs & outer_pairs:
        raise RuntimeError("pair leakage in substrate-wide folds")
    train_graph_list = [training_graphs[pair] for pair in sorted(train_pairs)]
    selection_graphs = [canonical_graphs[pair] for pair in sorted(selection_pairs)]
    outer_graphs = [canonical_graphs[pair] for pair in sorted(outer_pairs)]
    weights = class_weights(train_graph_list)
    fold_dir = output_dir / mode / context / f"outer_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "context": context,
        "mode": mode,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "epochs": epochs,
        "graph_batch_size": graph_batch_size,
        "training_pairs": len(train_pairs),
        "selection_pairs": len(selection_pairs),
        "outer_pairs": len(outer_pairs),
        "class_weights": {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in weights.items()
        },
        "canonical_test_access": False,
    }
    config_path = fold_dir / "config.json"
    serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if config_path.exists() and config_path.read_text() != serialized:
        raise RuntimeError(f"refusing incompatible substrate resume: {config_path}")
    config_path.write_text(serialized)
    selection_loader = DataLoader(selection_graphs, batch_size=graph_batch_size, shuffle=False)
    outer_loader = DataLoader(outer_graphs, batch_size=graph_batch_size, shuffle=False)
    fold_seed = seed + 1000 * CONTEXTS.index(context) + 100 * MODES.index(mode) + outer_fold
    configure(fold_seed, 8)
    model = SubstrateGAT(mode == "hierarchical_multitask").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_path, last_path = fold_dir / "best.pth", fold_dir / "last.pth"
    history_path = fold_dir / "history.csv"
    best = {"f1": -1.0, "epoch": 0, "threshold": None}
    start_epoch = 1
    if last_path.exists():
        saved = torch_load(last_path, device)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng_state(saved["rng_state"])
        best, start_epoch = saved["best"], int(saved["epoch"]) + 1
    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_loader = DataLoader(
            train_graph_list, batch_size=graph_batch_size, shuffle=True
        )
        losses = train_epoch(model, train_loader, optimizer, mode, weights, device)
        _, probability, labels = predict(model, selection_loader, device)
        threshold, score = choose_threshold(labels, probability)
        if score > best["f1"] + 1e-6:
            best = {"f1": score, "epoch": epoch, "threshold": threshold}
            atomic_save({"model_state": model.state_dict(), "best": best}, best_path)
        append_history(
            history_path,
            {
                "epoch": epoch,
                **losses,
                "selection_f1": score,
                "selection_auroc": roc_auc_score(labels, probability),
                "selection_auprc": average_precision_score(labels, probability),
                "threshold": threshold,
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
                },
                last_path,
            )
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"[{mode} {context} outer={outer_fold}] {epoch}/{epochs} "
                f"loss={losses['loss']:.5f} selection_f1={score:.5f} best={best['epoch']}",
                flush=True,
            )
    saved = torch_load(best_path, device)
    model.load_state_dict(saved["model_state"])
    row_id, probability, labels = predict(model, outer_loader, device)
    outer = canonical.loc[row_id].copy()
    if not np.array_equal(outer["label"].to_numpy(), labels):
        raise RuntimeError("substrate-wide outer prediction order mismatch")
    outer = outer[
        [
            "example_id",
            "context",
            "pair_id",
            "pair_local_key",
            "label",
            "development_fold",
            "baseline_oof_probability",
            "baseline_oof_prediction",
        ]
    ]
    outer["mode"] = mode
    outer["probability"] = probability
    outer["threshold"] = float(saved["best"]["threshold"])
    outer["prediction"] = (outer["probability"] >= outer["threshold"]).astype(int)
    outer["best_epoch"] = int(saved["best"]["epoch"])
    outer.to_parquet(fold_dir / "outer_predictions.parquet", index=False)
    return outer


def run(
    repository: Path,
    site_master_path: Path,
    raw_path: Path,
    context: str,
    mode: str,
    output_dir: Path,
    epochs: int,
    graph_batch_size: int,
    checkpoint_every: int,
    bootstrap_replicates: int,
    seed: int,
    device: torch.device,
) -> dict:
    baseline = upstream_modules(repository)
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("substrate model received a sealed split")
    canonical = master[master["context"] == context].copy().reset_index(drop=True)
    if canonical["baseline_oof_probability"].isna().any():
        raise RuntimeError(f"incomplete development reference for {context}")
    raw = pd.read_parquet(raw_path)
    raw = raw[raw["context"] == context].copy().reset_index(drop=True)
    training = canonical if mode == "joint_binary" else raw
    canonical_graph_list = build_substrate_graphs(canonical, baseline, raw_labels=False)
    training_graph_list = build_substrate_graphs(
        training, baseline, raw_labels=mode != "joint_binary"
    )
    canonical_graphs = {
        int(graph.pair_id_tensor.item()): graph for graph in canonical_graph_list
    }
    training_graphs = {
        int(graph.pair_id_tensor.item()): graph for graph in training_graph_list
    }
    predictions = [
        run_fold(
            canonical,
            canonical_graphs,
            training,
            training_graphs,
            context,
            mode,
            fold,
            epochs,
            graph_batch_size,
            checkpoint_every,
            seed,
            device,
            output_dir,
        )
        for fold in range(5)
    ]
    oof = pd.concat(predictions, ignore_index=True)
    destination = output_dir / mode / context
    oof_path = destination / "oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)
    candidate = metric_row(oof["label"], oof["probability"], oof["prediction"])
    reference = metric_row(
        oof["label"], oof["baseline_oof_probability"], oof["baseline_oof_prediction"]
    )
    bootstrap = bootstrap_delta(oof, bootstrap_replicates, seed + 3000)
    bootstrap.to_parquet(destination / "pair_bootstrap.parquet", index=False)
    confidence = (
        bootstrap.groupby("metric")["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    confidence.to_csv(destination / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "canonical train+valid outer-fold evaluation only",
        "context": context,
        "mode": mode,
        "canonical_rows": len(canonical),
        "training_rows": len(training),
        "canonical_pairs": int(canonical["pair_id"].nunique()),
        "training_pairs": int(training["pair_id"].nunique()),
        "candidate_metrics": candidate,
        "development_reference_metrics": reference,
        "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"substrate_{mode}_{context}_seed{seed}_epochs{epochs}",
            "experiment_type": "substrate_wide_grouped_oof",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "raw_development_sites": sha256_file(raw_path),
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": "single duplex graph; site, partner, and pooled substrate representations",
            "hyperparameters": {
                "mode": mode,
                "epochs": epochs,
                "graph_batch_size": graph_batch_size,
            },
            "artifact": str(
                oof_path.relative_to(repository)
                if oof_path.is_relative_to(repository)
                else oof_path
            ),
            "metrics": summary,
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--site-master", type=Path, default=Path("data/derived/site_master.parquet"))
    parser.add_argument("--raw-sites", type=Path, default=Path("data/derived/cross_tissue/development_raw_individual_tissue_sites.parquet"))
    parser.add_argument("--context", choices=CONTEXTS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/runs/substrate_wide"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--graph-batch-size", type=int, default=24)
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
                args.context,
                args.mode,
                resolve(args.output_dir),
                args.epochs,
                args.graph_batch_size,
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
