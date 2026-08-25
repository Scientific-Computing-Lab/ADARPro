"""Grouped OOF comparison of natural-binary and continuous/ordinal multi-task GATs."""

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
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from adarpro.features.genomic_context import pair_intrinsic_features
from adarpro.models.baseline_oof import (
    CONTEXTS,
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
from adarpro.models.residual import matched_recall_threshold, metrics
from adarpro.quarantine import sha256_file
from adarpro.registry import register

MODES = (
    "binary_natural",
    "multitask",
    "binary_natural_local",
    "multitask_local",
    "binary_pair_balanced",
    "multitask_pair_balanced",
    "binary_pair_balanced_local",
    "binary_pair_balanced_shifted_local",
    "binary_pair_balanced_target_pool",
    "binary_pair_balanced_anchor_pool",
    "binary_pair_balanced_duplicate_pool",
    "binary_pair_balanced_context",
    "binary_pair_balanced_context_engagement",
)
LOCAL_RADIUS = 32
LOCAL_CONTROL_SHIFT = 96
PAIR_CONTEXT_COLUMNS = (
    "geometry__arm1_length",
    "geometry__arm2_length",
    "geometry__arm_length_abs_difference",
    "geometry__genomic_gap",
    "geometry__genomic_gap_log1p",
    "geometry__genomic_span",
    "geometry__arm1_gc",
    "geometry__arm2_gc",
    "geometry__arm_gc_abs_difference",
    "intrinsic__sequence_length",
    "intrinsic__whole_gc",
    "intrinsic__paired_fraction",
    "intrinsic__base_pair_count",
    "intrinsic__mean_base_pair_span",
    "intrinsic__max_base_pair_span",
    "intrinsic__stem_opening_run_count",
)


class MultiTaskGAT(nn.Module):
    """Checkpoint-independent copy of the compact GAT encoder with three heads."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        heads: int = 4,
        use_local: bool = False,
        use_target_pool: bool = False,
        use_anchor_pool: bool = False,
        use_duplicate_pool: bool = False,
        use_pair_context: bool = False,
        pair_context_mean: torch.Tensor | None = None,
        pair_context_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.use_local = use_local
        self.use_target_pool = use_target_pool
        self.use_anchor_pool = use_anchor_pool
        self.use_duplicate_pool = use_duplicate_pool
        self.use_pair_context = use_pair_context
        if sum((use_target_pool, use_anchor_pool, use_duplicate_pool)) > 1:
            raise ValueError(
                "target, anchor and duplicate pooling are mutually exclusive"
            )
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
        embedding_dim = hidden_dim * heads
        if use_target_pool or use_anchor_pool or use_duplicate_pool:
            embedding_dim += hidden_dim * heads
        if use_local:
            self.local_convolution = nn.Sequential(
                nn.Conv1d(12, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=5, padding=4, dilation=2),
                nn.ReLU(),
            )
            embedding_dim += 64
        if use_pair_context:
            self.pair_context_encoder = nn.Sequential(
                nn.Linear(len(PAIR_CONTEXT_COLUMNS), 32),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.engagement_head = nn.Linear(32, 1)
            embedding_dim += 32
            mean = (
                torch.zeros(len(PAIR_CONTEXT_COLUMNS))
                if pair_context_mean is None
                else pair_context_mean.float()
            )
            std = (
                torch.ones(len(PAIR_CONTEXT_COLUMNS))
                if pair_context_std is None
                else pair_context_std.float()
            )
            self.register_buffer("pair_context_mean", mean)
            self.register_buffer("pair_context_std", std)
        self.binary_head = nn.Linear(embedding_dim, 1)
        self.ordinal_head = nn.Linear(embedding_dim, 3)
        self.continuous_head = nn.Linear(embedding_dim, 1)

    def encode(self, data) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = data.x
        for layer, normalization in zip(self.gat_layers, self.batch_norm_layers):
            x = layer(x, data.edge_index)
            x = self.dropout(normalization(F.relu(x)))
        embedding = global_mean_pool(x, data.batch)
        if self.use_target_pool:
            target_mask = data.x[:, 7] > 0.5
            target_batch = data.batch[target_mask]
            expected = torch.arange(
                embedding.shape[0], device=target_batch.device, dtype=target_batch.dtype
            )
            if target_batch.shape != expected.shape or not torch.equal(
                target_batch, expected
            ):
                raise RuntimeError(
                    "each graph must contain exactly one ordered target node"
                )
            embedding = torch.cat([embedding, x[target_mask]], dim=1)
        elif self.use_anchor_pool:
            embedding = torch.cat([embedding, x[data.ptr[:-1]]], dim=1)
        elif self.use_duplicate_pool:
            embedding = torch.cat([embedding, embedding], dim=1)
        if self.use_local:
            local_map = self.local_convolution(data.local_window)
            local = torch.cat(
                [
                    local_map[:, :, LOCAL_RADIUS],
                    F.adaptive_max_pool1d(local_map, 1).squeeze(-1),
                ],
                dim=1,
            )
            embedding = torch.cat([embedding, local], dim=1)
        engagement = None
        if self.use_pair_context:
            standardized = (
                data.pair_context - self.pair_context_mean
            ) / self.pair_context_std
            pair_embedding = self.pair_context_encoder(standardized)
            engagement = torch.sigmoid(self.engagement_head(pair_embedding).view(-1))
            embedding = torch.cat([embedding, pair_embedding], dim=1)
        return embedding, engagement

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding, _ = self.encode(data)
        return (
            self.binary_head(embedding).view(-1),
            self.ordinal_head(embedding),
            torch.sigmoid(self.continuous_head(embedding).view(-1)),
        )

    def forward_with_engagement(
        self, data
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding, engagement = self.encode(data)
        if engagement is None:
            raise RuntimeError("engagement output requested without pair context")
        return (
            self.binary_head(embedding).view(-1),
            self.ordinal_head(embedding),
            torch.sigmoid(self.continuous_head(embedding).view(-1)),
            engagement,
        )


def local_window(sequence: str, structure: str, target: int) -> torch.Tensor:
    """Encode target and helix-aligned partner windows for a local CNN."""

    base_index = {"A": 0, "G": 1, "C": 2, "T": 3, "U": 3, "N": 4}
    width = 2 * LOCAL_RADIUS + 1
    values = torch.zeros((1, 12, width), dtype=torch.float32)
    values[:, 4, :] = 1.0
    values[:, 10, :] = 1.0
    partners = np.full(len(structure), -1, dtype=np.int64)
    stack = []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")" and stack:
            partner = stack.pop()
            partners[index] = partner
            partners[partner] = index
    target_partner = int(partners[target])
    for output_position, offset in enumerate(range(-LOCAL_RADIUS, LOCAL_RADIUS + 1)):
        partner_position = target_partner - offset if target_partner >= 0 else -1
        positions = (target + offset, partner_position)
        for channel_offset, source_position in zip((0, 6), positions):
            if not 0 <= source_position < len(sequence):
                continue
            values[0, channel_offset + 4, output_position] = 0.0
            values[
                0,
                channel_offset + base_index.get(sequence[source_position].upper(), 4),
                output_position,
            ] = 1.0
            values[0, channel_offset + 5, output_position] = float(
                structure[source_position] in "()"
            )
    return values


def shifted_local_position(sequence_length: int, target: int) -> int:
    """Choose a deterministic, nonoverlapping local-control anchor."""

    if not 0 <= target < sequence_length:
        raise ValueError("target outside sequence")
    for candidate in (target + LOCAL_CONTROL_SHIFT, target - LOCAL_CONTROL_SHIFT):
        if 0 <= candidate < sequence_length:
            return candidate
    return 0 if target >= sequence_length - 1 - target else sequence_length - 1


def attach_local_windows(
    graphs: list, table: pd.DataFrame, *, shifted: bool = False
) -> None:
    if len(graphs) != len(table):
        raise RuntimeError("local-window graph/table length mismatch")
    for graph, row in zip(graphs, table.itertuples(index=False)):
        target = int(row.local_position)
        anchor = (
            shifted_local_position(len(row.sequence), target) if shifted else target
        )
        graph.local_window = local_window(row.sequence, row.structure, anchor)


def make_training_graphs(
    raw: pd.DataFrame,
    baseline,
    *,
    include_local: bool = True,
    shifted_local: bool = False,
) -> list:
    graphs = []
    for row in raw.itertuples(index=False):
        graph = baseline.create_rna_graph(
            row.sequence, row.structure, int(row.local_position), 0
        )
        if graph is None:
            raise RuntimeError(f"unbuildable raw graph: {row.example_id}")
        state = 0 if row.editing_index < 1 else 2 if row.editing_index >= 15 else 1
        graph.ordinal_label = torch.tensor([state], dtype=torch.long)
        graph.editing_fraction = torch.tensor(
            [float(row.editing_index) / 100], dtype=torch.float32
        )
        graph.binary_mask = torch.tensor([state != 1], dtype=torch.bool)
        graph.binary_label = torch.tensor([state == 2], dtype=torch.float32)
        if include_local:
            target = int(row.local_position)
            anchor = (
                shifted_local_position(len(row.sequence), target)
                if shifted_local
                else target
            )
            graph.local_window = local_window(row.sequence, row.structure, anchor)
        graphs.append(graph)
    return graphs


def pair_context_table(raw: pd.DataFrame) -> pd.DataFrame:
    pairs = raw.drop_duplicates("pair_id").copy()
    rows = []
    for row in pairs.itertuples(index=False):
        values = pair_intrinsic_features(pd.Series(row._asdict()))
        rows.append({"pair_id": row.pair_id, **values})
    result = pd.DataFrame(rows)
    observed = set(result) - {"pair_id"}
    if observed != set(PAIR_CONTEXT_COLUMNS):
        raise RuntimeError("unexpected pair-context feature schema")
    high_low = raw[(raw["editing_index"] < 1) | (raw["editing_index"] >= 15)].copy()
    high_low["high"] = high_low["editing_index"] >= 15
    engagement = (
        high_low.groupby("pair_id")["high"].mean().rename("engagement_fraction")
    )
    result = result.merge(engagement, on="pair_id", how="left", validate="one_to_one")
    # Pairs containing only intermediate sites are never sampled by a binary
    # mode. Keep their static context buildable while giving the unused
    # auxiliary target a finite placeholder.
    result["engagement_fraction"] = result["engagement_fraction"].fillna(0.0)
    if result[list(PAIR_CONTEXT_COLUMNS)].isna().any().any():
        raise RuntimeError("non-finite pair-context feature")
    return result


def attach_pair_context(
    graphs: list, table: pd.DataFrame, pair_context: pd.DataFrame
) -> None:
    if len(graphs) != len(table):
        raise RuntimeError("pair-context graph/table length mismatch")
    lookup = pair_context.set_index("pair_id")
    missing = set(table["pair_id"]) - set(lookup.index)
    if missing:
        raise RuntimeError(f"missing pair context for {len(missing)} pairs")
    for graph, pair_id in zip(graphs, table["pair_id"]):
        row = lookup.loc[pair_id]
        graph.pair_context = torch.tensor(
            row[list(PAIR_CONTEXT_COLUMNS)].to_numpy(dtype=np.float32)[None, :]
        )
        graph.engagement_fraction = torch.tensor(
            [float(row["engagement_fraction"])], dtype=torch.float32
        )


def create_model(
    device: torch.device,
    mode: str,
    pair_context_mean: torch.Tensor | None = None,
    pair_context_std: torch.Tensor | None = None,
):
    use_pair_context = "_context" in mode
    model = MultiTaskGAT(
        use_local=mode.endswith("_local"),
        use_target_pool=mode.endswith("_target_pool"),
        use_anchor_pool=mode.endswith("_anchor_pool"),
        use_duplicate_pool=mode.endswith("_duplicate_pool"),
        use_pair_context=use_pair_context,
        pair_context_mean=pair_context_mean,
        pair_context_std=pair_context_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return model, optimizer


def sample_epoch_indices(
    state_indices: dict[int, np.ndarray],
    sample_size: int,
    mode: str,
    rng: np.random.Generator,
    pair_values: np.ndarray | None = None,
) -> np.ndarray:
    states = (0, 2) if mode.startswith("binary") else (0, 1, 2)
    counts = [sample_size // len(states)] * len(states)
    for index in range(sample_size % len(states)):
        counts[index] += 1
    selected = []
    for state, count in zip(states, counts):
        available = state_indices[state]
        if not len(available):
            raise RuntimeError(f"no training observations for ordinal state {state}")
        if "pair_balanced" not in mode:
            selected.append(
                rng.choice(available, size=count, replace=count > len(available))
            )
            continue
        if pair_values is None:
            raise ValueError("pair-balanced sampling requires pair_values")
        available_pairs = pair_values[available]
        unique_pairs = np.unique(available_pairs)
        sampled_pairs = rng.choice(unique_pairs, size=count, replace=True)
        by_pair = {pair: available[available_pairs == pair] for pair in unique_pairs}
        selected.append(
            np.asarray([rng.choice(by_pair[pair]) for pair in sampled_pairs])
        )
    return rng.permutation(np.concatenate(selected))


def train_epoch(
    model: MultiTaskGAT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mode: str,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {"loss": [], "binary": [], "ordinal": [], "continuous": []}
    if mode.endswith("_context_engagement"):
        totals["engagement"] = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        if mode.endswith("_context_engagement"):
            binary_logit, ordinal_logit, continuous, engagement = (
                model.forward_with_engagement(batch)
            )
            engagement_loss = F.huber_loss(
                engagement,
                batch.engagement_fraction.view(-1),
                delta=0.1,
            )
        else:
            binary_logit, ordinal_logit, continuous = model(batch)
            engagement_loss = torch.zeros((), device=device)
        binary_mask = batch.binary_mask.view(-1).bool()
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logit[binary_mask], batch.binary_label.view(-1)[binary_mask]
        )
        if mode.startswith("binary"):
            ordinal_loss = torch.zeros((), device=device)
            continuous_loss = torch.zeros((), device=device)
            loss = binary_loss + 0.5 * engagement_loss
        else:
            ordinal_loss = F.cross_entropy(ordinal_logit, batch.ordinal_label.view(-1))
            continuous_loss = F.huber_loss(
                continuous, batch.editing_fraction.view(-1), delta=0.1
            )
            loss = binary_loss + 0.5 * ordinal_loss + 0.5 * continuous_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite multi-task loss")
        loss.backward()
        optimizer.step()
        values = [
            ("loss", loss),
            ("binary", binary_loss),
            ("ordinal", ordinal_loss),
            ("continuous", continuous_loss),
        ]
        if mode.endswith("_context_engagement"):
            values.append(("engagement", engagement_loss))
        for key, value in values:
            totals[key].append(float(value.detach().cpu()))
    return {key: float(np.mean(values)) for key, values in totals.items()}


@torch.no_grad()
def predict_binary(
    model: MultiTaskGAT, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        binary_logit, _, _ = model(batch)
        probabilities.extend(torch.sigmoid(binary_logit).cpu().tolist())
        labels.extend(batch.y.view(-1).int().cpu().tolist())
    return np.asarray(probabilities), np.asarray(labels)


@torch.no_grad()
def predict_binary_and_engagement(
    model: MultiTaskGAT, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels, engagement_values = [], [], []
    for batch in loader:
        batch = batch.to(device)
        binary_logit, _, _, engagement = model.forward_with_engagement(batch)
        probabilities.extend(torch.sigmoid(binary_logit).cpu().tolist())
        labels.extend(batch.y.view(-1).int().cpu().tolist())
        engagement_values.extend(engagement.cpu().tolist())
    return (
        np.asarray(probabilities),
        np.asarray(labels),
        np.asarray(engagement_values),
    )


def append_history(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_fold(
    canonical: pd.DataFrame,
    canonical_graphs: list,
    raw: pd.DataFrame,
    raw_graphs: list,
    pair_context: pd.DataFrame,
    context: str,
    mode: str,
    outer_fold: int,
    epochs: int,
    batch_size: int,
    checkpoint_every: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    inner_fold = (outer_fold + 1) % 5
    canonical_masks = {
        "train": ~canonical["development_fold"].isin([outer_fold, inner_fold]),
        "selection": canonical["development_fold"] == inner_fold,
        "outer": canonical["development_fold"] == outer_fold,
    }
    raw_train_mask = ~raw["development_fold"].isin([outer_fold, inner_fold])
    pair_sets = {
        "raw_train": set(raw.loc[raw_train_mask, "pair_id"]),
        "selection": set(canonical.loc[canonical_masks["selection"], "pair_id"]),
        "outer": set(canonical.loc[canonical_masks["outer"], "pair_id"]),
    }
    if (
        pair_sets["raw_train"] & pair_sets["selection"]
        or pair_sets["raw_train"] & pair_sets["outer"]
        or pair_sets["selection"] & pair_sets["outer"]
    ):
        raise RuntimeError("pair leakage in multi-task fold")
    selection_indices = np.flatnonzero(canonical_masks["selection"].to_numpy())
    outer_indices = np.flatnonzero(canonical_masks["outer"].to_numpy())
    raw_train_indices = np.flatnonzero(raw_train_mask.to_numpy())
    raw_states = np.where(
        raw["editing_index"].to_numpy() < 1,
        0,
        np.where(raw["editing_index"].to_numpy() >= 15, 2, 1),
    )
    state_indices = {
        state: raw_train_indices[raw_states[raw_train_indices] == state]
        for state in (0, 1, 2)
    }
    per_epoch_samples = int(canonical_masks["train"].sum())
    pair_context_indexed = pair_context.set_index("pair_id")
    training_pair_ids = sorted(pair_sets["raw_train"])
    training_pair_values = pair_context_indexed.loc[
        training_pair_ids, list(PAIR_CONTEXT_COLUMNS)
    ].to_numpy(dtype=np.float32)
    pair_mean = torch.tensor(training_pair_values.mean(axis=0), dtype=torch.float32)
    pair_std = torch.tensor(training_pair_values.std(axis=0), dtype=torch.float32)
    pair_std = torch.where(pair_std < 1e-8, torch.ones_like(pair_std), pair_std)
    fold_dir = output_dir / mode / context / f"outer_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "context": context,
        "mode": mode,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "epochs": epochs,
        "batch_size": batch_size,
        "per_epoch_samples": per_epoch_samples,
        "sampler": "equal ordinal states; stochastic raw-site sample with canonical-training-row-sized update budget",
        "loss": (
            "binary BCE"
            if mode.startswith("binary")
            else "binary BCE + 0.5 ordinal CE + 0.5 continuous Huber(delta=0.1)"
        ),
        "selection": "canonical inner-fold binary F1 and threshold",
        "canonical_test_access": False,
    }
    if "pair_balanced" in mode:
        states = (
            "absent/high" if mode.startswith("binary") else "absent/intermediate/high"
        )
        config["sampler"] = (
            f"equal {states} states; uniform pair then uniform site within pair; "
            "canonical-training-row-sized update budget"
        )
    if mode.endswith("_shifted_local"):
        config["architecture"] = (
            "compact GAT + capacity-matched +/-32 nt dilated local CNN centered "
            "96 nt away from the target; nonfocal alignment control"
        )
    elif mode.endswith("_local"):
        config["architecture"] = (
            "compact GAT + target/helix-partner-aligned +/-32 nt dilated local CNN"
        )
    if mode.endswith("_target_pool"):
        config["architecture"] = (
            "compact GAT global mean embedding concatenated with the final focal-node embedding"
        )
    if mode.endswith("_anchor_pool"):
        config["architecture"] = (
            "compact GAT global mean embedding concatenated with the final first-node embedding; "
            "matched-capacity negative control for focal-node pooling"
        )
    if mode.endswith("_duplicate_pool"):
        config["architecture"] = (
            "compact GAT global mean embedding concatenated with itself; "
            "matched-width negative control for focal-node pooling"
        )
    if "_context" in mode:
        config["architecture"] = (
            config.get("architecture", "compact GAT")
            + " + 16-feature fold-standardized substrate context MLP"
        )
        config["pair_context_columns"] = PAIR_CONTEXT_COLUMNS
        config["pair_context_mean"] = pair_mean.tolist()
        config["pair_context_std"] = pair_std.tolist()
        if mode.endswith("_context_engagement"):
            config["loss"] += " + 0.5 pair edited-fraction Huber(delta=0.1)"
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
    # Keep initialization comparable across supervision/sampler variants.
    fold_seed = seed + 1000 * CONTEXTS.index(context) + outer_fold
    configure(fold_seed, threads=8)
    sample_rng = np.random.default_rng(fold_seed + 700_000)
    model, optimizer = create_model(device, mode, pair_mean, pair_std)
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
        sampled = sample_epoch_indices(
            state_indices,
            per_epoch_samples,
            mode,
            sample_rng,
            pair_values=raw["pair_id"].to_numpy(),
        )
        loader = DataLoader(
            [raw_graphs[index] for index in sampled],
            batch_size=batch_size,
            shuffle=False,
        )
        losses = train_epoch(model, loader, optimizer, mode, device)
        probability, labels = predict_binary(model, selection_loader, device)
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
                **losses,
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
                f"[{mode} {context} outer={outer_fold}] {epoch}/{epochs} "
                f"loss={losses['loss']:.5f} selection_f1={selection_f1:.5f} "
                f"best={best['epoch']}",
                flush=True,
            )
    saved = torch_load(best_path, device)
    model.load_state_dict(saved["model_state"])
    if mode.endswith("_context_engagement"):
        selection_probability, selection_labels, _ = predict_binary_and_engagement(
            model, selection_loader, device
        )
    else:
        selection_probability, selection_labels = predict_binary(
            model, selection_loader, device
        )
    baseline_selection_recall = recall_score(
        selection_labels,
        canonical.loc[canonical_masks["selection"], "baseline_oof_prediction"],
        zero_division=0,
    )
    recall_threshold = matched_recall_threshold(
        selection_labels, selection_probability, baseline_selection_recall
    )
    if mode.endswith("_context_engagement"):
        probability, labels, predicted_engagement = predict_binary_and_engagement(
            model, outer_loader, device
        )
    else:
        probability, labels = predict_binary(model, outer_loader, device)
    threshold = float(saved["best"]["threshold"])
    outer = canonical.iloc[outer_indices][
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
    ].copy()
    if not np.array_equal(outer["label"].to_numpy(), labels):
        raise RuntimeError("multi-task outer prediction order mismatch")
    outer["mode"] = mode
    outer["probability"] = probability
    outer["threshold"] = threshold
    outer["prediction"] = (probability >= threshold).astype(int)
    outer["matched_recall_threshold"] = recall_threshold
    outer["matched_recall_prediction"] = (probability >= recall_threshold).astype(int)
    outer["inner_baseline_recall"] = baseline_selection_recall
    outer["best_epoch"] = int(saved["best"]["epoch"])
    if mode.endswith("_context_engagement"):
        outer["predicted_engagement"] = predicted_engagement
        engagement_lookup = pair_context.set_index("pair_id")["engagement_fraction"]
        outer["observed_engagement"] = outer["pair_id"].map(engagement_lookup)
    outer.to_parquet(fold_dir / "outer_predictions.parquet", index=False)
    return outer


def bootstrap_delta(
    predictions: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    pairs = predictions["pair_id"].unique()
    values = predictions["pair_id"].to_numpy()
    indices = {pair: np.flatnonzero(values == pair) for pair in pairs}
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        subset = predictions.iloc[np.concatenate([indices[pair] for pair in sampled])]
        candidate = metrics(
            subset["label"], subset["probability"], subset["prediction"]
        )
        baseline = metrics(
            subset["label"],
            subset["baseline_oof_probability"],
            subset["baseline_oof_prediction"],
        )
        for name in (
            "auroc",
            "auprc",
            "f1",
            "precision",
            "recall",
            "specificity",
            "balanced_accuracy",
            "mcc",
            "brier",
        ):
            rows.append(
                {
                    "replicate": replicate,
                    "metric": name,
                    "difference": candidate[name] - baseline[name],
                }
            )
    return pd.DataFrame(rows)


def run(
    repository: Path,
    site_master_path: Path,
    raw_path: Path,
    context: str,
    mode: str,
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
        raise RuntimeError("multi-task site master contains a sealed split")
    canonical = master[master["context"] == context].copy().reset_index(drop=True)
    if canonical["baseline_oof_probability"].isna().any():
        raise RuntimeError(f"canonical baseline OOF is incomplete for {context}")
    raw = pd.read_parquet(raw_path)
    raw = raw[raw["context"] == context].copy().reset_index(drop=True)
    if set(raw["development_fold"].unique()) != set(range(5)):
        raise RuntimeError("raw multi-task table lacks five development folds")
    print(
        f"[{mode} {context}] building {len(canonical)} canonical and {len(raw)} raw graphs",
        flush=True,
    )
    canonical_graphs = build_graphs(canonical, baseline)
    use_local = mode.endswith("_local")
    shifted_local = mode.endswith("_shifted_local")
    if use_local:
        attach_local_windows(canonical_graphs, canonical, shifted=shifted_local)
    raw_graphs = make_training_graphs(
        raw,
        baseline,
        include_local=use_local,
        shifted_local=shifted_local,
    )
    pair_context = pair_context_table(raw)
    attach_pair_context(canonical_graphs, canonical, pair_context)
    attach_pair_context(raw_graphs, raw, pair_context)
    predictions = [
        run_fold(
            canonical,
            canonical_graphs,
            raw,
            raw_graphs,
            pair_context,
            context,
            mode,
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
    destination = output_dir / mode / context
    oof_path = destination / "oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)
    candidate_metrics = metric_row(oof["label"], oof["probability"], oof["prediction"])
    baseline_metrics = metric_row(
        oof["label"], oof["baseline_oof_probability"], oof["baseline_oof_prediction"]
    )
    bootstrap = bootstrap_delta(oof, bootstrap_replicates, seed + 2000)
    bootstrap.to_parquet(destination / "pair_bootstrap.parquet", index=False)
    ci = (
        bootstrap.groupby("metric")["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    ci.to_csv(destination / "pair_bootstrap_ci.csv", index=False)
    matched_oof = oof.copy()
    matched_oof["prediction"] = matched_oof["matched_recall_prediction"]
    matched_bootstrap = bootstrap_delta(matched_oof, bootstrap_replicates, seed + 3000)
    matched_bootstrap.to_parquet(
        destination / "pair_bootstrap_matched_recall.parquet", index=False
    )
    matched_ci = (
        matched_bootstrap.groupby("metric")["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    matched_ci.to_csv(destination / "pair_bootstrap_matched_recall_ci.csv", index=False)
    summary = {
        "scope": "canonical train+valid outer-fold evaluation; raw development observations used only in non-held-out folds",
        "context": context,
        "mode": mode,
        "canonical_rows": len(canonical),
        "raw_training_pool_rows": len(raw),
        "pairs": int(canonical["pair_id"].nunique()),
        "epochs": epochs,
        "candidate_metrics": candidate_metrics,
        "candidate_matched_recall_metrics": metric_row(
            oof["label"], oof["probability"], oof["matched_recall_prediction"]
        ),
        "development_reference_metrics": baseline_metrics,
        "canonical_test_access": False,
    }
    if "predicted_engagement" in oof:
        pair_predictions = oof[
            ["pair_id", "observed_engagement", "predicted_engagement"]
        ].drop_duplicates("pair_id")
        summary["engagement_metrics"] = {
            "pairs": len(pair_predictions),
            "r2": float(
                r2_score(
                    pair_predictions["observed_engagement"],
                    pair_predictions["predicted_engagement"],
                )
            ),
            "mae": float(
                mean_absolute_error(
                    pair_predictions["observed_engagement"],
                    pair_predictions["predicted_engagement"],
                )
            ),
            "spearman": float(
                spearmanr(
                    pair_predictions["observed_engagement"],
                    pair_predictions["predicted_engagement"],
                ).statistic
            ),
        }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"{mode}_oof_{context}_seed{seed}_epochs{epochs}",
            "experiment_type": "natural_binary_or_multitask_grouped_oof",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "raw_development_sites": sha256_file(raw_path),
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": (
                "exact compact GAT graph plus capacity-matched nonfocal +/-32 nt local CNN shifted 96 nt from target"
                if mode.endswith("_shifted_local")
                else (
                    "exact compact GAT graph plus prespecified target/partner-aligned +/-32 nt local CNN"
                    if mode.endswith("_local")
                    else (
                        "exact compact GAT graph with global-mean plus focal-node pooling"
                        if mode.endswith("_target_pool")
                        else (
                            "exact compact GAT graph with global-mean plus first-node control pooling"
                            if mode.endswith("_anchor_pool")
                            else (
                                "exact compact GAT graph with duplicated global-mean control pooling"
                                if mode.endswith("_duplicate_pool")
                                else (
                                    "compact GAT plus pair context and engagement auxiliary target"
                                    if mode.endswith("_context_engagement")
                                    else (
                                        "compact GAT plus pair context"
                                        if "_context" in mode
                                        else "exact compact GAT graph; richer supervision only"
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            "hyperparameters": {
                "mode": mode,
                "epochs": epochs,
                "batch_size": batch_size,
                "per_epoch_update_budget": "canonical training rows",
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
    parser.add_argument("--context", choices=CONTEXTS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
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
                args.context,
                args.mode,
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
