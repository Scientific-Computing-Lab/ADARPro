"""Pair-disjoint development training for the unified ADARPro architecture."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

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
from adarpro.models.final_training import truncate_history_after_checkpoint
from adarpro.models.multitask_oof import (
    append_history,
    attach_local_windows,
    make_training_graphs,
    sample_epoch_indices,
)
from adarpro.models.unified import VARIANTS, UnifiedADARPro, parameter_count
from adarpro.quarantine import sha256_file
from adarpro.registry import register


@torch.no_grad()
def predict_unified(
    model: UnifiedADARPro, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict final and training-only auxiliary probabilities."""

    model.eval()
    probabilities: list[float] = []
    auxiliary_probabilities: list[float] = []
    labels: list[int] = []
    for batch in loader:
        batch = batch.to(device)
        final_logit, auxiliary_logit = model(batch)
        probabilities.extend(torch.sigmoid(final_logit).cpu().tolist())
        auxiliary_probabilities.extend(torch.sigmoid(auxiliary_logit).cpu().tolist())
        labels.extend(batch.y.view(-1).int().cpu().tolist())
    return (
        np.asarray(probabilities),
        np.asarray(auxiliary_probabilities),
        np.asarray(labels),
    )


def train_unified_epoch(
    model: UnifiedADARPro,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    auxiliary_weight: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """Train all unified modules jointly for one sampled epoch."""

    model.train()
    losses: list[float] = []
    final_losses: list[float] = []
    auxiliary_losses: list[float] = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        final_logit, auxiliary_logit = model(batch)
        labels = batch.binary_label.view(-1)
        final_loss = F.binary_cross_entropy_with_logits(final_logit, labels)
        auxiliary_loss = F.binary_cross_entropy_with_logits(auxiliary_logit, labels)
        loss = final_loss + auxiliary_weight * auxiliary_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite unified-model loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        final_losses.append(float(final_loss.detach().cpu()))
        auxiliary_losses.append(float(auxiliary_loss.detach().cpu()))
    return (
        float(np.mean(losses)),
        float(np.mean(final_losses)),
        float(np.mean(auxiliary_losses)),
    )


def create_unified_model(
    variant_name: str, device: torch.device
) -> tuple[UnifiedADARPro, torch.optim.Optimizer]:
    """Instantiate one preregistered model variant and optimizer."""

    variant = VARIANTS[variant_name]
    model = UnifiedADARPro(anchor=variant.anchor).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return model, optimizer


def _validate_pair_partitions(pair_sets: dict[str, set[int]]) -> None:
    names = sorted(pair_sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = pair_sets[left_name] & pair_sets[right_name]
            if overlap:
                raise RuntimeError(
                    f"pair leakage between {left_name} and {right_name}: "
                    f"{len(overlap)} pairs"
                )


def run_fold(
    canonical: pd.DataFrame,
    canonical_graphs: list,
    raw: pd.DataFrame,
    raw_graphs: list,
    context: str,
    variant_name: str,
    outer_fold: int,
    epochs: int,
    batch_size: int,
    checkpoint_every: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    """Fit one outer-fold model with an isolated inner selection fold."""

    variant = VARIANTS[variant_name]
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
    _validate_pair_partitions(pair_sets)
    selection_indices = canonical.index[canonical_masks["selection"]].to_numpy()
    outer_indices = canonical.index[canonical_masks["outer"]].to_numpy()
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

    fold_dir = output_dir / variant_name / context / f"outer_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_seed = seed + 1000 * CONTEXTS.index(context) + outer_fold
    config = {
        "architecture": variant_name,
        "auxiliary_global_loss_weight": variant.auxiliary_weight,
        "batch_size": batch_size,
        "canonical_test_access": False,
        "context": context,
        "epochs": epochs,
        "fold_seed": fold_seed,
        "inner_fold": inner_fold,
        "optimizer": "Adam(lr=1e-3, weight_decay=1e-4)",
        "outer_fold": outer_fold,
        "per_epoch_samples": per_epoch_samples,
        "sampler": "equal high/low; uniform pair then uniform site within pair",
        "selection": "canonical inner-fold binary F1 and threshold",
        "training_split": "three development folds; expanded raw high/low sites",
        "trainable_parameters": parameter_count(
            UnifiedADARPro(anchor=variant.anchor)
        ),
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
    configure(fold_seed, threads=8)
    sample_rng = np.random.default_rng(fold_seed + 700_000)
    model, optimizer = create_unified_model(variant_name, device)

    best_path = fold_dir / "best.pth"
    last_path = fold_dir / "last.pth"
    history_path = fold_dir / "history.csv"
    best = {"f1": -1.0, "epoch": 0, "threshold": None}
    start_epoch = 1
    if last_path.exists():
        saved = torch_load(last_path, device)
        if saved.get("config") != config:
            raise RuntimeError(f"checkpoint configuration mismatch: {last_path}")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng_state(saved["rng_state"])
        sample_rng.bit_generator.state = saved["sample_rng_state"]
        best = saved["best"]
        start_epoch = int(saved["epoch"]) + 1
        truncate_history_after_checkpoint(history_path, int(saved["epoch"]))

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        sampled = sample_epoch_indices(
            state_indices,
            per_epoch_samples,
            "binary_pair_balanced",
            sample_rng,
            pair_values=raw["pair_id"].to_numpy(),
        )
        loader = DataLoader(
            [raw_graphs[index] for index in sampled],
            batch_size=batch_size,
            shuffle=False,
        )
        loss, final_loss, auxiliary_loss = train_unified_epoch(
            model, loader, optimizer, variant.auxiliary_weight, device
        )
        probability, auxiliary_probability, labels = predict_unified(
            model, selection_loader, device
        )
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
                "loss": loss,
                "final_loss": final_loss,
                "auxiliary_loss": auxiliary_loss,
                "selection_f1": selection_f1,
                "selection_auroc": roc_auc_score(labels, probability),
                "selection_auprc": average_precision_score(labels, probability),
                "selection_auxiliary_auroc": roc_auc_score(
                    labels, auxiliary_probability
                ),
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
                f"[{variant_name} {context} outer={outer_fold}] {epoch}/{epochs} "
                f"loss={loss:.5f} selection_f1={selection_f1:.5f} "
                f"best={best['epoch']}",
                flush=True,
            )

    saved = torch_load(best_path, device)
    model.load_state_dict(saved["model_state"])
    probability, auxiliary_probability, labels = predict_unified(
        model, outer_loader, device
    )
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
        raise RuntimeError("unified outer prediction order mismatch")
    outer["variant"] = variant_name
    outer["probability"] = probability
    outer["auxiliary_global_probability"] = auxiliary_probability
    outer["threshold"] = threshold
    outer["prediction"] = (probability >= threshold).astype(int)
    outer["best_epoch"] = int(saved["best"]["epoch"])
    outer.to_parquet(fold_dir / "outer_predictions.parquet", index=False)
    return outer


def run(
    repository: Path,
    site_master_path: Path,
    raw_path: Path,
    context: str,
    variant_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    checkpoint_every: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Train all five outer folds and save development-only OOF predictions."""

    if variant_name not in VARIANTS:
        raise ValueError(f"unsupported unified variant: {variant_name}")
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("unified site master contains a sealed split")
    canonical = master[master["context"] == context].copy().reset_index(drop=True)
    if canonical.empty or canonical["baseline_oof_probability"].isna().any():
        raise RuntimeError(f"incomplete canonical development rows for {context}")
    raw = pd.read_parquet(raw_path)
    raw = raw[raw["context"] == context].copy().reset_index(drop=True)
    if set(raw["development_fold"].unique()) != set(range(5)):
        raise RuntimeError("raw unified table lacks five development folds")
    if not set(canonical["pair_id"]).issubset(set(raw["pair_id"])):
        raise RuntimeError("canonical pair missing from expanded raw table")

    baseline = upstream_modules(repository)
    canonical_graphs = build_graphs(canonical, baseline)
    variant = VARIANTS[variant_name]
    attach_local_windows(
        canonical_graphs, canonical, shifted=variant.shifted_local
    )
    raw_graphs = make_training_graphs(
        raw,
        baseline,
        include_local=True,
        shifted_local=variant.shifted_local,
    )
    predictions = [
        run_fold(
            canonical,
            canonical_graphs,
            raw,
            raw_graphs,
            context,
            variant_name,
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
    destination = output_dir / variant_name / context
    prediction_path = destination / "oof_predictions.parquet"
    oof.to_parquet(prediction_path, index=False)
    summary = {
        "architecture": variant_name,
        "auxiliary_global_loss_weight": variant.auxiliary_weight,
        "canonical_rows": len(canonical),
        "canonical_test_access": False,
        "candidate_metrics": metric_row(
            oof["label"], oof["probability"], oof["prediction"]
        ),
        "context": context,
        "epochs": epochs,
        "pairs": int(canonical["pair_id"].nunique()),
        "raw_training_pool_rows": len(raw),
        "scope": (
            "canonical train+valid outer-fold evaluation; raw development "
            "observations used only in non-held-out folds"
        ),
        "trainable_parameters": parameter_count(
            UnifiedADARPro(anchor=variant.anchor)
        ),
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    register(
        repository,
        {
            "run_id": (
                f"{variant_name}_oof_{context}_seed{seed}_epochs{epochs}"
            ),
            "experiment_type": "unified_end_to_end_grouped_oof",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "raw_development_sites": sha256_file(raw_path),
                "unified_policy": sha256_file(
                    repository / "provenance/unified_model_selection_policy.json"
                ),
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": (
                "joint compact GAT global + focal node + aligned local CNN "
                "with representation interactions"
            ),
            "hyperparameters": {
                "architecture": variant_name,
                "auxiliary_global_loss_weight": variant.auxiliary_weight,
                "batch_size": batch_size,
                "epochs": epochs,
                "per_epoch_update_budget": "canonical training rows",
            },
            "artifact": str(prediction_path.relative_to(repository)),
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
    parser.add_argument("--raw-sites", type=Path)
    parser.add_argument("--context", choices=CONTEXTS, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/runs/unified")
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    if args.raw_sites is None:
        raw_path = Path(
            "data/derived/cross_tissue/development_raw_combined_sites.parquet"
            if args.context == "Combined"
            else "data/derived/cross_tissue/development_raw_individual_tissue_sites.parquet"
        )
    else:
        raw_path = args.raw_sites
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    summary = run(
        root,
        resolve(args.site_master),
        resolve(raw_path),
        args.context,
        args.variant,
        resolve(args.output_dir),
        args.epochs,
        args.batch_size,
        args.checkpoint_every,
        args.seed,
        device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
