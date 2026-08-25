"""Train a frozen ADARpro component on published train/validation splits only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

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
from adarpro.models.multitask_oof import (
    MODES,
    append_history,
    attach_local_windows,
    create_model,
    make_training_graphs,
    predict_binary,
    sample_epoch_indices,
)
from adarpro.quarantine import sha256_file
from adarpro.registry import register

FINAL_MODES = (
    "binary_pair_balanced",
    "binary_pair_balanced_target_pool",
    "binary_pair_balanced_local",
)


def truncate_history_after_checkpoint(path: Path, checkpoint_epoch: int) -> None:
    """Discard uncheckpointed history rows before a deterministic resume."""

    if not path.exists():
        return
    history = pd.read_csv(path)
    if "epoch" not in history:
        raise RuntimeError(f"training history lacks an epoch column: {path}")
    if history["epoch"].duplicated().any():
        raise RuntimeError(f"training history already contains duplicate epochs: {path}")
    retained = history[history["epoch"] <= checkpoint_epoch]
    if len(retained) == len(history):
        return
    temporary = path.with_name(f".{path.name}.resume-{checkpoint_epoch}.tmp")
    retained.to_csv(temporary, index=False)
    temporary.replace(path)


def assign_raw_source_split(raw: pd.DataFrame, pair_master: pd.DataFrame) -> pd.DataFrame:
    """Map raw observations to the immutable published pair split."""

    pair_splits = pair_master[["pair_id", "source_split"]].drop_duplicates()
    if pair_splits["pair_id"].duplicated().any():
        raise RuntimeError("a canonical pair occurs in multiple published splits")
    result = raw.merge(pair_splits, on="pair_id", how="left", validate="many_to_one")
    if result["source_split"].isna().any():
        raise RuntimeError("raw observation has no published pair assignment")
    if set(result["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("final training input contains a sealed pair")
    return result


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
    seed: int,
    device: torch.device,
) -> dict:
    if mode not in FINAL_MODES or mode not in MODES:
        raise ValueError(f"unsupported frozen component mode: {mode}")
    # Capture input identity before loading so a concurrent, development-only
    # OOF annotation cannot make the registry describe a later file revision.
    site_master_sha256 = sha256_file(site_master_path)
    raw_sha256 = sha256_file(raw_path)
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("final trainer received a sealed canonical split")
    canonical = master[master["context"] == context].copy().reset_index(drop=True)
    if canonical.empty:
        raise RuntimeError(f"no canonical development rows for {context}")
    raw = pd.read_parquet(raw_path)
    raw = raw[raw["context"] == context].copy().reset_index(drop=True)
    # The raw pool can contain a globally assigned development pair that was
    # not sampled into this context's balanced canonical subset.  Resolve the
    # immutable split from the all-context master, then use this context's
    # canonical validation rows only for checkpoint/threshold selection.
    raw = assign_raw_source_split(raw, master)
    raw = raw[(raw["editing_index"] < 1) | (raw["editing_index"] >= 15)].copy()
    raw = raw.reset_index(drop=True)
    train_indices = np.flatnonzero(raw["source_split"].eq("train").to_numpy())
    validation = canonical[canonical["source_split"] == "valid"].copy()
    if not len(train_indices) or validation.empty:
        raise RuntimeError("empty published train or validation partition")
    if set(raw.loc[train_indices, "pair_id"]) & set(validation["pair_id"]):
        raise RuntimeError("pair leakage between final training and validation")

    baseline = upstream_modules(repository)
    validation_graphs = build_graphs(validation, baseline)
    use_local = mode.endswith("_local")
    if use_local:
        attach_local_windows(validation_graphs, validation)
    raw_graphs = make_training_graphs(raw, baseline, include_local=use_local)
    raw_states = np.where(
        raw["editing_index"].to_numpy() < 1,
        0,
        np.where(raw["editing_index"].to_numpy() >= 15, 2, 1),
    )
    state_indices = {
        state: train_indices[raw_states[train_indices] == state] for state in (0, 1, 2)
    }
    per_epoch_samples = int(canonical["source_split"].eq("train").sum())
    destination = output_dir / context / mode
    destination.mkdir(parents=True, exist_ok=True)
    config = {
        "context": context,
        "mode": mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "per_epoch_samples": per_epoch_samples,
        "training_split": "published train pairs; expanded raw high/low sites",
        "selection_split": "published validation canonical rows",
        "sampler": "equal high/low; uniform pair then uniform site within pair",
        "canonical_test_access": False,
    }
    config_path = destination / "config.json"
    serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if config_path.exists() and config_path.read_text() != serialized:
        raise RuntimeError(f"refusing incompatible resume: {config_path}")
    config_path.write_text(serialized)

    configure(seed, threads=8)
    sample_rng = np.random.default_rng(seed + 700_000)
    model, optimizer = create_model(device, mode)
    validation_loader = DataLoader(
        validation_graphs, batch_size=batch_size, shuffle=False
    )
    best_path = destination / "best.pth"
    last_path = destination / "last.pth"
    history_path = destination / "history.csv"
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
        truncate_history_after_checkpoint(history_path, int(saved["epoch"]))

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
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            binary_logit, _, _ = model(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                binary_logit, batch.binary_label.view(-1)
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probability, labels = predict_binary(model, validation_loader, device)
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
                "validation_f1": selection_f1,
                "validation_auroc": roc_auc_score(labels, probability),
                "validation_auprc": average_precision_score(labels, probability),
                "validation_threshold": threshold,
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
                f"[final-train {context} {mode}] {epoch}/{epochs} "
                f"loss={np.mean(losses):.5f} valid_f1={selection_f1:.5f} "
                f"best={best['epoch']}",
                flush=True,
            )

    saved = torch_load(best_path, device)
    model.load_state_dict(saved["model_state"])
    probability, labels = predict_binary(model, validation_loader, device)
    threshold = float(saved["best"]["threshold"])
    validation["probability"] = probability
    validation["prediction"] = (probability >= threshold).astype(int)
    validation["threshold"] = threshold
    prediction_path = destination / "validation_predictions.parquet"
    validation.to_parquet(prediction_path, index=False)
    validation_metrics = metric_row(
        validation["label"], validation["probability"], validation["prediction"]
    )
    summary = {
        "scope": "published train/validation only; test remains sealed",
        "context": context,
        "mode": mode,
        "train_raw_rows": len(train_indices),
        "train_pairs": int(raw.loc[train_indices, "pair_id"].nunique()),
        "validation_rows": len(validation),
        "validation_pairs": int(validation["pair_id"].nunique()),
        "best_epoch": int(saved["best"]["epoch"]),
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "checkpoint_sha256": sha256_file(best_path),
        "checkpoint": str(best_path.relative_to(repository)),
        "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"final_training_{context}_{mode}_seed{seed}_epochs{epochs}",
            "experiment_type": "frozen_candidate_train_validation_only",
            "context": context,
            "input_hashes": {
                "site_master": site_master_sha256,
                "raw_development_sites": raw_sha256,
            },
            "fold_definition": "published train pairs; published validation selection",
            "seed": seed,
            "features": mode,
            "hyperparameters": config,
            "artifact": str(best_path.relative_to(repository)),
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
    parser.add_argument("--context", required=True)
    parser.add_argument("--mode", choices=FINAL_MODES, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/frozen"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    raw_path = args.raw_sites
    if raw_path is None:
        raw_path = Path(
            "data/derived/cross_tissue/development_raw_combined_sites.parquet"
            if args.context == "Combined"
            else "data/derived/cross_tissue/development_raw_individual_tissue_sites.parquet"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(
        json.dumps(
            run(
                root,
                resolve(args.site_master),
                resolve(raw_path),
                args.context,
                args.mode,
                resolve(args.output_dir),
                args.epochs,
                args.batch_size,
                args.checkpoint_every,
                args.seed,
                device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
