"""Aggregate-only evaluation of immutable supplied human checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

from adarpro.quarantine import published_reproduction_access, sha256_file

CONTEXTS = ("Artery", "Brain", "Liver", "MuscleSkeletal", "Combined")
VARIANTS = ("baseline", "bioaware")


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    return {
        "auroc": roc_auc_score(labels, probabilities),
        "auprc": average_precision_score(labels, probabilities),
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "specificity": recall_score(labels, predictions, pos_label=0, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "accuracy": accuracy_score(labels, predictions),
        "mcc": matthews_corrcoef(labels, predictions),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sample_count": len(labels),
    }


def evaluate_one(repository: Path, context: str, variant: str, device: torch.device) -> dict:
    code_dir = repository / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import bioaware_gnn as bio
    import gnnadar_verb_compact as baseline
    import train_strict_long as trainer

    checkpoint_dir = repository / "checkpoints" / f"{variant}_{context}"
    checkpoint_path = checkpoint_dir / "best.pth"
    summary = json.loads((checkpoint_dir / "summary.json").read_text())
    test_path = repository / "data" / "human" / context / "test.jsonl"
    metadata_path = repository / "data" / "human" / context / "test.metadata.csv"
    if variant == "baseline":
        graphs = baseline.load_jsonl_to_graphs(test_path)
    else:
        graphs = bio.load_graphs(
            str(test_path), use_neighbors=True, use_geometry=True, plfold_dir=None
        )
    loader = DataLoader(graphs, batch_size=256, shuffle=False, num_workers=0)
    model, _, _ = trainer.create_model(variant, int(graphs[0].x.shape[1]), device)
    saved = trainer.torch_load(checkpoint_path, map_location=device)
    model.load_state_dict(saved["model_state"])
    probabilities, labels = trainer.evaluate(variant, model, loader, device)
    threshold = float(saved["best"]["metrics"]["threshold"])
    observed = metrics(labels, probabilities, threshold)
    expected = summary["test_metrics"]
    comparisons = {
        "auroc": "auc",
        "auprc": "auprc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "specificity": "specificity",
        "accuracy": "acc",
    }
    for observed_key, expected_key in comparisons.items():
        if abs(float(observed[observed_key]) - float(expected[expected_key])) > 2e-5:
            raise RuntimeError(
                f"{variant}/{context} mismatch for {observed_key}: "
                f"{observed[observed_key]} versus {expected[expected_key]}"
            )
    observed.update(
        {
            "context": context,
            "model": "Baseline GAT" if variant == "baseline" else "Bio-aware GNN",
            "variant": variant,
            "pair_count": int(pd.read_csv(metadata_path)["pair_id"].nunique()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "data_sha256": sha256_file(test_path),
            "best_epoch": int(saved["best"]["epoch"]),
            "status": "reproduced",
        }
    )
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--policy", type=Path, default=Path("provenance/test_quarantine.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/adarpro/baseline_reproduction.csv")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    repository = args.repository.resolve()
    policy = args.policy if args.policy.is_absolute() else repository / args.policy
    output = args.output if args.output.is_absolute() else repository / args.output
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    command = "python -m adarpro.reproduce.evaluate_supplied"
    with published_reproduction_access(policy, command=command) as audit:
        rows = [
            evaluate_one(repository, context, variant, device)
            for context in CONTEXTS
            for variant in VARIANTS
        ]
        table = pd.DataFrame(rows).sort_values(["context", "variant"])
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
        audit["result_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    print(table.to_string(index=False))
    print(f"aggregate-only result: {output}")


if __name__ == "__main__":
    main()
