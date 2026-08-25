"""Natural-prevalence OOF evaluation of an expanded-supervision neural model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from adarpro.models.baseline_oof import metric_row, torch_load, upstream_modules
from adarpro.models.multitask_oof import (
    MultiTaskGAT,
    attach_pair_context,
    bootstrap_delta,
    make_training_graphs,
    pair_context_table,
    predict_binary,
)
from adarpro.quarantine import sha256_file
from adarpro.registry import register


def run(
    repository: Path,
    raw_path: Path,
    baseline_path: Path,
    run_dir: Path,
    output_dir: Path,
    context: str,
    mode: str,
    batch_size: int,
    bootstrap_replicates: int,
    seed: int,
    device: torch.device,
) -> dict:
    raw = pd.read_parquet(raw_path)
    raw = (
        raw[
            (raw["context"] == context)
            & ((raw["editing_index"] < 1) | (raw["editing_index"] >= 15))
        ]
        .copy()
        .reset_index(drop=True)
    )
    if set(raw["development_fold"].unique()) != set(range(5)):
        raise RuntimeError("natural-prevalence raw table lacks development folds")
    baseline_oof = pd.read_parquet(baseline_path).rename(
        columns={
            "probability": "baseline_oof_probability",
            "threshold": "baseline_oof_threshold",
            "prediction": "baseline_oof_prediction",
        }
    )
    identity = [
        "example_id",
        "context",
        "pair_id",
        "pair_local_key",
        "development_fold",
    ]
    expected = raw[identity].sort_values("example_id").reset_index(drop=True)
    observed = baseline_oof[identity].sort_values("example_id").reset_index(drop=True)
    if not expected.equals(observed):
        raise RuntimeError(
            "natural-prevalence reference does not match raw high/low rows"
        )
    baseline = upstream_modules(repository)
    graphs = make_training_graphs(
        raw,
        baseline,
        include_local=mode.endswith("_local"),
        shifted_local=mode.endswith("_shifted_local"),
    )
    if "_context" in mode:
        attach_pair_context(graphs, raw, pair_context_table(raw))
    outputs, checkpoint_hashes = [], {}
    for fold in range(5):
        indices = raw.index[raw["development_fold"] == fold].to_numpy()
        loader = DataLoader(
            [graphs[index] for index in indices], batch_size=batch_size, shuffle=False
        )
        checkpoint_path = run_dir / f"outer_{fold}" / "best.pth"
        checkpoint_hashes[f"outer_{fold}"] = sha256_file(checkpoint_path)
        saved = torch_load(checkpoint_path, device)
        model = MultiTaskGAT(
            use_local=mode.endswith("_local"),
            use_target_pool=mode.endswith("_target_pool"),
            use_anchor_pool=mode.endswith("_anchor_pool"),
            use_duplicate_pool=mode.endswith("_duplicate_pool"),
            use_pair_context="_context" in mode,
        ).to(device)
        model.load_state_dict(saved["model_state"])
        probability, _ = predict_binary(model, loader, device)
        outer = raw.loc[indices, identity + ["editing_index", "binary_label"]].copy()
        outer["label"] = outer.pop("binary_label").astype(int)
        outer["probability"] = probability
        outer["threshold"] = float(saved["best"]["threshold"])
        outer["prediction"] = (outer["probability"] >= outer["threshold"]).astype(int)
        outputs.append(outer)
    predictions = pd.concat(outputs, ignore_index=True)
    predictions = predictions.merge(
        baseline_oof[
            identity
            + [
                "baseline_oof_probability",
                "baseline_oof_threshold",
                "baseline_oof_prediction",
            ]
        ],
        on=identity,
        validate="one_to_one",
    )
    candidate_metrics = metric_row(
        predictions["label"], predictions["probability"], predictions["prediction"]
    )
    reference_metrics = metric_row(
        predictions["label"],
        predictions["baseline_oof_probability"],
        predictions["baseline_oof_prediction"],
    )
    bootstrap = bootstrap_delta(predictions, bootstrap_replicates, seed)
    confidence = (
        bootstrap.groupby("metric")["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(output_dir / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "all development high/low sites at natural prevalence; held-out global pair folds",
        "context": context,
        "mode": mode,
        "rows": len(predictions),
        "pairs": int(predictions["pair_id"].nunique()),
        "prevalence": float(predictions["label"].mean()),
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
            "run_id": f"natural_prevalence_{mode}_{context}_seed{seed}",
            "experiment_type": "expanded_supervision_natural_prevalence_oof",
            "context": context,
            "input_hashes": {
                "raw_development_sites": sha256_file(raw_path),
                "baseline_natural_prevalence": sha256_file(baseline_path),
                "checkpoints": checkpoint_hashes,
            },
            "fold_definition": "five global pair folds; canonical inner-selected threshold",
            "seed": seed,
            "features": f"completed {mode} OOF model applied to every high/low raw site",
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
        "--raw-sites",
        type=Path,
        default=Path(
            "data/derived/cross_tissue/development_raw_individual_tissue_sites.parquet"
        ),
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=Path(
            "experiments/runs/natural_prevalence/MuscleSkeletal/oof_predictions.parquet"
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument("--mode", default="binary_natural")
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
                resolve(args.raw_sites),
                resolve(args.baseline_predictions),
                resolve(args.run_dir),
                resolve(args.output_dir),
                args.context,
                args.mode,
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
