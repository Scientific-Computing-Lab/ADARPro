"""Assemble a frozen equal-logit ensemble on published validation data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from adarpro.models.baseline_oof import (
    build_graphs,
    choose_threshold,
    metric_row,
    torch_load,
    upstream_modules,
)
from adarpro.models.fixed_checkpoint_ensemble import (
    equal_logit_average_many,
    model_for_mode,
)
from adarpro.models.multitask_oof import attach_local_windows, predict_binary
from adarpro.quarantine import sha256_file
from adarpro.registry import register


def validate_checkpoint(saved: dict, context: str, mode: str) -> None:
    """Reject a component that was not trained under the frozen specification."""

    config = saved.get("config", {})
    if config.get("context") != context or config.get("mode") != mode:
        raise RuntimeError("component checkpoint context/mode mismatch")
    if config.get("canonical_test_access") is not False:
        raise RuntimeError("component checkpoint lacks sealed-development provenance")
    if config.get("training_split") != (
        "published train pairs; expanded raw high/low sites"
    ):
        raise RuntimeError("component checkpoint has an unexpected training split")
    if config.get("selection_split") != "published validation canonical rows":
        raise RuntimeError("component checkpoint has an unexpected selection split")


def run(
    repository: Path,
    site_master_path: Path,
    context: str,
    components: list[tuple[str, str, Path]],
    output_dir: Path,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    if len(components) < 2:
        raise ValueError("a frozen ensemble requires at least two components")
    names = [name for name, _, _ in components]
    if len(names) != len(set(names)):
        raise ValueError("component names must be unique")
    if any("_context" in mode or mode.endswith("_shifted_local") for _, mode, _ in components):
        raise ValueError("unsupported frozen ensemble component")

    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("final ensemble received a sealed canonical split")
    validation = master[
        (master["context"] == context) & (master["source_split"] == "valid")
    ].copy().reset_index(drop=True)
    if validation.empty:
        raise RuntimeError(f"no published validation rows for {context}")
    graphs = build_graphs(validation, upstream_modules(repository))
    if any(mode.endswith("_local") for _, mode, _ in components):
        attach_local_windows(graphs, validation)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    probabilities = []
    checkpoint_hashes = {}
    checkpoint_paths = {}
    for name, mode, checkpoint in components:
        saved = torch_load(checkpoint, device)
        validate_checkpoint(saved, context, mode)
        model = model_for_mode(mode, device)
        model.load_state_dict(saved["model_state"])
        probability, labels = predict_binary(model, loader, device)
        if not (labels == validation["label"].to_numpy()).all():
            raise RuntimeError(f"validation row mismatch for component {name}")
        probabilities.append(probability)
        validation[f"{name}_probability"] = probability
        checkpoint_hashes[name] = sha256_file(checkpoint)
        checkpoint_paths[name] = str(checkpoint.relative_to(repository))

    validation["probability"] = equal_logit_average_many(probabilities)
    threshold, selection_f1 = choose_threshold(
        validation["label"].to_numpy(), validation["probability"].to_numpy()
    )
    validation["threshold"] = threshold
    validation["prediction"] = (
        validation["probability"] >= threshold
    ).astype(int)
    destination = output_dir / context
    destination.mkdir(parents=True, exist_ok=True)
    prediction_path = destination / "validation_predictions.parquet"
    validation.to_parquet(prediction_path, index=False)
    summary = {
        "scope": "published validation only; canonical test remains sealed",
        "context": context,
        "ensemble": "fixed equal-logit average",
        "components": {name: mode for name, mode, _ in components},
        "weights": [1 / len(components)] * len(components),
        "threshold": threshold,
        "selection_f1": selection_f1,
        "validation_rows": len(validation),
        "validation_pairs": int(validation["pair_id"].nunique()),
        "validation_metrics": metric_row(
            validation["label"], validation["probability"], validation["prediction"]
        ),
        "checkpoints": checkpoint_paths,
        "checkpoint_sha256": checkpoint_hashes,
        "canonical_test_access": False,
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    register(
        repository,
        {
            "run_id": f"frozen_equal_logit_ensemble_{context}_seed{seed}",
            "experiment_type": "frozen_ensemble_validation_only",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "checkpoints": checkpoint_hashes,
            },
            "fold_definition": "published train checkpoints; published validation threshold",
            "seed": seed,
            "features": summary["components"],
            "hyperparameters": {
                "weights": summary["weights"],
                "batch_size": batch_size,
            },
            "artifact": str(summary_path.relative_to(repository)),
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
    parser.add_argument("--context", required=True)
    parser.add_argument(
        "--component",
        action="append",
        required=True,
        help="NAME=MODE=path/to/best.pth; repeat at least twice",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/frozen_ensemble"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    components = []
    for specification in args.component:
        parts = specification.split("=", 2)
        if len(parts) != 3 or not all(parts):
            parser.error(f"invalid --component value: {specification}")
        name, mode, raw_path = parts
        components.append((name, mode, resolve(Path(raw_path))))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(
        json.dumps(
            run(
                root,
                resolve(args.site_master),
                args.context,
                components,
                resolve(args.output_dir),
                args.batch_size,
                args.seed,
                device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
