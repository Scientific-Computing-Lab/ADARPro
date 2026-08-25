"""Fixed equal-logit ensemble of completed grouped-OOF neural runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
from adarpro.models.multitask_oof import (
    MultiTaskGAT,
    attach_local_windows,
    predict_binary,
)
from adarpro.quarantine import sha256_file
from adarpro.registry import register


def equal_logit_average_many(probabilities: list[np.ndarray]) -> np.ndarray:
    """Average logits with fixed equal weights and return probabilities."""

    if len(probabilities) < 2:
        raise ValueError("an ensemble requires at least two probability arrays")
    epsilon = np.finfo(np.float64).eps
    arrays = [np.asarray(values, dtype=np.float64) for values in probabilities]
    if any(values.shape != arrays[0].shape for values in arrays[1:]):
        raise ValueError("ensemble probability arrays have different shapes")
    clipped = [np.clip(values, epsilon, 1 - epsilon) for values in arrays]
    mean_logit = np.mean(
        [np.log(values / (1 - values)) for values in clipped], axis=0
    )
    return 1 / (1 + np.exp(-mean_logit))


def equal_logit_average(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Average two logits with a fixed 50:50 weight and return probabilities."""

    return equal_logit_average_many([left, right])


def model_for_mode(mode: str, device: torch.device) -> MultiTaskGAT:
    return MultiTaskGAT(
        use_local=mode.endswith("_local"),
        use_target_pool=mode.endswith("_target_pool"),
        use_anchor_pool=mode.endswith("_anchor_pool"),
        use_duplicate_pool=mode.endswith("_duplicate_pool"),
        use_pair_context="_context" in mode,
    ).to(device)


def predict_checkpoint(
    checkpoint_path: Path,
    mode: str,
    inner_loader: DataLoader,
    outer_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    saved = torch_load(checkpoint_path, device)
    model = model_for_mode(mode, device)
    model.load_state_dict(saved["model_state"])
    inner_probability, inner_labels = predict_binary(model, inner_loader, device)
    outer_probability, outer_labels = predict_binary(model, outer_loader, device)
    return inner_probability, inner_labels, outer_probability, outer_labels


def run_components(
    repository: Path,
    site_master_path: Path,
    components: list[tuple[str, str, Path]],
    context: str,
    output_dir: Path,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    if len(components) < 2:
        raise ValueError("an ensemble requires at least two components")
    names = [name for name, _, _ in components]
    if len(names) != len(set(names)):
        raise ValueError("ensemble component names must be unique")
    if any(mode.endswith("_shifted_local") for _, mode, _ in components):
        raise ValueError("shifted-local controls cannot be ensemble components")
    if any("_context" in mode for _, mode, _ in components):
        raise ValueError("pair-context ensembles require an explicit graph adapter")
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("fixed ensemble received a sealed split")
    canonical = master[master["context"] == context].copy().reset_index(drop=True)
    if canonical["baseline_oof_probability"].isna().any():
        raise RuntimeError(f"development reference is incomplete for {context}")
    graphs = build_graphs(canonical, upstream_modules(repository))
    # Local attributes are ignored by graph-only components. Adding the
    # target-aligned window once lets local and graph-only checkpoints share
    # the exact same ordered loaders without duplicating graph construction.
    if any(mode.endswith("_local") for _, mode, _ in components):
        attach_local_windows(graphs, canonical)
    outputs, selections, checkpoint_hashes = [], [], {}
    for outer_fold in range(5):
        inner_fold = (outer_fold + 1) % 5
        inner_indices = canonical.index[
            canonical["development_fold"] == inner_fold
        ].to_numpy()
        outer_indices = canonical.index[
            canonical["development_fold"] == outer_fold
        ].to_numpy()
        inner_loader = DataLoader(
            [graphs[index] for index in inner_indices],
            batch_size=batch_size,
            shuffle=False,
        )
        outer_loader = DataLoader(
            [graphs[index] for index in outer_indices],
            batch_size=batch_size,
            shuffle=False,
        )
        component_predictions = {}
        labels = None
        for name, mode, run_dir in components:
            checkpoint_path = run_dir / f"outer_{outer_fold}" / "best.pth"
            checkpoint_hashes[f"{name}_outer_{outer_fold}"] = sha256_file(
                checkpoint_path
            )
            inner_p, inner_y, outer_p, outer_y = predict_checkpoint(
                checkpoint_path, mode, inner_loader, outer_loader, device
            )
            saved_outer_path = (
                run_dir / f"outer_{outer_fold}" / "outer_predictions.parquet"
            )
            saved_outer = pd.read_parquet(saved_outer_path)
            expected_ids = canonical.iloc[outer_indices]["example_id"].to_numpy()
            if not np.array_equal(saved_outer["example_id"].to_numpy(), expected_ids):
                raise RuntimeError(f"{name} saved outer row order mismatch")
            if not np.allclose(
                saved_outer["probability"].to_numpy(), outer_p, rtol=0, atol=1e-5
            ):
                raise RuntimeError(
                    f"{name} checkpoint does not reproduce saved outer probabilities"
                )
            if labels is None:
                labels = (inner_y, outer_y)
            elif not np.array_equal(labels[0], inner_y) or not np.array_equal(
                labels[1], outer_y
            ):
                raise RuntimeError("component prediction label mismatch")
            component_predictions[name] = (inner_p, outer_p)
        inner_probability = equal_logit_average_many(
            [component_predictions[name][0] for name in names]
        )
        outer_probability = equal_logit_average_many(
            [component_predictions[name][1] for name in names]
        )
        threshold, inner_f1 = choose_threshold(labels[0], inner_probability)
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
        if not np.array_equal(outer["label"].to_numpy(), labels[1]):
            raise RuntimeError("ensemble outer prediction order mismatch")
        outer["probability"] = outer_probability
        outer["threshold"] = threshold
        outer["prediction"] = (outer_probability >= threshold).astype(int)
        for name in names:
            outer[f"{name}_probability"] = component_predictions[name][1]
        outputs.append(outer)
        selections.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "threshold": threshold,
                "inner_f1": inner_f1,
            }
        )
    predictions = pd.concat(outputs, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "oof_predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    pd.DataFrame(selections).to_csv(output_dir / "fold_selections.csv", index=False)
    summary = {
        "scope": "canonical train+valid grouped OOF only",
        "context": context,
        "ensemble": f"fixed equal-logit average of {len(components)} components",
        "components": {name: mode for name, mode, _ in components},
        "rows": len(predictions),
        "pairs": int(predictions["pair_id"].nunique()),
        "candidate_metrics": metric_row(
            predictions["label"], predictions["probability"], predictions["prediction"]
        ),
        "development_reference_metrics": metric_row(
            predictions["label"],
            predictions["baseline_oof_probability"],
            predictions["baseline_oof_prediction"],
        ),
        "canonical_test_access": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": (
                "fixed_logit_ensemble_"
                + "_".join(mode for _, mode, _ in components)
                + f"_{context}_seed{seed}"
            ),
            "experiment_type": "fixed_checkpoint_ensemble_grouped_oof",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "checkpoints": checkpoint_hashes,
            },
            "fold_definition": "five global pair folds; threshold from inner (k+1)%5",
            "seed": seed,
            "features": "fixed equal-logit ensemble of "
            + ", ".join(mode for _, mode, _ in components),
            "hyperparameters": {
                "weights": [1 / len(components)] * len(components),
                "batch_size": batch_size,
            },
            "artifact": str(prediction_path.relative_to(repository)),
            "metrics": summary,
            "canonical_test_access": False,
        },
    )
    return summary


def run(
    repository: Path,
    site_master_path: Path,
    left_run_dir: Path,
    left_mode: str,
    right_run_dir: Path,
    right_mode: str,
    context: str,
    output_dir: Path,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Backward-compatible two-component ensemble entry point."""

    summary = run_components(
        repository,
        site_master_path,
        [("left", left_mode, left_run_dir), ("right", right_mode, right_run_dir)],
        context,
        output_dir,
        batch_size,
        seed,
        device,
    )
    summary["ensemble"] = "fixed 50:50 average in logit space"
    summary["left_mode"] = left_mode
    summary["right_mode"] = right_mode
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument("--left-run-dir", type=Path)
    parser.add_argument("--left-mode")
    parser.add_argument("--right-run-dir", type=Path)
    parser.add_argument("--right-mode")
    parser.add_argument(
        "--component",
        action="append",
        help="NAME=MODE=run/directory; repeat for a fixed equal-weight ensemble",
    )
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.component:
        components = []
        for specification in args.component:
            parts = specification.split("=", 2)
            if len(parts) != 3 or not all(parts):
                parser.error(f"invalid --component value: {specification}")
            name, mode, raw_path = parts
            components.append((name, mode, resolve(Path(raw_path))))
        result = run_components(
            root,
            resolve(args.site_master),
            components,
            args.context,
            resolve(args.output_dir),
            args.batch_size,
            args.seed,
            device,
        )
    else:
        required = {
            "--left-run-dir": args.left_run_dir,
            "--left-mode": args.left_mode,
            "--right-run-dir": args.right_run_dir,
            "--right-mode": args.right_mode,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            parser.error("missing two-component arguments: " + ", ".join(missing))
        result = run(
            root,
            resolve(args.site_master),
            resolve(args.left_run_dir),
            args.left_mode,
            resolve(args.right_run_dir),
            args.right_mode,
            args.context,
            resolve(args.output_dir),
            args.batch_size,
            args.seed,
            device,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
