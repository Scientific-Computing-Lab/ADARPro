"""Cross-fitted substrate-wide profile from Baseline GAT predictions at every A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from adarpro.models.baseline_oof import torch_load, upstream_modules
from adarpro.quarantine import sha256_file
from adarpro.registry import register


@torch.no_grad()
def predict_graphs(model, graphs: list, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    values = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        output, _ = model(batch.to(device))
        values.extend(output.detach().cpu().view(-1).tolist())
    return np.asarray(values, dtype=float)


def summarize_profile(
    probabilities: np.ndarray, focal_index: int, threshold: float
) -> dict[str, float]:
    focal = probabilities[focal_index]
    others = np.delete(probabilities, focal_index)

    def summary(prefix: str, values: np.ndarray) -> dict[str, float]:
        top = np.sort(values)[-min(5, len(values)) :]
        return {
            f"substrate_profile__{prefix}_mean": float(np.mean(values)),
            f"substrate_profile__{prefix}_std": float(np.std(values)),
            f"substrate_profile__{prefix}_max": float(np.max(values)),
            f"substrate_profile__{prefix}_q75": float(np.quantile(values, 0.75)),
            f"substrate_profile__{prefix}_q90": float(np.quantile(values, 0.90)),
            f"substrate_profile__{prefix}_top5_mean": float(np.mean(top)),
            f"substrate_profile__{prefix}_fraction_above_threshold": float(
                np.mean(values >= threshold)
            ),
        }

    result = summary("all_a", probabilities)
    if len(others):
        result.update(summary("other_a", others))
    else:
        result.update(
            {
                f"substrate_profile__other_a_{name}": np.nan
                for name in (
                    "mean",
                    "std",
                    "max",
                    "q75",
                    "q90",
                    "top5_mean",
                    "fraction_above_threshold",
                )
            }
        )
    result.update(
        {
            "substrate_profile__a_count_log1p": float(np.log1p(len(probabilities))),
            "substrate_profile__focal_percentile": float(
                (np.sum(probabilities < focal) + 0.5 * np.sum(probabilities == focal))
                / len(probabilities)
            ),
            "substrate_profile__focal_zscore": float(
                (focal - np.mean(probabilities))
                / max(float(np.std(probabilities)), 1e-8)
            ),
        }
    )
    return result


def run(
    repository: Path,
    site_master_path: Path,
    baseline_dir: Path,
    context: str,
    output_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("substrate profiling received a sealed split")
    table = master[master["context"] == context].copy().reset_index(drop=True)
    if table["baseline_oof_probability"].isna().any():
        raise RuntimeError(f"incomplete baseline OOF for {context}")
    baseline = upstream_modules(repository)
    profiles: dict[int, dict] = {}
    checkpoint_hashes = {}
    for fold in range(5):
        pairs = table[table["development_fold"] == fold].drop_duplicates("pair_id")
        graphs, metadata = [], []
        for row in pairs.itertuples(index=False):
            for local_position, base in enumerate(row.sequence):
                if base.upper() != "A":
                    continue
                graph = baseline.create_rna_graph(
                    row.sequence, row.structure, local_position, 0
                )
                if graph is None:
                    raise RuntimeError(
                        f"unbuildable all-A graph: pair={row.pair_id} pos={local_position}"
                    )
                graphs.append(graph)
                metadata.append((int(row.pair_id), local_position))
        checkpoint_path = baseline_dir / f"outer_{fold}" / "best.pth"
        checkpoint_hashes[f"outer_{fold}"] = sha256_file(checkpoint_path)
        saved = torch_load(checkpoint_path, device)
        model = baseline.RNAEditingGNN(8, 32, 1, 4).to(device)
        model.load_state_dict(saved["model_state"])
        probabilities = predict_graphs(model, graphs, device, batch_size)
        threshold = float(saved["best"]["threshold"])
        by_pair: dict[int, list[tuple[int, float]]] = {}
        for (pair_id, position), probability in zip(metadata, probabilities):
            by_pair.setdefault(pair_id, []).append((position, float(probability)))
        for pair_id, values in by_pair.items():
            values.sort()
            profiles[pair_id] = {
                "positions": np.asarray([item[0] for item in values], dtype=int),
                "probabilities": np.asarray([item[1] for item in values], dtype=float),
                "threshold": threshold,
                "fold": fold,
            }
    rows = []
    maximum_focal_difference = 0.0
    for row in table.itertuples(index=False):
        profile = profiles[int(row.pair_id)]
        matches = np.flatnonzero(profile["positions"] == int(row.local_position))
        if len(matches) != 1:
            raise RuntimeError(f"canonical focal A absent from all-A profile: {row.example_id}")
        focal_index = int(matches[0])
        focal_probability = float(profile["probabilities"][focal_index])
        maximum_focal_difference = max(
            maximum_focal_difference,
            abs(focal_probability - float(row.baseline_oof_probability)),
        )
        values = {
            "example_id": row.example_id,
            "context": row.context,
            "pair_id": row.pair_id,
            "pair_local_key": row.pair_local_key,
            "substrate_profile__focal_probability_recomputed": focal_probability,
        }
        values.update(
            summarize_profile(
                profile["probabilities"], focal_index, profile["threshold"]
            )
        )
        rows.append(values)
    if maximum_focal_difference > 1e-5:
        raise RuntimeError(
            f"all-A/focal OOF prediction mismatch: {maximum_focal_difference}"
        )
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    summary = {
        "scope": "every adenosine on development substrates; fold-held-out Baseline GAT only",
        "context": context,
        "rows": len(result),
        "pairs": int(result["pair_id"].nunique()),
        "maximum_recomputed_focal_difference": maximum_focal_difference,
        "canonical_test_access": False,
    }
    (output_path.parent / f"{output_path.stem}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"substrate_profile_{context}",
            "experiment_type": "all_adenosine_oof_substrate_profile",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "checkpoints": checkpoint_hashes,
            },
            "fold_definition": "global pair fold k predicted only by outer-k held-out Baseline GAT",
            "seed": "inherited from baseline OOF checkpoints",
            "features": "distribution of Baseline GAT probabilities across every A on each substrate",
            "hyperparameters": {"batch_size": batch_size},
            "artifact": str(output_path.relative_to(repository)),
            "metrics": summary,
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--site-master", type=Path, default=Path("data/derived/site_master.parquet"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("experiments/runs/baseline_oof/MuscleSkeletal"))
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument("--output", type=Path, default=Path("data/derived/features/substrate_profile_MuscleSkeletal.parquet"))
    parser.add_argument("--batch-size", type=int, default=256)
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
                resolve(args.baseline_dir),
                args.context,
                resolve(args.output),
                args.batch_size,
                device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
