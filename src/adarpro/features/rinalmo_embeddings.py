"""Cache outcome-independent RiNALMo embeddings for development substrates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adarpro.quarantine import sha256_file
from adarpro.registry import register


def pool_representation(
    representation: np.ndarray,
    sequence_length: int,
    target_position: int,
    local_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target, sequence-mean and target-local means with token alignment."""

    if representation.ndim != 2 or representation.shape[0] < sequence_length + 2:
        raise ValueError("RiNALMo representation does not contain CLS/sequence/EOS")
    if not 0 <= target_position < sequence_length:
        raise ValueError("target position is outside the sequence")
    sequence_tokens = representation[1 : sequence_length + 1]
    start = max(0, target_position - local_radius)
    stop = min(sequence_length, target_position + local_radius + 1)
    return (
        sequence_tokens[target_position],
        sequence_tokens.mean(axis=0),
        sequence_tokens[start:stop].mean(axis=0),
    )


def load_model(checkpoint: Path, model_size: str, device: torch.device):
    """Load a pinned official RiNALMo architecture and checkpoint."""

    from rinalmo.config import model_config
    from rinalmo.data.alphabet import Alphabet
    from rinalmo.model.model import RiNALMo

    config = model_config(model_size)
    model = RiNALMo(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), Alphabet(**config["alphabet"])


def run(
    repository: Path,
    site_master_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    context: str,
    model_size: str,
    batch_size: int,
    local_radius: int,
    device: torch.device,
) -> dict:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("RiNALMo cache received a sealed split")
    sites = (
        master[master["context"] == context]
        .sort_values("example_id")
        .reset_index(drop=True)
    )
    if sites.empty:
        raise RuntimeError(f"no development sites for {context}")
    pair_nunique = sites.groupby("pair_id")[["sequence", "structure"]].nunique()
    if (pair_nunique > 1).any().any():
        raise RuntimeError("a pair has inconsistent intrinsic representations")
    if not all(
        sequence[int(position)].upper() == "A"
        for sequence, position in zip(sites["sequence"], sites["local_position"])
    ):
        raise RuntimeError("a focal position is not adenosine in model orientation")

    pairs = (
        sites[["pair_id", "sequence"]]
        .drop_duplicates("pair_id")
        .sort_values("pair_id")
        .reset_index(drop=True)
    )
    model, alphabet = load_model(checkpoint_path, model_size, device)
    target_embeddings = None
    global_embeddings = None
    local_embeddings = None
    rows_by_pair = sites.groupby("pair_id").indices
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs.iloc[start : start + batch_size]
            sequences = batch["sequence"].tolist()
            tokens = torch.tensor(
                alphabet.batch_tokenize(sequences), dtype=torch.int64, device=device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                representations = model(tokens)["representation"]
            representations = representations.float().cpu().numpy()
            if target_embeddings is None:
                width = representations.shape[-1]
                target_embeddings = np.empty((len(sites), width), dtype=np.float16)
                global_embeddings = np.empty((len(sites), width), dtype=np.float16)
                local_embeddings = np.empty((len(sites), width), dtype=np.float16)
            for offset, row in enumerate(batch.itertuples(index=False)):
                pair_rows = np.asarray(rows_by_pair[row.pair_id])
                for site_index in pair_rows:
                    target, global_mean, local_mean = pool_representation(
                        representations[offset],
                        len(row.sequence),
                        int(sites.iloc[site_index]["local_position"]),
                        local_radius,
                    )
                    target_embeddings[site_index] = target
                    global_embeddings[site_index] = global_mean
                    local_embeddings[site_index] = local_mean
            print(
                f"[RiNALMo {model_size} {context}] "
                f"{min(start + batch_size, len(pairs))}/"
                f"{len(pairs)} pairs",
                flush=True,
            )
    if target_embeddings is None:
        raise RuntimeError("RiNALMo cache produced no embeddings")
    if not all(
        np.isfinite(values).all()
        for values in (target_embeddings, global_embeddings, local_embeddings)
    ):
        raise RuntimeError("RiNALMo cache contains non-finite values")

    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / "embeddings.npz"
    np.savez_compressed(
        array_path,
        example_id=sites["example_id"].to_numpy(dtype=str),
        target=target_embeddings,
        global_mean=global_embeddings,
        local_mean=local_embeddings,
    )
    index_columns = [
        "example_id",
        "context",
        "pair_id",
        "pair_local_key",
        "local_position",
        "development_fold",
        "source_split",
    ]
    index_path = output_dir / "index.parquet"
    sites[index_columns].to_parquet(index_path, index=False)
    summary = {
        "context": context,
        "model_size": model_size,
        "sites": len(sites),
        "pairs": int(sites["pair_id"].nunique()),
        "embedding_width": int(target_embeddings.shape[1]),
        "local_radius": local_radius,
        "dtype": str(target_embeddings.dtype),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "array_sha256": sha256_file(array_path),
        "index_sha256": sha256_file(index_path),
        "canonical_test_access": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"rinalmo_{model_size}_embedding_cache_{context}",
            "experiment_type": "outcome_independent_foundation_embedding_cache",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "checkpoint": sha256_file(checkpoint_path),
            },
            "fold_definition": "development train+valid pairs only; no fitting",
            "seed": None,
            "features": "frozen final-layer target, sequence-mean and +/- local mean embeddings",
            "hyperparameters": {
                "model": f"RiNALMo {model_size}",
                "batch_size": batch_size,
                "local_radius": local_radius,
                "precision": "CUDA autocast float16; pooled cache float16",
            },
            "artifact": str(array_path.relative_to(repository)),
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
        "--checkpoint",
        type=Path,
        default=Path("data/external/rinalmo/rinalmo_micro_pretrained.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/foundation/rinalmo_micro/MuscleSkeletal"),
    )
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument("--model-size", choices=("micro", "mega"), default="micro")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--local-radius", type=int, default=16)
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
                resolve(args.checkpoint),
                resolve(args.output_dir),
                args.context,
                args.model_size,
                args.batch_size,
                args.local_radius,
                device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
