"""Run frozen RBPNet models once per development substrate and summarize sites."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.rbp.eclip_features import WINDOWS, prepare_sites

ZENODO_RECORD = "https://zenodo.org/records/10185223"
ZENODO_MD5 = "e88fb57483ba9a3d81b842cc5aa140fb"
ZENODO_BYTES = 836_278_121
MODEL_GENES = (
    "ILF3",
    "XRCC6",
    "TROVE2",
    "DROSHA",
    "MATR3",
    "HNRNPC",
    "HNRNPM",
    "RBFOX2",
    "TIA1",
    "IGF2BP1",
)


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_models(archive: Path, genes: tuple[str, ...], output_dir: Path) -> dict:
    if archive.stat().st_size != ZENODO_BYTES or file_digest(archive, "md5") != ZENODO_MD5:
        raise RuntimeError("RBPNet model archive does not match Zenodo record 10185223")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        for gene in genes:
            member = f"models/{gene}_HepG2.model.h5"
            if member not in names:
                raise RuntimeError(f"RBPNet model unavailable: {member}")
            path = output_dir / Path(member).name
            if not path.exists():
                with handle.open(member) as source, path.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
            rows[gene] = {
                "path": str(path.resolve()),
                "sha256": file_digest(path, "sha256"),
                "bytes": path.stat().st_size,
            }
    return rows


def batches(values: list[int], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def predict_gene(model_path: Path, pairs: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    from rbpnet.models import load_model

    gene = model_path.name.split("_", 1)[0]
    target = f"{gene}_HepG2"
    model = load_model(str(model_path))
    output_rows = []
    for _, group in pairs.groupby(pairs["sequence"].str.len(), sort=True):
        indices = group.index.tolist()
        for batch_indices in batches(indices, batch_size):
            batch = pairs.loc[batch_indices]
            predictions = model.predict_from_sequence(batch["sequence"].tolist())
            profile = np.asarray(predictions[f"{target}_profile"].numpy())
            target_profile = np.asarray(
                predictions[f"{target}_profile_target"].numpy()
            )
            mixing = np.asarray(
                predictions[f"{target}_mixing_coefficient"].numpy()
            ).reshape(-1)
            if profile.ndim == 1:
                profile, target_profile = profile[None, :], target_profile[None, :]
            for pair, combined, specific, coefficient in zip(
                batch.itertuples(index=False), profile, target_profile, mixing
            ):
                output_rows.append(
                    {
                        "pair_id": int(pair.pair_id),
                        "gene": gene,
                        "sequence_sha256": hashlib.sha256(
                            pair.sequence.encode()
                        ).hexdigest(),
                        "profile": combined.astype(np.float32).tolist(),
                        "profile_target": specific.astype(np.float32).tolist(),
                        "mixing_coefficient": float(coefficient),
                    }
                )
    return pd.DataFrame(output_rows).sort_values("pair_id").reset_index(drop=True)


def window_summary(values: np.ndarray, position: int, prefix: str) -> dict[str, float]:
    output = {f"{prefix}_value": float(values[position])}
    for window in WINDOWS:
        left, right = max(0, position - window), min(len(values), position + window + 1)
        region = values[left:right]
        output[f"{prefix}_max_w{window}"] = float(np.max(region))
        output[f"{prefix}_mean_w{window}"] = float(np.mean(region))
        output[f"{prefix}_mass_w{window}"] = float(np.sum(region))
    return output


def entropy(values: np.ndarray) -> float:
    positive = values[values > 0]
    return float(-np.sum(positive * np.log(positive)))


def summarize_sites(
    sites: pd.DataFrame, pair_profiles: pd.DataFrame
) -> pd.DataFrame:
    by_pair = {int(row.pair_id): row for row in pair_profiles.itertuples(index=False)}
    rows = []
    for site in sites.itertuples(index=False):
        pair = by_pair[int(site.pair_id)]
        combined = np.asarray(pair.profile, dtype=float)
        specific = np.asarray(pair.profile_target, dtype=float)
        if len(combined) != len(site.structure) or len(specific) != len(site.structure):
            raise RuntimeError(f"RBPNet profile length mismatch for pair {site.pair_id}")
        focal = int(site.local_position)
        partner = int(site.partner_local_position)
        first_length = int(site.end1 - site.start1 + 1)
        second_length = int(site.end2 - site.start2 + 1)
        if site.strand == "+":
            arm_slices = {1: slice(0, first_length), 2: slice(first_length + 10, None)}
        else:
            arm_slices = {2: slice(0, second_length), 1: slice(second_length + 10, None)}
        row = {
            "pair_local_key": site.pair_local_key,
            "pair_id": int(site.pair_id),
            "site_id": site.site_id,
            "gene": pair.gene,
            "cell_line": "HepG2",
            "model_output_semantics": "relative_CLIP_signal_profile_not_occupancy_probability",
            "mixing_coefficient": float(pair.mixing_coefficient),
            "focal_is_paired": bool(site.focal_is_paired),
        }
        for layer_name, values in (("profile", combined), ("target", specific)):
            row.update(window_summary(values, focal, f"{layer_name}_focal"))
            if partner >= 0:
                row.update(window_summary(values, partner, f"{layer_name}_partner"))
            else:
                for name in window_summary(values, focal, f"{layer_name}_partner"):
                    row[name] = math.nan
            focal_arm_values = values[arm_slices[int(site.focal_arm)]]
            opposite = 2 if int(site.focal_arm) == 1 else 1
            opposite_values = values[arm_slices[opposite]]
            row[f"{layer_name}_duplex_max"] = float(np.max(values))
            row[f"{layer_name}_duplex_entropy"] = entropy(values)
            row[f"{layer_name}_focal_arm_mass"] = float(np.sum(focal_arm_values))
            row[f"{layer_name}_opposite_arm_mass"] = float(np.sum(opposite_values))
            row[f"{layer_name}_arm_mass_asymmetry"] = float(
                np.sum(focal_arm_values) - np.sum(opposite_values)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run(
    site_master_path: Path,
    archive_path: Path,
    model_dir: Path,
    output_dir: Path,
    genes: tuple[str, ...],
    batch_size: int,
) -> dict:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("RBPNet input contains a sealed split")
    sites = prepare_sites(master)
    pairs = sites[["pair_id", "structure"]].drop_duplicates("pair_id").rename(
        columns={"structure": "sequence_placeholder"}
    )
    sequences = master[["pair_id", "sequence"]].drop_duplicates()
    pairs = pairs.drop(columns="sequence_placeholder").merge(
        sequences, on="pair_id", validate="one_to_one"
    )
    if pairs["sequence"].str.contains("[^ACGTN]").any():
        raise RuntimeError("RBPNet input includes an unsupported nucleotide")
    models = extract_models(archive_path, genes, model_dir)
    pair_dir = output_dir / "pair_profiles"
    pair_dir.mkdir(parents=True, exist_ok=True)
    site_tables = []
    for gene in genes:
        pair_path = pair_dir / f"{gene}.parquet"
        if pair_path.exists():
            profiles = pd.read_parquet(pair_path)
        else:
            profiles = predict_gene(Path(models[gene]["path"]), pairs, batch_size)
            profiles.to_parquet(pair_path, index=False)
        if len(profiles) != len(pairs):
            raise RuntimeError(f"incomplete RBPNet pair cache for {gene}")
        site_tables.append(summarize_sites(sites, profiles))
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
        except ImportError:
            pass
        gc.collect()
        print(f"[{gene}] predicted {len(pairs)} pairs", flush=True)
    site_features = pd.concat(site_tables, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    site_path = output_dir / "rbpnet_site_features.parquet"
    site_features.to_parquet(site_path, index=False)
    provenance = {
        "scope": "development train+valid pairs only; no outcomes used",
        "source": ZENODO_RECORD,
        "zenodo_doi": "10.5281/zenodo.10185223",
        "archive_md5": ZENODO_MD5,
        "archive_sha256": file_digest(archive_path, "sha256"),
        "archive_bytes": archive_path.stat().st_size,
        "models": models,
        "genes": list(genes),
        "pair_count": len(pairs),
        "site_count": int(sites["pair_local_key"].nunique()),
        "site_feature_rows": len(site_features),
        "windows_nt_prespecified": list(WINDOWS),
        "output_semantics": (
            "RBPNet predicts the relative distribution of CLIP signal along each "
            "input sequence; values are not calibrated binding probabilities"
        ),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument("--models-zip", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("data/external/rbpnet/models")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/derived/rbp/rbpnet")
    )
    parser.add_argument("--genes", nargs="+", choices=MODEL_GENES, default=MODEL_GENES)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    provenance = run(
        resolve(args.site_master),
        resolve(args.models_zip),
        resolve(args.model_dir),
        resolve(args.output_dir),
        tuple(args.genes),
        args.batch_size,
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
