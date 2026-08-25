"""Annotate development sites with independent GRCh38 100-mer Umap scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

SOURCE_URL = (
    "https://hgdownload.soe.ucsc.edu/gbdb/hg38/hoffmanMappability/"
    "k100.Umap.MultiTrackMappability.bw"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "mean": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "unique_fraction": 0.0,
            "missing_fraction": 1.0,
        }
    return {
        "mean": float(np.mean(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "unique_fraction": float(np.mean(np.isclose(finite, 1.0))),
        "missing_fraction": float(1 - len(finite) / len(values)),
    }


def run(site_master_path: Path, bigwig_path: Path, output_path: Path) -> dict:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("mappability input contains a sealed split")
    sites = master[
        ["pair_local_key", "site_id", "chromosome", "genomic_position_0based"]
    ].drop_duplicates()
    if sites["pair_local_key"].duplicated().any():
        raise RuntimeError("non-unique pair/local site in mappability input")
    bigwig = pyBigWig.open(str(bigwig_path))
    rows = []
    try:
        for row in sites.itertuples(index=False):
            position = int(row.genomic_position_0based)
            point = bigwig.values(row.chromosome, position, position + 1)[0]
            # A 100-nt genomic k-mer overlaps the focal base when its start is
            # in [position-99, position]. Umap is represented at k-mer starts.
            starts = bigwig.values(
                row.chromosome, max(0, position - 99), position + 1, numpy=True
            )
            values = np.full(100, np.nan)
            values[-len(starts) :] = starts
            summary = summarize(values)
            rows.append(
                {
                    "pair_local_key": row.pair_local_key,
                    "site_id": row.site_id,
                    "umap100_at_focal": point,
                    **{
                        f"umap100_overlapping_kmer_{key}": value
                        for key, value in summary.items()
                    },
                }
            )
    finally:
        bigwig.close()
    output = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    sha256 = sha256_file(bigwig_path)
    result = {
        "scope": "development train+valid only",
        "source_url": SOURCE_URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "genome_build": "GRCh38/hg38",
        "kmer_length": 100,
        "source_sha256": sha256,
        "source_bytes": bigwig_path.stat().st_size,
        "site_count": len(output),
        "fully_unique_focal_fraction": float(
            np.mean(np.isclose(output["umap100_at_focal"], 1.0))
        ),
        "citation": "Karimzadeh et al., Bioinformatics 2018, doi:10.1093/bioinformatics/bty149",
        "interpretation_limit": (
            "Umap measures genomic k-mer uniqueness; it does not undo the ENCODE "
            "eCLIP pipeline's prior removal of reads mapping to RepBase."
        ),
    }
    output_path.with_suffix(".provenance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument(
        "--bigwig",
        type=Path,
        default=Path(
            "data/external/mappability/k100.Umap.MultiTrackMappability.bw"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/genomic_mapping/umap100.parquet"),
    )
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(
        json.dumps(
            run(resolve(args.site_master), resolve(args.bigwig), resolve(args.output)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
