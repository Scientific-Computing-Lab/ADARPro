"""Map frozen hg38 PrismNet cell-aware binding predictions to development sites."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adarpro.rbp.eclip_features import (
    WINDOWS,
    build_indices,
    interval_features,
    prefix_features,
    prepare_sites,
    region_features,
)

DOWNLOAD_PAGE = "https://prismnetweb.zhanglab.net/download/"
ARCHIVES = {
    "K562": {
        "name": "PrismNet_predictions_K562.zip",
        "sha256": "b3b63413b0078e35c47e320e31546e5fbf1c44a931a5174dea7061d99c34e991",
        "bytes": 36_631_414,
    },
    "HepG2": {
        "name": "PrismNet_predictions_HepG2.zip",
        "sha256": "c110ec25f94997b49bd4333a529d257a73782b9f5153a3fa46438fe6f7c0c3a7",
        "bytes": 20_940_590,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_genes(archive: Path, cell_line: str) -> set[str]:
    suffix = f"_{cell_line}.bed"
    with zipfile.ZipFile(archive) as handle:
        return {
            Path(name).name.removesuffix(suffix)
            for name in handle.namelist()
            if name.endswith(suffix) and not name.startswith("__MACOSX/")
        }


def read_prediction(archive: Path, cell_line: str, gene: str) -> pd.DataFrame:
    member = f"PrismNet_predictions_{cell_line}/{gene}_{cell_line}.bed"
    columns = ["chromosome", "start", "end", "name", "signal", "strand"]
    with zipfile.ZipFile(archive) as handle, handle.open(member) as source:
        predictions = pd.read_csv(source, sep="\t", header=None, names=columns)
    predictions[["start", "end"]] = predictions[["start", "end"]].astype(int)
    predictions["signal"] = predictions["signal"].astype(float)
    if not predictions["strand"].isin(["+", "-"]).all():
        raise RuntimeError(f"invalid PrismNet strand in {member}")
    return predictions


def annotate_prediction(
    sites: pd.DataFrame, peaks: pd.DataFrame, gene: str, cell_line: str
) -> pd.DataFrame:
    indices = build_indices(peaks)
    pair_rows = sites.drop_duplicates("pair_id").set_index("pair_id")
    pair_cache = {}
    for pair_id, pair in pair_rows.iterrows():
        index = indices.get((pair["chromosome"], pair["strand"]))
        arm1 = (int(pair["start1"]), int(pair["end1"]) + 1)
        arm2 = (int(pair["start2"]), int(pair["end2"]) + 1)
        pair_cache[pair_id] = {
            **prefix_features("duplex", region_features(index, [arm1, arm2])),
            **prefix_features("arm1", region_features(index, [arm1])),
            **prefix_features("arm2", region_features(index, [arm2])),
        }
    rows = []
    for site in sites.itertuples(index=False):
        index = indices.get((site.chromosome, site.strand))
        focal = interval_features(index, int(site.genomic_position_0based))
        partner = (
            interval_features(index, int(site.partner_genomic_position_0based))
            if site.focal_is_paired
            else interval_features(None, -1)
        )
        arm = pair_cache[site.pair_id]
        row = {
            "pair_local_key": site.pair_local_key,
            "pair_id": int(site.pair_id),
            "site_id": site.site_id,
            "gene": gene,
            "cell_line": cell_line,
            "genome_build": "hg38",
            "prediction_semantics": "thresholded_PrismNet_binding_region_score",
            "focal_is_paired": bool(site.focal_is_paired),
            "focal_arm": int(site.focal_arm),
            "partner_arm": int(site.partner_arm),
            **prefix_features("focal", focal),
            **prefix_features("partner", partner),
            **arm,
        }
        for name in ("peak_count", "covered_nt", "coverage_fraction", "max_signal"):
            row[f"focal_arm_{name}"] = arm[f"arm{site.focal_arm}_{name}"]
            opposite = 2 if site.focal_arm == 1 else 1
            row[f"opposite_arm_{name}"] = arm[f"arm{opposite}_{name}"]
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_cells(features: pd.DataFrame) -> pd.DataFrame:
    work = features.copy()
    for feature in ["focal_overlap", "partner_overlap"]:
        work[f"{feature}_binary"] = (work[feature] > 0).astype(float)
    for window in WINDOWS:
        work[f"focal_near_w{window}_binary"] = (
            work[f"focal_count_w{window}"] > 0
        ).astype(float)
    work["duplex_peak_binary"] = (work["duplex_peak_count"] > 0).astype(float)
    work["nearest_valid"] = work["focal_nearest_distance"].where(
        work["focal_nearest_distance"] >= 0
    )
    aggregations: dict[str, tuple[str, str]] = {
        "prismnet_cell_count": ("cell_line", "nunique"),
        "prismnet_focal_overlap_any": ("focal_overlap_binary", "max"),
        "prismnet_focal_overlap_cell_fraction": ("focal_overlap_binary", "mean"),
        "prismnet_partner_overlap_any": ("partner_overlap_binary", "max"),
        "prismnet_partner_overlap_cell_fraction": ("partner_overlap_binary", "mean"),
        "prismnet_duplex_peak_any": ("duplex_peak_binary", "max"),
        "prismnet_duplex_peak_cell_fraction": ("duplex_peak_binary", "mean"),
        "prismnet_nearest_distance": ("nearest_valid", "min"),
        "prismnet_duplex_coverage_fraction_max": ("duplex_coverage_fraction", "max"),
        "prismnet_duplex_max_score": ("duplex_max_signal", "max"),
        "prismnet_focal_arm_coverage_fraction_max": (
            "focal_arm_coverage_fraction",
            "max",
        ),
        "prismnet_opposite_arm_coverage_fraction_max": (
            "opposite_arm_coverage_fraction",
            "max",
        ),
    }
    for window in WINDOWS:
        aggregations.update(
            {
                f"prismnet_focal_near_w{window}_any": (
                    f"focal_near_w{window}_binary",
                    "max",
                ),
                f"prismnet_focal_near_w{window}_cell_fraction": (
                    f"focal_near_w{window}_binary",
                    "mean",
                ),
                f"prismnet_focal_max_score_w{window}": (
                    f"focal_max_signal_w{window}",
                    "max",
                ),
                f"prismnet_focal_mean_score_w{window}": (
                    f"focal_mean_signal_w{window}",
                    "max",
                ),
            }
        )
    return work.groupby(["pair_local_key", "gene"], as_index=False).agg(**aggregations)


def run(
    site_master_path: Path,
    panel_path: Path,
    archive_dir: Path,
    output_dir: Path,
) -> dict:
    master = pd.read_parquet(site_master_path)
    sites = prepare_sites(master)
    panel_genes = set(pd.read_csv(panel_path)["gene"])
    tables, manifest_rows = [], []
    for cell_line, metadata in ARCHIVES.items():
        archive = archive_dir / metadata["name"]
        observed_hash = sha256_file(archive)
        if archive.stat().st_size != metadata["bytes"] or observed_hash != metadata["sha256"]:
            raise RuntimeError(f"PrismNet archive checksum mismatch: {archive}")
        genes = sorted(panel_genes & available_genes(archive, cell_line))
        for gene in genes:
            peaks = read_prediction(archive, cell_line, gene)
            tables.append(annotate_prediction(sites, peaks, gene, cell_line))
            manifest_rows.append(
                {
                    "gene": gene,
                    "cell_line": cell_line,
                    "prediction_regions": len(peaks),
                    "archive": metadata["name"],
                    "archive_sha256": observed_hash,
                    "genome_build": "hg38",
                }
            )
    features = pd.concat(tables, ignore_index=True)
    aggregate = aggregate_cells(features)
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_dir / "prismnet_cell_features.parquet", index=False)
    aggregate.to_parquet(output_dir / "prismnet_site_gene.parquet", index=False)
    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)
    result = {
        "scope": "development train+valid sites only; outcome-blind annotation",
        "source_download_page": DOWNLOAD_PAGE,
        "source_page_states_genome_build": "Human hg38",
        "archives": ARCHIVES,
        "cell_lines": sorted(features["cell_line"].unique()),
        "genes": sorted(features["gene"].unique()),
        "site_count": int(sites["pair_local_key"].nunique()),
        "cell_feature_rows": len(features),
        "aggregated_rows": len(aggregate),
        "windows_nt_prespecified": list(WINDOWS),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "downloaded BEDs are thresholded genome-wide binding-region predictions "
            "from sequence plus cell-matched in-vivo structure, not raw probabilities "
            "at every nucleotide"
        ),
    }
    (output_dir / "provenance.json").write_text(
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
        "--panel", type=Path, default=Path("context/rbp/candidate_panel.csv")
    )
    parser.add_argument(
        "--archive-dir", type=Path, default=Path("data/external/prismnet")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/derived/rbp/prismnet")
    )
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    result = run(
        resolve(args.site_master),
        resolve(args.panel),
        resolve(args.archive_dir),
        resolve(args.output_dir),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
