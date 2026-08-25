"""Create outcome-blind coverage audits for experimental RBP features."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd


def count_lines(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return sum(1 for _ in handle)


def run(
    track_path: Path,
    manifest_path: Path,
    site_gene_path: Path,
    mappability_path: Path,
    output_dir: Path,
) -> dict:
    track_columns = [
        "gene",
        "cell_line",
        "experiment_accession",
        "file_accession",
        "legacy_archived_control",
        "focal_overlap",
        "focal_count_w25",
        "focal_count_w100",
        "partner_overlap",
        "duplex_peak_count",
        "duplex_coverage_fraction",
    ]
    tracks = pd.read_parquet(track_path, columns=track_columns)
    manifest = pd.read_csv(manifest_path)
    peak_counts = {
        row.file_accession: count_lines(Path(row.local_path))
        for row in manifest.itertuples(index=False)
    }
    grouped = tracks.groupby(
        [
            "gene",
            "cell_line",
            "experiment_accession",
            "file_accession",
            "legacy_archived_control",
        ],
        as_index=False,
    )
    track_audit = grouped.agg(
        site_count=("focal_overlap", "size"),
        focal_overlap_n=("focal_overlap", "sum"),
        focal_near25_n=("focal_count_w25", lambda values: (values > 0).sum()),
        focal_near100_n=("focal_count_w100", lambda values: (values > 0).sum()),
        partner_overlap_n=("partner_overlap", "sum"),
        duplex_any_n=("duplex_peak_count", lambda values: (values > 0).sum()),
        duplex_coverage_mean=("duplex_coverage_fraction", "mean"),
        duplex_coverage_max=("duplex_coverage_fraction", "max"),
    )
    track_audit["global_peak_count"] = track_audit["file_accession"].map(peak_counts)
    for numerator in [
        "focal_overlap_n",
        "focal_near25_n",
        "focal_near100_n",
        "partner_overlap_n",
        "duplex_any_n",
    ]:
        track_audit[f"{numerator[:-2]}_fraction"] = (
            track_audit[numerator] / track_audit["site_count"]
        )

    site_gene = pd.read_parquet(site_gene_path)
    gene_audit = site_gene.groupby("gene", as_index=False).agg(
        site_count=("pair_local_key", "size"),
        track_count=("eclip_track_count", "max"),
        cell_line_count=("eclip_cell_line_count", "max"),
        focal_overlap_n=("eclip_focal_overlap_any", "sum"),
        focal_near25_n=("eclip_focal_near_w25_any", "sum"),
        focal_near100_n=("eclip_focal_near_w100_any", "sum"),
        partner_overlap_n=("eclip_partner_overlap_any", "sum"),
        duplex_any_n=("eclip_duplex_peak_any", "sum"),
        duplex_coverage_mean=("eclip_duplex_coverage_fraction_max", "mean"),
        duplex_coverage_max=("eclip_duplex_coverage_fraction_max", "max"),
    )
    for numerator in [
        "focal_overlap_n",
        "focal_near25_n",
        "focal_near100_n",
        "partner_overlap_n",
        "duplex_any_n",
    ]:
        gene_audit[f"{numerator[:-2]}_fraction"] = (
            gene_audit[numerator] / gene_audit["site_count"]
        )

    mappability = pd.read_parquet(mappability_path)
    map_summary = (
        mappability.drop(columns=["pair_local_key", "site_id"])
        .describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
        .transpose()
        .reset_index(names="feature")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    track_audit.to_csv(output_dir / "eclip_track_coverage.csv", index=False)
    gene_audit.to_csv(output_dir / "eclip_gene_coverage.csv", index=False)
    map_summary.to_csv(output_dir / "umap100_summary.csv", index=False)
    summary = {
        "scope": "outcome-blind development feature coverage audit",
        "tracks": int(track_audit["file_accession"].nunique()),
        "genes": int(gene_audit["gene"].nunique()),
        "tracks_with_no_duplex_peak": int((track_audit["duplex_any_n"] == 0).sum()),
        "genes_with_any_focal_overlap": int((gene_audit["focal_overlap_n"] > 0).sum()),
        "genes_with_any_duplex_peak": int((gene_audit["duplex_any_n"] > 0).sum()),
        "ilf3_focal_overlap_sites": int(
            gene_audit.loc[gene_audit["gene"] == "ILF3", "focal_overlap_n"].iloc[0]
        ),
        "ilf3_duplex_sites": int(
            gene_audit.loc[gene_audit["gene"] == "ILF3", "duplex_any_n"].iloc[0]
        ),
        "umap100_focal_fully_unique_fraction": float(
            (mappability["umap100_at_focal"] == 1).mean()
        ),
        "eclip_pipeline_mapping": (
            "ENCODE eCLIP SOP v2.2 first maps/removes reads against human RepBase, "
            "then uses STAR genome mapping with --outFilterMultimapNmax 1"
        ),
        "conclusion": (
            "positive peaks are interpretable, but experimental peak absence in "
            "inverted Alu is not a reliable negative binding measurement"
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--track-features",
        type=Path,
        default=Path("data/derived/rbp/eclip_track_features.parquet"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("context/rbp/encode_eclip_manifest.csv")
    )
    parser.add_argument(
        "--site-gene",
        type=Path,
        default=Path("data/derived/features/eclip_site_gene.parquet"),
    )
    parser.add_argument(
        "--mappability",
        type=Path,
        default=Path("data/derived/genomic_mapping/umap100.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/adarpro/rbp_feature_audit")
    )
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    result = run(
        resolve(args.track_features),
        resolve(args.manifest),
        resolve(args.site_gene),
        resolve(args.mappability),
        resolve(args.output_dir),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
