"""Recover and exhaustively validate GRCh38 coordinates for development sites."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta

from adarpro.quarantine import sha256_file

CONTEXT_TO_RAW = {
    "Artery": "Artery_Site_in_PairAlu_cov100.csv",
    "Brain": "Brain_Site_in_PairAlu_cov100.csv",
    "Liver": "Liver_Site_in_PairAlu_cov100.csv",
    "MuscleSkeletal": "MuscleSkeletal_Site_in_PairAlu_cov100.csv",
    "Combined": "Combined_Site_in_PairAlu_cov100.csv",
}


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def sequence_key(structure: str, left: str, right: str) -> str:
    payload = f"{structure}\0{left}\0{right}".encode()
    return hashlib.sha256(payload).hexdigest()


def read_bed(path: Path) -> pd.DataFrame:
    columns = ["chr1", "start1", "end1", "chr2", "start2", "end2", "strand"]
    bed = pd.read_csv(path, sep="\t", header=None, names=columns)
    if len(bed) != 905:
        raise RuntimeError(f"expected 905 Alu pairs, found {len(bed)}")
    if not (bed["chr1"] == bed["chr2"]).all():
        raise RuntimeError("cross-chromosome Alu pair encountered")
    if not bed["strand"].isin(["+", "-"]).all():
        raise RuntimeError("invalid strand in pair BED")
    bed["bed_row"] = np.arange(len(bed), dtype=int)
    return bed


def reconstruct_bed_sequences(bed: pd.DataFrame, fasta_path: Path) -> pd.DataFrame:
    reference = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    rows = []
    for row in bed.itertuples(index=False):
        first = str(reference[row.chr1][row.start1 : row.end1 + 1])
        second = str(reference[row.chr2][row.start2 : row.end2 + 1])
        genomic = first + "N" * 10 + second
        sequence = genomic if row.strand == "+" else reverse_complement(genomic)
        rows.append(
            {
                "bed_row": row.bed_row,
                "chr1": row.chr1,
                "start1": row.start1,
                "end1": row.end1,
                "chr2": row.chr2,
                "start2": row.start2,
                "end2": row.end2,
                "strand": row.strand,
                "sequence": sequence,
            }
        )
    return pd.DataFrame(rows)


def map_pairs(sites: pd.DataFrame, bed_sequences: pd.DataFrame) -> pd.DataFrame:
    canonical = sites[["pair_id", "sequence"]].drop_duplicates()
    if canonical["pair_id"].duplicated().any() or canonical["sequence"].duplicated().any():
        raise RuntimeError("canonical pair-to-sequence mapping is not one-to-one")
    matched_bed = bed_sequences[bed_sequences["sequence"].isin(canonical["sequence"])].copy()
    ambiguous = matched_bed[matched_bed["sequence"].duplicated(keep=False)]
    if not ambiguous.empty:
        raise RuntimeError(
            "development substrate sequence maps to multiple genomic loci; "
            f"BED rows={ambiguous['bed_row'].tolist()}"
        )
    merged = canonical.merge(
        matched_bed, on="sequence", how="left", validate="one_to_one"
    )
    if merged["bed_row"].isna().any():
        missing = merged.loc[merged["bed_row"].isna(), "pair_id"].tolist()
        raise RuntimeError(f"{len(missing)} development pairs do not map to BED: {missing[:10]}")
    return merged.drop(columns="sequence").sort_values("pair_id").reset_index(drop=True)


def read_raw_matches(raw_path: Path, required_keys: set[str]) -> pd.DataFrame:
    columns = [
        "PositionChr",
        "PositionStart",
        "structure",
        "L",
        "R",
        "EditingIndex",
    ]
    matches = []
    for chunk in pd.read_csv(raw_path, usecols=columns, chunksize=20_000):
        chunk = chunk.dropna(subset=["structure", "L", "R", "PositionStart"])
        keys = [
            sequence_key(str(structure), str(left).upper(), str(right).upper())
            for structure, left, right in zip(
                chunk["structure"], chunk["L"], chunk["R"]
            )
        ]
        chunk = chunk.assign(raw_key=keys)
        selected = chunk[chunk["raw_key"].isin(required_keys)]
        if not selected.empty:
            matches.append(selected)
    if not matches:
        raise RuntimeError(f"no canonical records found in {raw_path}")
    output = pd.concat(matches, ignore_index=True)
    duplicates = output[output["raw_key"].duplicated(keep=False)]
    if not duplicates.empty:
        variability = duplicates.groupby("raw_key").agg(
            chromosomes=("PositionChr", "nunique"),
            positions=("PositionStart", "nunique"),
            levels=("EditingIndex", "nunique"),
        )
        if (variability > 1).any(axis=None):
            raise RuntimeError(f"ambiguous duplicate raw records in {raw_path}")
        output = output.drop_duplicates("raw_key")
    return output


def expected_genomic_zero_based(site: pd.Series) -> int:
    local = int(site["local_position"])
    first_length = int(site["end1"] - site["start1"] + 1)
    second_length = int(site["end2"] - site["start2"] + 1)
    linker_length = 10
    if site["strand"] == "+":
        if local < first_length:
            return int(site["start1"] + local)
        second_offset = local - first_length - linker_length
        if not 0 <= second_offset < second_length:
            raise RuntimeError(f"focal position lies in/outside linker: {site['example_id']}")
        return int(site["start2"] + second_offset)
    if local < second_length:
        return int(site["end2"] - local)
    first_offset = local - second_length - linker_length
    if not 0 <= first_offset < first_length:
        raise RuntimeError(f"focal position lies in/outside linker: {site['example_id']}")
    return int(site["end1"] - first_offset)


def validate_reference_sites(sites: pd.DataFrame, fasta_path: Path) -> None:
    reference = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    unique = sites[["chromosome", "genomic_position_0based", "strand", "focal_base"]].drop_duplicates()
    bad = []
    for row in unique.itertuples(index=False):
        genomic = str(reference[row.chromosome][row.genomic_position_0based : row.genomic_position_0based + 1])
        transcript = genomic if row.strand == "+" else reverse_complement(genomic)
        if transcript != row.focal_base:
            bad.append((row.chromosome, row.genomic_position_0based, row.strand, genomic, transcript))
    if bad:
        raise RuntimeError(f"{len(bad)} focal bases disagree with GRCh38; first={bad[0]}")


def reconstruct(
    site_master_path: Path,
    raw_dir: Path,
    bed_path: Path,
    fasta_path: Path,
    pair_output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sites = pd.read_parquet(site_master_path)
    if set(sites["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("site master contains a sealed split")
    bed = read_bed(bed_path)
    pair_mapping = map_pairs(sites, reconstruct_bed_sequences(bed, fasta_path))
    sites = sites.drop(
        columns=[
            "site_id",
            "chromosome",
            "genomic_position_1based",
            "genomic_position_0based",
            "strand",
            "genome_build",
            "editing_index",
            "bed_row",
            "chr1",
            "start1",
            "end1",
            "chr2",
            "start2",
            "end2",
        ],
        errors="ignore",
    ).merge(pair_mapping, on="pair_id", how="left", validate="many_to_one")
    context_tables = []
    for context, filename in CONTEXT_TO_RAW.items():
        subset = sites[sites["context"] == context].copy()
        subset["raw_key"] = [
            sequence_key(structure, left, right)
            for structure, left, right in zip(
                subset["structure"], subset["left_sequence"], subset["right_sequence"]
            )
        ]
        raw_path = raw_dir / filename
        raw = read_raw_matches(raw_path, set(subset["raw_key"]))
        subset = subset.merge(
            raw[["raw_key", "PositionChr", "PositionStart", "EditingIndex"]],
            on="raw_key",
            how="left",
            validate="many_to_one",
        )
        if subset["PositionStart"].isna().any():
            raise RuntimeError(f"unmapped canonical raw records for {context}")
        context_tables.append(subset)
    sites = pd.concat(context_tables, ignore_index=True)
    if not (sites["PositionChr"] == sites["chr1"]).all():
        raise RuntimeError("raw chromosome and BED chromosome disagree")
    sites["genomic_position_0based"] = sites.apply(expected_genomic_zero_based, axis=1)
    if not np.array_equal(
        sites["genomic_position_0based"].to_numpy(),
        sites["PositionStart"].astype(int).to_numpy(),
    ):
        mismatch = sites[
            sites["genomic_position_0based"] != sites["PositionStart"].astype(int)
        ]
        raise RuntimeError(f"{len(mismatch)} raw coordinates disagree with construction transform")
    sites["chromosome"] = sites["chr1"]
    sites["genomic_position_1based"] = sites["genomic_position_0based"] + 1
    sites["site_id"] = (
        sites["chromosome"]
        + ":"
        + sites["genomic_position_1based"].astype(str)
        + ":"
        + sites["strand"]
    )
    sites["genome_build"] = "GRCh38"
    sites["editing_index"] = sites["EditingIndex"].astype(float)
    positive_bad = sites[(sites["label"] == 1) & (sites["editing_index"] < 15.0)]
    negative_bad = sites[(sites["label"] == 0) & (sites["editing_index"] >= 1.0)]
    if len(positive_bad) or len(negative_bad):
        raise RuntimeError(
            f"editing-index labels disagree: positive={len(positive_bad)}, negative={len(negative_bad)}"
        )
    validate_reference_sites(sites, fasta_path)
    identity = sites.groupby("pair_local_key").agg(
        site_ids=("site_id", "nunique"),
        positions=("genomic_position_1based", "nunique"),
        strands=("strand", "nunique"),
    )
    if (identity != 1).any(axis=None):
        raise RuntimeError("cross-context pair/local identity maps inconsistently")
    sites = sites.drop(columns=["raw_key", "PositionChr", "PositionStart", "EditingIndex"])
    site_master_path.parent.mkdir(parents=True, exist_ok=True)
    pair_output.parent.mkdir(parents=True, exist_ok=True)
    sites.to_parquet(site_master_path, index=False)
    pair_mapping.to_csv(pair_output, index=False)
    summary = {
        "scope": "canonical train+valid only; held-out test excluded",
        "mapped_rows": len(sites),
        "contexts": sites.groupby("context").size().astype(int).to_dict(),
        "source_splits": sites.groupby("source_split").size().astype(int).to_dict(),
        "mapped_pairs": int(pair_mapping["pair_id"].nunique()),
        "mapped_sites": int(sites["site_id"].nunique()),
        "pair_local_to_site_id_one_to_one": True,
        "raw_coordinate_transform_matches": len(sites),
        "reference_focal_adenosines_validated": int(sites["site_id"].nunique()),
        "genome_build": "GRCh38",
        "site_master_sha256": sha256_file(site_master_path),
        "pair_mapping_sha256": sha256_file(pair_output),
        "alu_bed_sha256": sha256_file(bed_path),
        "reference_fasta_sha256": sha256_file(fasta_path),
        "raw_table_sha256": {
            context: sha256_file(raw_dir / filename)
            for context, filename in CONTEXT_TO_RAW.items()
        },
    }
    return sites, pair_mapping, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument(
        "--raw-editing-dir", type=Path, default=Path("data/external/gtex_editing")
    )
    parser.add_argument(
        "--alu-bed", type=Path, default=Path("data/raw/alu_pairs/Pair_Alu_withStrand.bed")
    )
    parser.add_argument(
        "--reference-fasta", type=Path, default=Path("data/external/reference/GRCh38.primary_assembly.genome.fa")
    )
    parser.add_argument(
        "--pair-output", type=Path, default=Path("data/derived/genomic_mapping/pair_mapping.csv")
    )
    args = parser.parse_args()
    repository = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repository / path

    site_master = resolve(args.site_master)
    _, _, summary = reconstruct(
        site_master,
        resolve(args.raw_editing_dir),
        resolve(args.alu_bed),
        resolve(args.reference_fasta),
        resolve(args.pair_output),
    )
    summary_path = site_master.with_name("genomic_mapping.summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    site_master.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
