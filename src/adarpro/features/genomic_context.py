"""Build outcome-independent IRAlu architecture and transcript-state features."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.quarantine import sha256_file
from adarpro.rbp.eclip_features import dot_bracket_partners

NEIGHBOR_WINDOWS = (1_000, 5_000, 10_000, 50_000)
RMSK_COLUMNS = [
    "score",
    "milli_div",
    "milli_del",
    "milli_ins",
    "chromosome",
    "start",
    "end",
    "strand",
    "repeat_name",
    "repeat_class",
    "repeat_family",
    "rep_start",
    "rep_end",
    "rep_left",
    "repeat_id",
]
RMSK_USECOLS = list(range(1, 8)) + list(range(9, 17))


def parse_attributes(value: str) -> dict[str, str]:
    attributes = {}
    for field in value.rstrip(";").split("; "):
        if " " not in field:
            continue
        key, item = field.split(" ", 1)
        attributes[key] = item.strip('"')
    return attributes


def overlaps_any(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start < right and end > left for left, right in intervals)


def read_gtf_transcripts(
    gtf_path: Path, pairs: pd.DataFrame
) -> tuple[dict[str, dict], dict[str, dict[str, list[tuple[int, int]]]]]:
    pair_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in pairs.itertuples(index=False):
        pair_intervals[row.chr1].append((int(row.start1), int(row.end2)))
    transcripts: dict[str, dict] = {}
    with gzip.open(gtf_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "transcript":
                continue
            chromosome = fields[0]
            if chromosome not in pair_intervals:
                continue
            start, end = int(fields[3]) - 1, int(fields[4])
            if not overlaps_any(start, end, pair_intervals[chromosome]):
                continue
            attributes = parse_attributes(fields[8])
            transcript_id = attributes.get("transcript_id")
            if transcript_id is None:
                continue
            transcripts[transcript_id] = {
                "transcript_id": transcript_id,
                "gene_id": attributes.get("gene_id", "unknown").split(".")[0],
                "gene_name": attributes.get("gene_name", "unknown"),
                "gene_type": attributes.get("gene_type", "unknown"),
                "transcript_type": attributes.get("transcript_type", "unknown"),
                "chromosome": chromosome,
                "start": start,
                "end": end,
                "strand": fields[6],
            }
    features: dict[str, dict[str, list[tuple[int, int]]]] = {
        transcript_id: defaultdict(list) for transcript_id in transcripts
    }
    with gzip.open(gtf_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            feature = fields[2] if len(fields) == 9 else ""
            if feature not in {"exon", "CDS", "UTR", "five_prime_utr", "three_prime_utr"}:
                continue
            attributes = parse_attributes(fields[8])
            transcript_id = attributes.get("transcript_id")
            if transcript_id not in transcripts:
                continue
            features[transcript_id][feature].append(
                (int(fields[3]) - 1, int(fields[4]))
            )
    for transcript_id, by_feature in features.items():
        for feature in by_feature:
            by_feature[feature].sort()
        exons = by_feature.get("exon", [])
        by_feature["intron"] = [
            (left[1], right[0])
            for left, right in pairwise(exons)
            if right[0] > left[1]
        ]
    return transcripts, features


def contains(intervals: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(left <= start and right >= end for left, right in intervals)


def point_inside(intervals: list[tuple[int, int]], position: int) -> bool:
    return any(left <= position < right for left, right in intervals)


def min_boundary_distance(intervals: list[tuple[int, int]], position: int) -> float:
    if not intervals:
        return np.nan
    return float(min(abs(position - boundary) for interval in intervals for boundary in interval))


def load_alu_repeats(path: Path) -> pd.DataFrame:
    table = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=RMSK_USECOLS,
        names=RMSK_COLUMNS,
        compression="gzip",
        dtype={
            "chromosome": "category",
            "strand": "category",
            "repeat_name": "category",
            "repeat_class": "category",
            "repeat_family": "category",
        },
    )
    return table[
        (table["repeat_class"] == "SINE") & (table["repeat_family"] == "Alu")
    ].copy()


def best_repeat(
    repeats: pd.DataFrame, start: int, end: int
) -> tuple[pd.Series | None, int | None, float]:
    candidates = repeats[(repeats["start"] < end) & (repeats["end"] > start)].copy()
    if candidates.empty:
        return None, None, 0.0
    overlap = np.minimum(candidates["end"], end) - np.maximum(candidates["start"], start)
    index = overlap.idxmax()
    coverage = float(overlap.loc[index] / max(1, end - start))
    return candidates.loc[index], int(index), coverage


def interval_distance(start: np.ndarray, end: np.ndarray, left: int, right: int) -> np.ndarray:
    return np.maximum.reduce([start - right, left - end, np.zeros(len(start), dtype=int)])


def alu_age_flags(name: str) -> dict[str, float]:
    return {
        "alu_age_young_y": float(name.startswith("AluY")),
        "alu_age_middle_s": float(name.startswith("AluS")),
        "alu_age_old_j": float(name.startswith("AluJ")),
        "alu_age_unclassified": float(not name.startswith(("AluY", "AluS", "AluJ"))),
    }


def pair_repeat_features(pair: pd.Series, repeats: pd.DataFrame) -> dict[str, float]:
    start1, end1 = int(pair.start1), int(pair.end1)
    start2, end2 = int(pair.start2), int(pair.end2)
    first, first_index, first_coverage = best_repeat(repeats, start1, end1)
    second, second_index, second_coverage = best_repeat(repeats, start2, end2)
    result: dict[str, float] = {
        "repeat__arm1_match_coverage": first_coverage,
        "repeat__arm2_match_coverage": second_coverage,
        "repeat__both_arms_matched": float(first is not None and second is not None),
    }
    matched_indices = {value for value in (first_index, second_index) if value is not None}
    span_start, span_end = start1, end2
    for window in NEIGHBOR_WINDOWS:
        local = repeats[
            (repeats["start"] < span_end + window)
            & (repeats["end"] > span_start - window)
            & ~repeats.index.isin(matched_indices)
        ]
        result[f"repeat__neighbor_alu_count_w{window}"] = float(len(local))
        result[f"repeat__neighbor_alu_density_per_kb_w{window}"] = float(
            len(local) / ((span_end - span_start + 2 * window) / 1000)
        )
    for arm, repeat, own_index, start, end in (
        ("arm1", first, first_index, start1, end1),
        ("arm2", second, second_index, start2, end2),
    ):
        if repeat is None:
            for name in (
                "milli_div",
                "milli_del",
                "milli_ins",
                "score_per_nt",
                "repeat_length",
                "nearest_opposite_distance_log1p",
            ):
                result[f"repeat__{arm}_{name}"] = np.nan
            for name, value in alu_age_flags("").items():
                result[f"repeat__{arm}_{name}"] = value
            continue
        repeat_name = str(repeat["repeat_name"])
        result.update(
            {
                f"repeat__{arm}_milli_div": float(repeat["milli_div"]),
                f"repeat__{arm}_milli_del": float(repeat["milli_del"]),
                f"repeat__{arm}_milli_ins": float(repeat["milli_ins"]),
                f"repeat__{arm}_score_per_nt": float(repeat["score"] / max(1, repeat["end"] - repeat["start"])),
                f"repeat__{arm}_repeat_length": float(repeat["end"] - repeat["start"]),
            }
        )
        for name, value in alu_age_flags(repeat_name).items():
            result[f"repeat__{arm}_{name}"] = value
        alternatives = repeats[
            (repeats["strand"] != repeat["strand"])
            & (repeats.index != own_index)
        ]
        if alternatives.empty:
            nearest = np.nan
        else:
            distances = interval_distance(
                alternatives["start"].to_numpy(),
                alternatives["end"].to_numpy(),
                start,
                end,
            )
            nearest = float(np.min(distances))
        result[f"repeat__{arm}_nearest_opposite_distance_log1p"] = np.log1p(nearest)
    if first is not None and second is not None:
        result["repeat__arms_inverted"] = float(first["strand"] != second["strand"])
        result["repeat__same_subfamily"] = float(first["repeat_name"] == second["repeat_name"])
        result["repeat__divergence_mean"] = float(
            (first["milli_div"] + second["milli_div"]) / 2
        )
        result["repeat__divergence_abs_difference"] = float(
            abs(first["milli_div"] - second["milli_div"])
        )
    else:
        result.update(
            {
                "repeat__arms_inverted": np.nan,
                "repeat__same_subfamily": np.nan,
                "repeat__divergence_mean": np.nan,
                "repeat__divergence_abs_difference": np.nan,
            }
        )
    return result


def sequence_fraction(sequence: str, bases: set[str]) -> float:
    observed = [base for base in sequence if base in {"A", "C", "G", "T"}]
    return sum(base in bases for base in observed) / len(observed) if observed else np.nan


def pair_intrinsic_features(pair: pd.Series) -> dict[str, float]:
    structure = pair.structure
    partners = dot_bracket_partners(structure)
    spans = [partner - index for index, partner in enumerate(partners) if partner is not None and partner > index]
    first_length = int(pair.end1 - pair.start1 + 1)
    second_length = int(pair.end2 - pair.start2 + 1)
    if pair.strand == "+":
        arm1 = pair.sequence[:first_length]
        arm2 = pair.sequence[first_length + 10 : first_length + 10 + second_length]
    else:
        arm2 = pair.sequence[:second_length]
        arm1 = pair.sequence[second_length + 10 : second_length + 10 + first_length]
    gap = int(pair.start2 - pair.end1)
    return {
        "geometry__arm1_length": float(first_length),
        "geometry__arm2_length": float(second_length),
        "geometry__arm_length_abs_difference": float(abs(first_length - second_length)),
        "geometry__genomic_gap": float(gap),
        "geometry__genomic_gap_log1p": float(np.log1p(max(gap, 0))),
        "geometry__genomic_span": float(pair.end2 - pair.start1),
        "geometry__arm1_gc": sequence_fraction(arm1, {"G", "C"}),
        "geometry__arm2_gc": sequence_fraction(arm2, {"G", "C"}),
        "geometry__arm_gc_abs_difference": abs(
            sequence_fraction(arm1, {"G", "C"}) - sequence_fraction(arm2, {"G", "C"})
        ),
        "intrinsic__sequence_length": float(len(pair.sequence)),
        "intrinsic__whole_gc": sequence_fraction(pair.sequence, {"G", "C"}),
        "intrinsic__paired_fraction": float(sum(value is not None for value in partners) / len(partners)),
        "intrinsic__base_pair_count": float(len(spans)),
        "intrinsic__mean_base_pair_span": float(np.mean(spans)) if spans else 0.0,
        "intrinsic__max_base_pair_span": float(max(spans, default=0)),
        "intrinsic__stem_opening_run_count": float(
            sum(symbol == "(" and (index == 0 or structure[index - 1] != "(") for index, symbol in enumerate(structure))
        ),
    }


def transcript_pair_features(
    pair: pd.Series,
    transcripts: dict[str, dict],
    transcript_features: dict[str, dict[str, list[tuple[int, int]]]],
    by_chromosome: dict[str, list[str]],
) -> tuple[dict[str, float], list[str], list[str]]:
    pair_start, pair_end = int(pair.start1), int(pair.end2)
    candidates = [
        transcript_id
        for transcript_id in by_chromosome.get(pair.chr1, [])
        if transcripts[transcript_id]["start"] < pair_end
        and transcripts[transcript_id]["end"] > pair_start
    ]
    same_strand = [
        transcript_id
        for transcript_id in candidates
        if transcripts[transcript_id]["strand"] == pair.strand
    ]
    compatible = [
        transcript_id
        for transcript_id in same_strand
        if transcripts[transcript_id]["start"] <= pair_start
        and transcripts[transcript_id]["end"] >= pair_end
    ]
    same_exon, same_intron, protein_coding = [], [], []
    for transcript_id in compatible:
        feature = transcript_features[transcript_id]
        same_exon.append(
            contains(feature.get("exon", []), int(pair.start1), int(pair.end1))
            and contains(feature.get("exon", []), int(pair.start2), int(pair.end2))
            and any(
                left <= int(pair.start1) and right >= int(pair.end2)
                for left, right in feature.get("exon", [])
            )
        )
        same_intron.append(
            any(
                left <= int(pair.start1) and right >= int(pair.end2)
                for left, right in feature.get("intron", [])
            )
        )
        protein_coding.append(
            transcripts[transcript_id]["transcript_type"] == "protein_coding"
        )
    genes = {transcripts[transcript_id]["gene_id"] for transcript_id in compatible}
    return (
        {
            "transcript__overlapping_transcript_count_log1p": float(np.log1p(len(candidates))),
            "transcript__same_strand_transcript_count_log1p": float(np.log1p(len(same_strand))),
            "transcript__compatible_transcript_count_log1p": float(np.log1p(len(compatible))),
            "transcript__compatible_gene_count_log1p": float(np.log1p(len(genes))),
            "transcript__antisense_overlap_any": float(len(candidates) > len(same_strand)),
            "transcript__same_exon_any": float(any(same_exon)),
            "transcript__same_intron_any": float(any(same_intron)),
            "transcript__same_exon_fraction": float(np.mean(same_exon)) if same_exon else 0.0,
            "transcript__same_intron_fraction": float(np.mean(same_intron)) if same_intron else 0.0,
            "transcript__protein_coding_fraction": float(np.mean(protein_coding)) if protein_coding else 0.0,
            "transcript__no_compatible_transcript": float(not compatible),
        },
        compatible,
        same_strand,
    )


def site_transcript_features(
    row: pd.Series,
    transcript_ids: list[str],
    fallback_ids: list[str],
    transcripts: dict[str, dict],
    features: dict[str, dict[str, list[tuple[int, int]]]],
) -> dict[str, float]:
    position = int(row.genomic_position_0based)
    selected = transcript_ids or [
        transcript_id
        for transcript_id in fallback_ids
        if transcripts[transcript_id]["start"] <= position < transcripts[transcript_id]["end"]
    ]
    if not selected:
        return {
            "site__genic": 0.0,
            "site__exonic_fraction": 0.0,
            "site__intronic_fraction": 0.0,
            "site__cds_fraction": 0.0,
            "site__utr_fraction": 0.0,
            "site__protein_coding_fraction": 0.0,
            "site__splice_distance_log1p": np.nan,
            "site__tss_distance_log1p": np.nan,
            "site__tes_distance_log1p": np.nan,
        }
    exonic, cds, utr, protein, splice, tss, tes = [], [], [], [], [], [], []
    for transcript_id in selected:
        metadata = transcripts[transcript_id]
        feature = features[transcript_id]
        exonic.append(point_inside(feature.get("exon", []), position))
        cds.append(point_inside(feature.get("CDS", []), position))
        utr_intervals = (
            feature.get("UTR", [])
            + feature.get("five_prime_utr", [])
            + feature.get("three_prime_utr", [])
        )
        utr.append(point_inside(utr_intervals, position))
        protein.append(metadata["transcript_type"] == "protein_coding")
        splice.append(min_boundary_distance(feature.get("exon", []), position))
        if metadata["strand"] == "+":
            tss.append(abs(position - metadata["start"]))
            tes.append(abs(position - metadata["end"]))
        else:
            tss.append(abs(position - metadata["end"]))
            tes.append(abs(position - metadata["start"]))
    finite_splice = [value for value in splice if np.isfinite(value)]
    return {
        "site__genic": 1.0,
        "site__exonic_fraction": float(np.mean(exonic)),
        "site__intronic_fraction": float(1 - np.mean(exonic)),
        "site__cds_fraction": float(np.mean(cds)),
        "site__utr_fraction": float(np.mean(utr)),
        "site__protein_coding_fraction": float(np.mean(protein)),
        "site__splice_distance_log1p": float(np.log1p(min(finite_splice))) if finite_splice else np.nan,
        "site__tss_distance_log1p": float(np.log1p(min(tss))),
        "site__tes_distance_log1p": float(np.log1p(min(tes))),
    }


def site_intrinsic_features(row: pd.Series) -> dict[str, float]:
    position = int(row.local_position)
    partners = dot_bracket_partners(row.structure)
    partner = partners[position]
    local21 = row.sequence[max(0, position - 10) : position + 11]
    local51 = row.sequence[max(0, position - 25) : position + 26]
    return {
        "intrinsic_site__focal_paired": float(partner is not None),
        "intrinsic_site__pair_span": float(abs(partner - position)) if partner is not None else 0.0,
        "intrinsic_site__local_gc_21": sequence_fraction(local21, {"G", "C"}),
        "intrinsic_site__local_gc_51": sequence_fraction(local51, {"G", "C"}),
        "intrinsic_site__distance_to_end_log1p": float(
            np.log1p(min(position, len(row.sequence) - position - 1))
        ),
        "intrinsic_site__relative_position": float(position / max(1, len(row.sequence) - 1)),
    }


def run(
    site_master_path: Path,
    repeatmasker_path: Path,
    gtf_path: Path,
    output_path: Path,
) -> dict:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("genomic context input contains a sealed split")
    required = {
        "chr1",
        "start1",
        "end1",
        "chr2",
        "start2",
        "end2",
        "strand",
        "genomic_position_0based",
    }
    if required - set(master.columns):
        raise RuntimeError(f"site master lacks coordinate fields: {required - set(master.columns)}")
    pair_columns = [
        "pair_id",
        "chr1",
        "start1",
        "end1",
        "chr2",
        "start2",
        "end2",
        "strand",
        "sequence",
        "structure",
    ]
    pairs = master[pair_columns].drop_duplicates("pair_id").sort_values("pair_id")
    if (pairs["chr1"] != pairs["chr2"]).any():
        raise RuntimeError("cross-chromosome Alu pair encountered")

    repeats = load_alu_repeats(repeatmasker_path)
    repeat_by_chromosome = {
        chromosome: group.sort_values("start")
        for chromosome, group in repeats.groupby("chromosome", observed=True)
    }
    transcripts, transcript_features = read_gtf_transcripts(gtf_path, pairs)
    transcripts_by_chromosome: dict[str, list[str]] = defaultdict(list)
    for transcript_id, metadata in transcripts.items():
        transcripts_by_chromosome[metadata["chromosome"]].append(transcript_id)

    pair_rows, compatible_by_pair, fallback_by_pair = [], {}, {}
    for pair in pairs.itertuples(index=False):
        pair_series = pd.Series(pair._asdict())
        transcript, compatible, fallback = transcript_pair_features(
            pair_series,
            transcripts,
            transcript_features,
            transcripts_by_chromosome,
        )
        pair_rows.append(
            {
                "pair_id": int(pair.pair_id),
                **pair_intrinsic_features(pair_series),
                **pair_repeat_features(
                    pair_series, repeat_by_chromosome.get(pair.chr1, repeats.iloc[:0])
                ),
                **transcript,
            }
        )
        compatible_by_pair[int(pair.pair_id)] = compatible
        fallback_by_pair[int(pair.pair_id)] = fallback
    pair_table = pd.DataFrame(pair_rows)

    site_rows = []
    for row in master.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        pair_id = int(row.pair_id)
        site_rows.append(
            {
                "example_id": row.example_id,
                **site_intrinsic_features(row_series),
                **site_transcript_features(
                    row_series,
                    compatible_by_pair[pair_id],
                    fallback_by_pair[pair_id],
                    transcripts,
                    transcript_features,
                ),
            }
        )
    output = master[["example_id", "context", "pair_id", "pair_local_key"]].merge(
        pair_table, on="pair_id", validate="many_to_one"
    ).merge(pd.DataFrame(site_rows), on="example_id", validate="one_to_one")
    feature_columns = [
        column
        for column in output
        if column not in {"example_id", "context", "pair_id", "pair_local_key"}
    ]
    if output[feature_columns].select_dtypes(exclude=[np.number]).shape[1]:
        raise RuntimeError("genomic context emitted non-numeric model features")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    summary = {
        "scope": "outcome-independent features on canonical train+valid examples only",
        "rows": len(output),
        "pairs": int(output["pair_id"].nunique()),
        "features": {
            prefix: len([column for column in feature_columns if column.startswith(prefix)])
            for prefix in ("geometry__", "repeat__", "transcript__", "site__", "intrinsic__", "intrinsic_site__")
        },
        "repeatmasker_alu_records": len(repeats),
        "gtf_candidate_transcripts": len(transcripts),
        "prespecified_neighbor_windows_nt": list(NEIGHBOR_WINDOWS),
        "input_hashes": {
            "site_master": sha256_file(site_master_path),
            "repeatmasker": sha256_file(repeatmasker_path),
            "gencode_gtf": sha256_file(gtf_path),
        },
        "output_sha256": sha256_file(output_path),
        "canonical_test_access": False,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument("--repeatmasker", type=Path, required=True)
    parser.add_argument("--gencode-gtf", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/features/genomic_context.parquet"),
    )
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(
        json.dumps(
            run(
                resolve(args.site_master),
                resolve(args.repeatmasker),
                resolve(args.gencode_gtf),
                resolve(args.output),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
