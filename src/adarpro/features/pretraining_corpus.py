"""Audit a leakage-excluded genome-wide inverted-Alu pretraining pool."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.features.genomic_context import load_alu_repeats
from adarpro.quarantine import sha256_file
from adarpro.registry import register

WINDOWS = (1_000, 2_000, 5_000, 10_000)
PAIR_COLUMNS = ("chr1", "start1", "end1", "chr2", "start2", "end2", "strand")


def benchmark_overlap_mask(repeats: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """Conservatively exclude every repeat overlapping either benchmark arm."""

    chromosome = repeats["chromosome"].astype(str).to_numpy()
    excluded = np.zeros(len(repeats), dtype=bool)
    for name, group in pairs.groupby("chr1", observed=True):
        indices = np.flatnonzero(chromosome == name)
        if not len(indices):
            continue
        starts = repeats.iloc[indices]["start"].to_numpy()
        ends = repeats.iloc[indices]["end"].to_numpy()
        local = np.zeros(len(indices), dtype=bool)
        for row in group.itertuples(index=False):
            local |= ((starts < row.end1) & (ends > row.start1)) | (
                (starts < row.end2) & (ends > row.start2)
            )
        excluded[indices[local]] = True
    return excluded


def count_inverted_pairs(
    repeats: pd.DataFrame, windows: tuple[int, ...] = WINDOWS
) -> tuple[dict[int, int], dict[str, int]]:
    """Count nonoverlapping opposite-strand pairs at fixed genomic gaps."""

    ordered_windows = tuple(sorted(windows))
    counts = {window: 0 for window in ordered_windows}
    chromosome_5kb = {}
    for chromosome, group in repeats.groupby("chromosome", observed=True):
        group = group.sort_values("start")
        starts = group["start"].to_numpy()
        ends = group["end"].to_numpy()
        strands = group["strand"].astype(str).to_numpy()
        subtotal = {window: 0 for window in ordered_windows}
        for index in range(len(group) - 1):
            lower = max(
                index + 1,
                int(np.searchsorted(starts, ends[index], side="left")),
            )
            upper = int(
                np.searchsorted(
                    starts, ends[index] + ordered_windows[-1], side="right"
                )
            )
            if upper <= lower:
                continue
            gaps = starts[lower:upper] - ends[index]
            opposite = strands[lower:upper] != strands[index]
            for window in ordered_windows:
                subtotal[window] += int(np.sum(opposite & (gaps <= window)))
        for window in ordered_windows:
            counts[window] += subtotal[window]
        if 5_000 in subtotal:
            chromosome_5kb[str(chromosome)] = subtotal[5_000]
    return counts, chromosome_5kb


def inverted_pair_spans(
    repeats: pd.DataFrame, maximum_gap: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return candidate genomic spans without materializing sequence content."""

    result = {}
    for chromosome, group in repeats.groupby("chromosome", observed=True):
        group = group.sort_values("start")
        starts = group["start"].to_numpy()
        ends = group["end"].to_numpy()
        strands = group["strand"].astype(str).to_numpy()
        left_chunks, right_chunks = [], []
        for index in range(len(group) - 1):
            lower = max(
                index + 1,
                int(np.searchsorted(starts, ends[index], side="left")),
            )
            upper = int(
                np.searchsorted(starts, ends[index] + maximum_gap, side="right")
            )
            if upper <= lower:
                continue
            candidate_indices = np.arange(lower, upper)
            candidate_indices = candidate_indices[
                strands[candidate_indices] != strands[index]
            ]
            if not len(candidate_indices):
                continue
            left_chunks.append(np.full(len(candidate_indices), starts[index]))
            right_chunks.append(ends[candidate_indices])
        if left_chunks:
            result[str(chromosome)] = (
                np.concatenate(left_chunks),
                np.concatenate(right_chunks),
            )
    return result


def read_transcript_intervals(gtf_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load GRCh38 transcript spans needed for a conservative expression proxy."""

    intervals: dict[str, list[tuple[int, int]]] = {}
    with gzip.open(gtf_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "transcript":
                continue
            intervals.setdefault(fields[0], []).append(
                (int(fields[3]) - 1, int(fields[4]))
            )
    result = {}
    for chromosome, values in intervals.items():
        ordered = np.asarray(sorted(values), dtype=np.int64)
        result[chromosome] = (ordered[:, 0], np.maximum.accumulate(ordered[:, 1]))
    return result


def count_transcript_compatible_pairs(
    pair_spans: dict[str, tuple[np.ndarray, np.ndarray]],
    transcripts: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[int, dict[str, int]]:
    """Count pair spans fully contained by at least one annotated transcript."""

    total = 0
    by_chromosome = {}
    for chromosome, (left, right) in pair_spans.items():
        if chromosome not in transcripts:
            continue
        transcript_start, prefix_max_end = transcripts[chromosome]
        positions = np.searchsorted(transcript_start, left, side="right") - 1
        valid = positions >= 0
        covered = np.zeros(len(left), dtype=bool)
        covered[valid] = prefix_max_end[positions[valid]] >= right[valid]
        count = int(covered.sum())
        total += count
        by_chromosome[chromosome] = count
    return total, by_chromosome


def run(
    repository: Path,
    repeats_path: Path,
    pair_path: Path,
    output: Path,
    gtf_path: Path | None = None,
) -> dict:
    repeats = load_alu_repeats(repeats_path)
    pairs = pd.read_csv(pair_path, sep="\t", header=None, names=PAIR_COLUMNS)
    excluded = benchmark_overlap_mask(repeats, pairs)
    retained = repeats.loc[~excluded].copy()
    counts, chromosome_5kb = count_inverted_pairs(retained)
    summary = {
        "scope": (
            "outcome-independent RepeatMasker upper-bound audit; all repeats "
            "overlapping any canonical benchmark arm excluded"
        ),
        "genome_build": "GRCh38",
        "repeatmasker_alus": len(repeats),
        "benchmark_pairs_listed": len(pairs),
        "benchmark_overlapping_alus_excluded": int(excluded.sum()),
        "remaining_alus": len(retained),
        "opposite_orientation_nonoverlapping_pair_counts": {
            str(window): int(counts[window]) for window in WINDOWS
        },
        "top_chromosomes_5kb": sorted(
            chromosome_5kb.items(), key=lambda item: item[1], reverse=True
        )[:10],
        "interpretation": (
            "This is a sequence-availability upper bound, not a transcribed or "
            "duplex-forming corpus. Transcript compatibility, redundancy control, "
            "foldability and chromosome-held-out contamination checks remain required."
        ),
        "canonical_test_labels_accessed": False,
    }
    input_hashes = {
        "repeatmasker": sha256_file(repeats_path),
        "canonical_pair_coordinates": sha256_file(pair_path),
    }
    if gtf_path is not None:
        spans = inverted_pair_spans(retained, 1_000)
        transcripts = read_transcript_intervals(gtf_path)
        compatible, compatible_by_chromosome = count_transcript_compatible_pairs(
            spans, transcripts
        )
        summary["within_1kb_transcript_compatible_pairs"] = compatible
        summary["within_1kb_transcript_compatible_fraction"] = compatible / counts[1_000]
        summary["top_chromosomes_transcript_compatible_1kb"] = sorted(
            compatible_by_chromosome.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        summary["interpretation"] = (
            "Transcript-span containment leaves a large candidate pool but is "
            "only a permissive transcription proxy. Isoform connectivity, "
            "expression, redundancy control, foldability and chromosome-aware "
            "contamination checks remain required."
        )
        input_hashes["gencode_gtf"] = sha256_file(gtf_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    register(
        repository,
        {
            "run_id": (
                "iralus_pretraining_pool_transcript_screen_v2"
                if gtf_path is not None
                else "iralus_pretraining_pool_feasibility_v1"
            ),
            "experiment_type": "outcome_independent_pretraining_pool_audit",
            "context": "genome-wide GRCh38",
            "input_hashes": input_hashes,
            "fold_definition": "not predictive evaluation; all benchmark arms excluded",
            "seed": "none",
            "features": "RepeatMasker Alu orientation and inter-repeat genomic gap",
            "hyperparameters": {"windows": list(WINDOWS)},
            "artifact": str(output.relative_to(repository)),
            "metrics": summary,
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--repeatmasker", type=Path, required=True)
    parser.add_argument("--gencode-gtf", type=Path)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/raw/alu_pairs/Pair_Alu_withStrand.bed"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/pretraining/iralus_feasibility.json"),
    )
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(
        json.dumps(
            run(
                root,
                resolve(args.repeatmasker),
                resolve(args.pairs),
                resolve(args.output),
                resolve(args.gencode_gtf) if args.gencode_gtf is not None else None,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
