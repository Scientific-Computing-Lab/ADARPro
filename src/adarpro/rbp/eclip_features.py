"""Construct strand-aware, multi-scale ENCODE eCLIP features on development sites."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 25, 50, 100)
GENOMIC_SHIFT_CONTROL_NT = 1_000_000


@dataclass(frozen=True)
class PeakIndex:
    starts: np.ndarray
    ends: np.ndarray
    signals: np.ndarray
    prefix_max_end: np.ndarray


def read_narrowpeak(path: Path) -> pd.DataFrame:
    columns = [
        "chromosome",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signal",
        "p_value",
        "q_value",
        "summit",
    ]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        peaks = pd.read_csv(handle, sep="\t", header=None, names=columns)
    peaks = peaks[peaks["strand"].isin(["+", "-"])].copy()
    peaks[["start", "end"]] = peaks[["start", "end"]].astype(int)
    peaks["signal"] = pd.to_numeric(peaks["signal"], errors="coerce").fillna(0.0)
    if (peaks["end"] <= peaks["start"]).any():
        raise RuntimeError(f"non-positive peak interval in {path}")
    return peaks


def build_indices(peaks: pd.DataFrame) -> dict[tuple[str, str], PeakIndex]:
    output = {}
    for key, group in peaks.groupby(["chromosome", "strand"], sort=False):
        group = group.sort_values(["start", "end"])
        ends = group["end"].to_numpy(dtype=np.int64)
        output[key] = PeakIndex(
            starts=group["start"].to_numpy(dtype=np.int64),
            ends=ends,
            signals=group["signal"].to_numpy(dtype=float),
            prefix_max_end=np.maximum.accumulate(ends),
        )
    return output


def interval_features(index: PeakIndex | None, position: int) -> dict[str, float | int]:
    """Summarize peaks around a zero-based nucleotide coordinate."""
    result: dict[str, float | int] = {
        "overlap": 0,
        "nearest_distance": -1,
    }
    for window in WINDOWS:
        result[f"count_w{window}"] = 0
        result[f"max_signal_w{window}"] = 0.0
        result[f"mean_signal_w{window}"] = 0.0
    if index is None or len(index.starts) == 0:
        return result

    right = int(np.searchsorted(index.starts, position, side="right"))
    overlap_left = right > 0 and index.prefix_max_end[right - 1] > position
    next_distance = (
        int(index.starts[right] - position) if right < len(index.starts) else np.inf
    )
    previous_distance = (
        int(position - index.prefix_max_end[right - 1] + 1)
        if right > 0 and not overlap_left
        else np.inf
    )
    result["overlap"] = int(overlap_left)
    result["nearest_distance"] = int(
        0 if overlap_left else min(next_distance, previous_distance)
    )

    for window in WINDOWS:
        high = int(np.searchsorted(index.starts, position + window + 1, side="left"))
        if high == 0:
            continue
        candidates = np.flatnonzero(index.ends[:high] > position - window)
        if len(candidates) == 0:
            continue
        signals = index.signals[candidates]
        result[f"count_w{window}"] = len(candidates)
        result[f"max_signal_w{window}"] = float(np.max(signals))
        result[f"mean_signal_w{window}"] = float(np.mean(signals))
    return result


def dot_bracket_partners(structure: str) -> list[int | None]:
    stack: list[int] = []
    partners: list[int | None] = [None] * len(structure)
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise RuntimeError("unbalanced dot-bracket structure")
            left = stack.pop()
            partners[left] = index
            partners[index] = left
        elif symbol != ".":
            raise RuntimeError(f"unsupported dot-bracket symbol: {symbol}")
    if stack:
        raise RuntimeError("unbalanced dot-bracket structure")
    return partners


def local_to_genomic(pair: pd.Series, local_position: int) -> tuple[int, int] | None:
    """Map transcript-oriented local index to genomic coordinate and Alu arm."""
    first_length = int(pair["end1"] - pair["start1"] + 1)
    second_length = int(pair["end2"] - pair["start2"] + 1)
    if pair["strand"] == "+":
        if local_position < first_length:
            return int(pair["start1"] + local_position), 1
        offset = local_position - first_length - 10
        if 0 <= offset < second_length:
            return int(pair["start2"] + offset), 2
        return None
    if local_position < second_length:
        return int(pair["end2"] - local_position), 2
    offset = local_position - second_length - 10
    if 0 <= offset < first_length:
        return int(pair["end1"] - offset), 1
    return None


def union_coverage(starts: np.ndarray, ends: np.ndarray) -> int:
    if len(starts) == 0:
        return 0
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    total = 0
    current_start, current_end = int(starts[0]), int(ends[0])
    for start, end in zip(starts[1:], ends[1:]):
        start, end = int(start), int(end)
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + current_end - current_start


def region_features(
    index: PeakIndex | None, regions: list[tuple[int, int]]
) -> dict[str, float | int]:
    length = sum(end - start for start, end in regions)
    clipped_starts, clipped_ends, signals = [], [], []
    if index is not None:
        used: set[int] = set()
        for region_start, region_end in regions:
            high = int(np.searchsorted(index.starts, region_end, side="left"))
            for peak_index in np.flatnonzero(index.ends[:high] > region_start):
                clipped_starts.append(max(int(index.starts[peak_index]), region_start))
                clipped_ends.append(min(int(index.ends[peak_index]), region_end))
                if int(peak_index) not in used:
                    signals.append(float(index.signals[peak_index]))
                    used.add(int(peak_index))
    covered = union_coverage(
        np.asarray(clipped_starts, dtype=int), np.asarray(clipped_ends, dtype=int)
    )
    return {
        "peak_count": len(signals),
        "covered_nt": covered,
        "coverage_fraction": covered / length if length else 0.0,
        "max_signal": max(signals, default=0.0),
    }


def prepare_sites(site_master: pd.DataFrame) -> pd.DataFrame:
    if set(site_master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("eCLIP feature input contains a sealed split")
    columns = [
        "pair_local_key",
        "pair_id",
        "local_position",
        "site_id",
        "chromosome",
        "genomic_position_0based",
        "strand",
        "structure",
        "start1",
        "end1",
        "start2",
        "end2",
    ]
    sites = site_master[columns].drop_duplicates()
    if sites["pair_local_key"].duplicated().any():
        raise RuntimeError("pair/local site identity is not unique")
    partner_by_pair = {
        pair_id: dot_bracket_partners(group["structure"].iloc[0])
        for pair_id, group in sites.groupby("pair_id")
    }
    partner_positions, partner_genomic, focal_arms, partner_arms = [], [], [], []
    for row in sites.itertuples(index=False):
        local = int(row.local_position)
        partners = partner_by_pair[row.pair_id]
        partner = partners[local]
        pair = pd.Series(row._asdict())
        focal_mapping = local_to_genomic(pair, local)
        partner_mapping = local_to_genomic(pair, partner) if partner is not None else None
        if focal_mapping is None or focal_mapping[0] != row.genomic_position_0based:
            raise RuntimeError(f"local/genomic mapping disagreement for {row.pair_local_key}")
        partner_positions.append(partner if partner is not None else -1)
        partner_genomic.append(partner_mapping[0] if partner_mapping else -1)
        focal_arms.append(focal_mapping[1])
        partner_arms.append(partner_mapping[1] if partner_mapping else -1)
    return sites.assign(
        partner_local_position=partner_positions,
        partner_genomic_position_0based=partner_genomic,
        focal_arm=focal_arms,
        partner_arm=partner_arms,
        focal_is_paired=np.asarray(partner_positions) >= 0,
    )


def prefix_features(prefix: str, values: dict[str, float | int]) -> dict[str, object]:
    return {f"{prefix}_{name}": value for name, value in values.items()}


def annotate_track(sites: pd.DataFrame, manifest_row: pd.Series) -> pd.DataFrame:
    peaks = read_narrowpeak(Path(manifest_row["local_path"]))
    indices = build_indices(peaks)
    pair_rows = sites.drop_duplicates("pair_id").set_index("pair_id")
    pair_cache = {}
    for pair_id, pair in pair_rows.iterrows():
        index = indices.get((pair["chromosome"], pair["strand"]))
        arm1 = (int(pair["start1"]), int(pair["end1"]) + 1)
        arm2 = (int(pair["start2"]), int(pair["end2"]) + 1)
        shifted_arm1 = tuple(value + GENOMIC_SHIFT_CONTROL_NT for value in arm1)
        shifted_arm2 = tuple(value + GENOMIC_SHIFT_CONTROL_NT for value in arm2)
        pair_cache[pair_id] = {
            **prefix_features("duplex", region_features(index, [arm1, arm2])),
            **prefix_features("arm1", region_features(index, [arm1])),
            **prefix_features("arm2", region_features(index, [arm2])),
            **prefix_features(
                "shifted_duplex",
                region_features(index, [shifted_arm1, shifted_arm2]),
            ),
        }

    rows = []
    for site in sites.itertuples(index=False):
        index = indices.get((site.chromosome, site.strand))
        focal = interval_features(index, int(site.genomic_position_0based))
        shifted_focal = interval_features(
            index, int(site.genomic_position_0based) + GENOMIC_SHIFT_CONTROL_NT
        )
        partner = (
            interval_features(index, int(site.partner_genomic_position_0based))
            if site.focal_is_paired
            else interval_features(None, -1)
        )
        shifted_partner = (
            interval_features(
                index,
                int(site.partner_genomic_position_0based) + GENOMIC_SHIFT_CONTROL_NT,
            )
            if site.focal_is_paired
            else interval_features(None, -1)
        )
        row = {
            "pair_local_key": site.pair_local_key,
            "pair_id": int(site.pair_id),
            "site_id": site.site_id,
            "gene": manifest_row["gene"],
            "cell_line": manifest_row["cell_line"],
            "experiment_accession": manifest_row["experiment_accession"],
            "file_accession": manifest_row["file_accession"],
            "assay_track_available": True,
            "legacy_archived_control": bool(manifest_row["legacy_archived_control"]),
            "negative_peak_call_is_uncertain_in_repeats": True,
            "focal_is_paired": bool(site.focal_is_paired),
            "focal_arm": int(site.focal_arm),
            "partner_arm": int(site.partner_arm),
            **prefix_features("focal", focal),
            **prefix_features("partner", partner),
            **prefix_features("shifted_focal", shifted_focal),
            **prefix_features("shifted_partner", shifted_partner),
            **pair_cache[site.pair_id],
        }
        arm = pair_cache[site.pair_id]
        for name in ("peak_count", "covered_nt", "coverage_fraction", "max_signal"):
            row[f"focal_arm_{name}"] = arm[f"arm{site.focal_arm}_{name}"]
            other = 2 if site.focal_arm == 1 else 1
            row[f"opposite_arm_{name}"] = arm[f"arm{other}_{name}"]
        rows.append(row)
    return pd.DataFrame(rows)


def run(site_master_path: Path, manifest_path: Path, output_path: Path) -> dict:
    sites = prepare_sites(pd.read_parquet(site_master_path))
    manifest = pd.read_csv(manifest_path)
    required = {"gene", "cell_line", "file_accession", "local_path"}
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"manifest lacks {required - set(manifest.columns)}")
    tables = [annotate_track(sites, row) for _, row in manifest.iterrows()]
    output = pd.concat(tables, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    summary = {
        "scope": "development train+valid only",
        "windows_nt_prespecified": list(WINDOWS),
        "negative_control_genomic_shift_nt": GENOMIC_SHIFT_CONTROL_NT,
        "site_count": int(sites["pair_local_key"].nunique()),
        "track_count": int(manifest["file_accession"].nunique()),
        "feature_rows": len(output),
        "genes": sorted(manifest["gene"].unique()),
        "cell_lines": sorted(manifest["cell_line"].unique()),
        "repeat_mapping_caveat": (
            "A negative peak call is uncertain because standard eCLIP peak tracks "
            "can under-represent multi-mapping inverted-Alu reads."
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("context/rbp/encode_eclip_manifest.csv")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/rbp/eclip_track_features.parquet"),
    )
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    summary = run(
        resolve(args.site_master), resolve(args.manifest), resolve(args.output)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
