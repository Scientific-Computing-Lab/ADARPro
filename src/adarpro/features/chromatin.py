"""Acquire and featurize exact-tissue ENCODE H3K36me3 peak tracks.

The module deliberately starts with released GRCh38 pseudoreplicated peak
files.  Large signal tracks are not silently substituted with cell-line
proxies.  Every selected file and its ENCODE MD5 is frozen in a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ENCODE = "https://www.encodeproject.org"
BIOSAMPLES = {
    "Artery": "tibial artery",
    "Liver": "liver",
    "MuscleSkeletal": "skeletal muscle tissue",
}
BED_COLUMNS = [
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


def get_json(path: str, parameters: dict[str, str] | None = None) -> dict:
    query = "" if not parameters else "?" + urllib.parse.urlencode(parameters)
    request = urllib.request.Request(
        ENCODE + path + query, headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def released_analysis(file_record: dict) -> bool:
    return any(
        analysis.get("status") == "released"
        for analysis in file_record.get("analyses", [])
        if isinstance(analysis, dict)
    )


def select_peak_file(experiment: dict) -> dict | None:
    candidates = [
        file_record
        for file_record in experiment.get("files", [])
        if isinstance(file_record, dict)
        and file_record.get("status") == "released"
        and file_record.get("assembly") == "GRCh38"
        and file_record.get("file_format") == "bed"
        and file_record.get("output_type")
        in {"pseudoreplicated peaks", "replicated peaks"}
        and file_record.get("preferred_default") is True
    ]
    if not candidates:
        return None
    # ENCODE can retain both ENCODE3 and ENCODE4 preferred outputs. Prefer a
    # current released analysis, then the newest created file. Some legacy
    # tissue donors have only an archived-analysis output; keeping them is
    # explicit in the manifest and preferable to silently dropping donors.
    candidates.sort(
        key=lambda item: (
            released_analysis(item),
            item.get("date_created", ""),
            item.get("accession", ""),
        ),
        reverse=True,
    )
    return candidates[0]


def build_manifest() -> dict:
    tracks = []
    excluded_experiments = []
    for context, biosample in BIOSAMPLES.items():
        search = get_json(
            "/search/",
            {
                "type": "Experiment",
                "status": "released",
                "assay_title": "Histone ChIP-seq",
                "biosample_ontology.term_name": biosample,
                "target.label": "H3K36me3",
                "format": "json",
                "limit": "all",
            },
        )
        for summary in sorted(search.get("@graph", []), key=lambda x: x["accession"]):
            experiment = get_json(f"/experiments/{summary['accession']}/", {"format": "json"})
            selected = select_peak_file(experiment)
            if selected is None:
                excluded_experiments.append(
                    {
                        "context": context,
                        "biosample": biosample,
                        "experiment": experiment["accession"],
                        "reason": "no preferred-default GRCh38 replicated/pseudoreplicated peak file",
                    }
                )
                continue
            tracks.append(
                {
                    "context": context,
                    "biosample": biosample,
                    "experiment": experiment["accession"],
                    "biosample_summary": experiment.get("biosample_summary"),
                    "file": selected["accession"],
                    "href": selected["href"],
                    "file_size": selected.get("file_size"),
                    "md5sum": selected.get("md5sum"),
                    "assembly": selected.get("assembly"),
                    "output_type": selected.get("output_type"),
                    "biological_replicates": selected.get("biological_replicates"),
                    "date_created": selected.get("date_created"),
                    "analysis": [
                        {
                            "title": analysis.get("title"),
                            "pipeline_version": analysis.get("pipeline_version"),
                            "status": analysis.get("status"),
                        }
                        for analysis in selected.get("analyses", [])
                        if isinstance(analysis, dict)
                    ],
                }
            )
    return {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "source": ENCODE,
        "target": "H3K36me3",
        "genome_build": "GRCh38",
        "selection_rule": (
            "released exact-biosample Histone ChIP-seq experiment; "
            "preferred-default GRCh38 pseudoreplicated/replicated peaks; "
            "prefer released analysis then newest file; legacy archived-analysis "
            "status retained explicitly when no released-analysis output exists"
        ),
        "tracks": tracks,
        "excluded_experiments": excluded_experiments,
    }


def md5(path: Path) -> str:
    digest = hashlib.md5()  # Required to verify ENCODE's published MD5.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_tracks(manifest: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for track in manifest["tracks"]:
        path = destination / f"{track['file']}.bed.gz"
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            urllib.request.urlretrieve(ENCODE + track["href"], temporary)
            temporary.replace(path)
        observed = md5(path)
        if observed != track["md5sum"]:
            raise RuntimeError(
                f"ENCODE checksum mismatch for {track['file']}: {observed}"
            )
        track["local_path"] = str(path)
        track["observed_md5"] = observed


def read_peaks(path: Path) -> dict[str, pd.DataFrame]:
    table = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=BED_COLUMNS,
        compression="gzip",
        usecols=range(10),
    )
    return {
        chromosome: group.sort_values("start").reset_index(drop=True)
        for chromosome, group in table.groupby("chromosome", observed=True)
    }


def query_peak_metrics(
    peaks: pd.DataFrame | None, starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return overlap, maximum signal, and nearest distance for query intervals."""

    overlap = np.zeros(len(starts), dtype=float)
    signal = np.zeros(len(starts), dtype=float)
    distance = np.full(len(starts), np.nan, dtype=float)
    if peaks is None or peaks.empty:
        return overlap, signal, distance
    peak_start = peaks["start"].to_numpy(dtype=np.int64)
    peak_end = peaks["end"].to_numpy(dtype=np.int64)
    peak_signal = peaks["signal"].to_numpy(dtype=float)
    prefix_max_end = np.maximum.accumulate(peak_end)
    max_width = int(np.max(peak_end - peak_start))
    for index, (start, end) in enumerate(zip(starts, ends)):
        right = int(np.searchsorted(peak_start, end, side="left"))
        left = int(np.searchsorted(peak_start, start - max_width, side="left"))
        local = np.flatnonzero(peak_end[left:right] > start) + left
        if len(local):
            overlap[index] = 1.0
            signal[index] = float(np.max(peak_signal[local]))
            distance[index] = 0.0
            continue
        insertion = int(np.searchsorted(peak_start, start, side="left"))
        candidates = []
        if insertion < len(peak_start):
            candidates.append(max(0, int(peak_start[insertion] - end)))
        if insertion:
            candidates.append(max(0, int(start - prefix_max_end[insertion - 1])))
        if candidates:
            distance[index] = float(min(candidates))
    return overlap, signal, distance


def track_features(
    rows: pd.DataFrame, peaks_by_chromosome: dict[str, pd.DataFrame], shift: int = 0
) -> pd.DataFrame:
    output = pd.DataFrame(index=rows.index)
    starts = rows["genomic_position_0based"].to_numpy(dtype=np.int64) + shift
    windows = {
        "site": (starts, starts + 1),
        "near250": (starts - 250, starts + 251),
        "pair": (
            rows["start1"].to_numpy(dtype=np.int64) + shift,
            rows["end2"].to_numpy(dtype=np.int64) + shift,
        ),
    }
    for scale, (left, right) in windows.items():
        overlap = np.zeros(len(rows), dtype=float)
        signal = np.zeros(len(rows), dtype=float)
        distance = np.full(len(rows), np.nan, dtype=float)
        for chromosome, indices in rows.groupby("chromosome", observed=True).groups.items():
            positions = rows.index.get_indexer(indices)
            values = query_peak_metrics(
                peaks_by_chromosome.get(str(chromosome)),
                left[positions],
                right[positions],
            )
            overlap[positions], signal[positions], distance[positions] = values
        output[f"{scale}__overlap"] = overlap
        output[f"{scale}__signal"] = signal
        output[f"{scale}__distance_log1p"] = np.log1p(distance)
    return output


def build_features(site_master: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    keys = ["example_id", "context", "pair_id", "pair_local_key"]
    output = site_master[keys].copy()
    feature_frames = []
    tracks_by_context: dict[str, list[dict]] = defaultdict(list)
    for track in manifest["tracks"]:
        tracks_by_context[track["context"]].append(track)
    for context, tracks in tracks_by_context.items():
        rows = site_master[site_master["context"] == context].copy()
        observed, shifted = [], []
        for track in tracks:
            peaks = read_peaks(Path(track["local_path"]))
            observed.append(track_features(rows, peaks, shift=0))
            shifted.append(track_features(rows, peaks, shift=1_000_000))
        context_features = pd.DataFrame(index=rows.index)
        for name in observed[0].columns:
            matrix = np.column_stack([frame[name].to_numpy() for frame in observed])
            context_features[f"chromatin__h3k36me3__{name}__donor_mean"] = np.nanmean(
                matrix, axis=1
            )
            context_features[f"chromatin__h3k36me3__{name}__donor_max"] = np.nanmax(
                matrix, axis=1
            )
            shifted_matrix = np.column_stack(
                [frame[name].to_numpy() for frame in shifted]
            )
            context_features[
                f"control_shift1mb__h3k36me3__{name}__donor_mean"
            ] = np.nanmean(shifted_matrix, axis=1)
            context_features[
                f"control_shift1mb__h3k36me3__{name}__donor_max"
            ] = np.nanmax(shifted_matrix, axis=1)
        context_features["example_id"] = rows["example_id"]
        feature_frames.append(context_features)
    if feature_frames:
        annotations = pd.concat(feature_frames, ignore_index=True)
        output = output.merge(annotations, on="example_id", how="left", validate="one_to_one")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-master", type=Path, default=Path("data/derived/site_master.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("provenance/encode_h3k36me3.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/external/chromatin/h3k36me3"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/features/chromatin_h3k36me3.parquet"))
    parser.add_argument("--refresh-manifest", action="store_true")
    args = parser.parse_args()
    if args.refresh_manifest or not args.manifest.exists():
        manifest = build_manifest()
    else:
        manifest = json.loads(args.manifest.read_text())
    download_tracks(manifest, args.data_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    features = build_features(pd.read_parquet(args.site_master), manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    print(json.dumps({"rows": len(features), "columns": len(features.columns), "tracks": len(manifest["tracks"])}))


if __name__ == "__main__":
    main()
