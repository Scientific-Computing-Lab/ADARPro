"""Download replicated GRCh38 ENCODE eCLIP peaks with complete provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ENCODE = "https://www.encodeproject.org"


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def experiments(gene: str) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "type": "Experiment",
            "assay_title": "eCLIP",
            "target.label": gene,
            "format": "json",
            "limit": "all",
        }
    )
    try:
        payload = request_json(f"{ENCODE}/search/?{query}")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return []
        raise
    return payload.get("@graph", [])


def combined_grch38_peaks(
    experiment: dict, *, allow_archived: bool = False
) -> list[dict]:
    accession = experiment["accession"]
    payload = request_json(f"{ENCODE}/experiments/{accession}/?format=json")
    return [
        file
        for file in payload.get("files", [])
        if (file.get("status") == "released" or allow_archived)
        and file.get("file_format") == "bed"
        and file.get("assembly") == "GRCh38"
        and file.get("output_type") == "peaks"
        and len(file.get("biological_replicates", [])) > 1
    ]


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with urllib.request.urlopen(url) as source, temporary.open("wb") as destination:
        while chunk := source.read(1024 * 1024):
            destination.write(chunk)
    temporary.replace(path)


def run(panel_path: Path, output_dir: Path, manifest_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(panel_path)
    genes = panel["gene"].tolist()
    rows = []
    availability_rows = []
    old_retrieval = {}
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        old_retrieval = dict(zip(old["file_accession"], old["retrieved_utc"]))
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for gene in genes:
        found = experiments(gene)
        selected_for_gene = 0
        for experiment in found:
            accession = experiment["accession"]
            metadata = request_json(
                f"{ENCODE}/experiments/{accession}/?format=json"
            )
            (metadata_dir / f"{accession}.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            # ADAR is retained as an explicitly marked legacy positive control:
            # its only ENCODE eCLIP experiment and files are archived. Archived
            # tracks are never silently mixed with released candidate tracks.
            allow_archived = gene == "ADAR" and metadata.get("status") == "archived"
            selected = combined_grch38_peaks(
                experiment, allow_archived=allow_archived
            )
            selected_for_gene += len(selected)
            for file in selected:
                cell = experiment["biosample_summary"].split()[-1]
                file_accession = file["accession"]
                url = ENCODE + file["href"]
                path = output_dir / gene / cell / f"{file_accession}.bed.gz"
                if not path.exists():
                    download(url, path)
                observed_md5 = digest(path, "md5")
                if observed_md5 != file["md5sum"]:
                    raise RuntimeError(
                        f"MD5 mismatch for {file_accession}: {observed_md5}"
                    )
                rows.append(
                    {
                        "gene": gene,
                        "role": panel.loc[panel["gene"] == gene, "role"].iloc[0],
                        "experiment_accession": accession,
                        "experiment_status": metadata.get("status"),
                        "biosample": experiment["biosample_summary"],
                        "cell_line": cell,
                        "file_accession": file_accession,
                        "file_status": file.get("status"),
                        "legacy_archived_control": bool(allow_archived),
                        "assembly": file["assembly"],
                        "output_type": file["output_type"],
                        "file_format_type": file.get("file_format_type"),
                        "biological_replicates": ";".join(
                            str(value) for value in file["biological_replicates"]
                        ),
                        "url": url,
                        "encode_md5": file["md5sum"],
                        "sha256": digest(path, "sha256"),
                        "bytes": path.stat().st_size,
                        "retrieved_utc": old_retrieval.get(
                            file_accession, datetime.now(timezone.utc).isoformat()
                        ),
                        "local_path": str(path.resolve()),
                    }
                )
        availability_rows.append(
            {
                "gene": gene,
                "encode_eclip_experiment_count": len(found),
                "selected_grch38_combined_track_count": selected_for_gene,
                "has_selected_track": selected_for_gene > 0,
                "selection_rule": (
                    "released replicated GRCh38 peaks; archived ADAR allowed only "
                    "as labeled positive control"
                ),
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["gene", "cell_line"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    pd.DataFrame(availability_rows).to_csv(
        manifest_path.with_name("encode_eclip_availability.csv"), index=False
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--panel", type=Path, default=Path("context/rbp/candidate_panel.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/external/encode_eclip")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("context/rbp/encode_eclip_manifest.csv"),
    )
    args = parser.parse_args()
    repository = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repository / path

    manifest = run(resolve(args.panel), resolve(args.output_dir), resolve(args.manifest))
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
