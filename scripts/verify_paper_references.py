#!/usr/bin/env python3
"""Audit manuscript citation keys and optionally resolve every DOI via Crossref."""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "references.bib"
AUDIT = ROOT / "paper" / "reference_audit.csv"
TEX_FILES = [ROOT / "paper" / "main.tex", ROOT / "paper" / "supplementary.tex"]


def bib_records(text: str) -> dict[str, str]:
    starts = list(re.finditer(r"@\w+\{([^,]+),", text))
    records: dict[str, str] = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        records[match.group(1)] = text[match.start() : end]
    return records


def doi_from_record(record: str) -> str:
    match = re.search(r"\bdoi\s*=\s*\{([^}]+)\}", record, flags=re.IGNORECASE)
    return match.group(1).lower() if match else ""


def cited_keys() -> set[str]:
    keys: set[str] = set()
    for path in TEX_FILES:
        for match in re.finditer(
            r"\\cite\{([^}]+)\}", path.read_text(encoding="utf-8")
        ):
            keys.update(key.strip() for key in match.group(1).split(","))
    return keys


def crossref_title(doi: str) -> str:
    encoded = urllib.parse.quote(doi, safe="")
    request = urllib.request.Request(
        f"https://api.crossref.org/works/{encoded}",
        headers={
            "User-Agent": (
                "ADARPro-reference-audit/1.0 "
                "(+https://github.com/Scientific-Computing-Lab/ADARPro)"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        message = json.load(response)["message"]
    return message["title"][0]


def normalize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--online", action="store_true", help="resolve all DOIs via Crossref"
    )
    args = parser.parse_args()

    records = bib_records(BIB.read_text(encoding="utf-8"))
    with AUDIT.open(newline="", encoding="utf-8") as handle:
        audit_rows = list(csv.DictReader(handle))
    audited = {row["key"]: row for row in audit_rows}

    citations = cited_keys()
    missing_bib = sorted(citations - records.keys())
    missing_audit = sorted(citations - audited.keys())
    if missing_bib or missing_audit:
        raise SystemExit(
            f"Missing bibliography={missing_bib}; missing audit={missing_audit}"
        )

    failures: list[str] = []
    for key, row in audited.items():
        if key not in records:
            failures.append(f"{key}: audit row missing from bibliography")
            continue
        bib_doi = doi_from_record(records[key])
        if bib_doi != row["doi"].lower():
            failures.append(f"{key}: DOI mismatch ({bib_doi} != {row['doi']})")
        if args.online:
            remote_title = crossref_title(row["doi"])
            expected = normalize(row["title"])
            observed = normalize(remote_title)
            overlap = len(expected & observed) / max(1, len(expected | observed))
            if overlap < 0.75:
                failures.append(f"{key}: Crossref title mismatch ({overlap:.2f})")

    if failures:
        raise SystemExit("Reference audit failed:\n" + "\n".join(failures))
    mode = "Crossref + static" if args.online else "static"
    print(
        f"PASS: {mode} reference audit; {len(audited)} DOI records; "
        f"{len(citations)} cited keys"
    )


if __name__ == "__main__":
    main()
