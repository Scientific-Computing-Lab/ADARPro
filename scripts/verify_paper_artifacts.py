#!/usr/bin/env python3
"""Verify manuscript formatting, traceability, and generated-artifact integrity."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latex_words(text: str) -> list[str]:
    text = re.sub(r"\\cite\{[^}]+\}", "", text)
    text = re.sub(r"\\[A-Za-z]+(?:\[[^]]*\])?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+|[$%{}]", " ", text)
    return re.findall(r"\b[\w'-]+\b", text)


def verify_pnas_fields() -> None:
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    if r"\templatetype{pnasresearcharticle}" not in main:
        raise AssertionError("manuscript does not select pnasresearcharticle")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.DOTALL)
    significance = re.search(r"\\significancestatement\{(.*?)\}\n", main, re.DOTALL)
    keywords = re.search(r"\\keywords\{(.*?)\}", main, re.DOTALL)
    if not abstract or len(latex_words(abstract.group(1))) > 250:
        raise AssertionError("abstract is absent or exceeds 250 words")
    if not significance:
        raise AssertionError("significance statement is absent")
    significance_count = len(latex_words(significance.group(1)))
    if not 50 <= significance_count <= 120:
        raise AssertionError("significance statement must contain 50--120 words")
    if not keywords or not 3 <= len(keywords.group(1).split(r"$|$")) <= 5:
        raise AssertionError("manuscript must contain 3--5 keywords")
    if "an internal instrument for estimating incremental signal" not in main:
        raise AssertionError("development/canonical regime distinction is absent")
    if (
        "One authorization then exposed the canonical test for exactly one evaluation"
        not in main
    ):
        raise AssertionError("single-use canonical-test statement is absent")


def verify_claim_traceability() -> None:
    with (PAPER / "claim_traceability.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24 or len({row["claim_id"] for row in rows}) != 24:
        raise AssertionError("claim traceability must contain 24 unique claims")
    for row in rows:
        for item in row["authoritative_artifact"].split("; "):
            paths = list(ROOT.glob(item)) if "*" in item else [ROOT / item]
            if not paths or any(not path.exists() for path in paths):
                raise AssertionError(f"{row['claim_id']}: missing artifact {item}")


def verify_manifest(path: Path, file_column: str, rooted: bool) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        artifact = ROOT / row[file_column] if rooted else path.parent / row[file_column]
        if not artifact.exists() or sha256(artifact) != row["sha256"]:
            raise AssertionError(f"generated-artifact hash mismatch: {artifact}")


def verify_quantitative_sources() -> None:
    verify_manifest(
        PAPER / "source_tables/figure_source_manifest.csv", "artifact", rooted=True
    )
    verify_manifest(PAPER / "generated/table_manifest.csv", "file", rooted=False)

    locked = pd.read_csv(ROOT / "results/adarpro/final_locked_test/metrics.csv")
    source = pd.read_csv(PAPER / "source_tables/figure4_locked_metrics.csv")
    columns = [
        "context",
        "model",
        "auroc",
        "auprc",
        "f1",
        "precision",
        "recall",
        "specificity",
        "brier",
    ]
    left = locked[columns].sort_values(["context", "model"]).reset_index(drop=True)
    right = source[columns].sort_values(["context", "model"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=True)


def verify_template_provenance() -> None:
    readme = (PAPER / "template/README.md").read_text(encoding="utf-8")
    rows = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", readme)
    if len(rows) != 11:
        raise AssertionError("template provenance table must contain 11 files")
    for name, expected in rows:
        path = PAPER / "template" / name
        if not path.exists() or sha256(path) != expected:
            raise AssertionError(f"PNAS template hash mismatch: {path}")


def verify_compilation() -> None:
    for name in ("main", "supplementary"):
        pdf = PAPER / f"{name}.pdf"
        log = PAPER / f"{name}.log"
        if not pdf.exists() or pdf.stat().st_size < 10_000:
            raise AssertionError(f"missing or implausibly small PDF: {pdf}")
        if not log.exists():
            raise AssertionError(f"missing compilation log: {log}")
        text = log.read_text(encoding="utf-8", errors="replace")
        forbidden = (
            "There were undefined references",
            "Citation `",
            "Emergency stop",
            "Fatal error",
        )
        if any(marker in text for marker in forbidden):
            raise AssertionError(f"unresolved LaTeX problem in {log}")


def main() -> None:
    verify_pnas_fields()
    verify_claim_traceability()
    verify_quantitative_sources()
    verify_template_provenance()
    verify_compilation()
    print(
        "PASS: PNAS limits, claim traceability, quantitative source hashes, "
        "and manuscript builds verified"
    )


if __name__ == "__main__":
    main()
