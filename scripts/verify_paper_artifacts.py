#!/usr/bin/env python3
"""Verify manuscript formatting, traceability, and generated-artifact integrity."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
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
    regime_concepts = (
        ("pair-disjoint out-of-fold", "pair-disjoint oof"),
        (
            "frozen before the canonical held-out evaluation",
            "frozen before canonical held-out evaluation",
            "frozen specification was evaluated on the canonical test",
        ),
        (
            "component attribution comes from adaptively selected",
            "development evidence therefore attributes branch-level effects",
        ),
    )
    manuscript = main.lower()
    if not all(
        any(marker in manuscript for marker in group) for group in regime_concepts
    ):
        raise AssertionError("development/canonical regime distinction is absent")


def verify_claim_traceability() -> None:
    with (PAPER / "claim_traceability.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 31 or len({row["claim_id"] for row in rows}) != 31:
        raise AssertionError("claim traceability must contain 31 unique claims")
    for row in rows:
        for item in row["authoritative_artifact"].split("; "):
            paths = list(ROOT.glob(item)) if "*" in item else [ROOT / item]
            if not paths or any(not path.exists() for path in paths):
                raise AssertionError(f"{row['claim_id']}: missing artifact {item}")

    ledger_path = PAPER / "numerical_claim_ledger.csv"
    with ledger_path.open(newline="", encoding="utf-8") as handle:
        ledger = list(csv.DictReader(handle))
    if len(ledger) < 300 or len({row["claim_id"] for row in ledger}) != len(ledger):
        raise AssertionError(
            "numerical claim ledger is incomplete or has duplicate IDs"
        )
    for row in ledger:
        artifact = ROOT / row["authoritative_artifact"]
        if not artifact.exists():
            raise AssertionError(
                f"{row['claim_id']}: missing numerical source {artifact}"
            )
    generated = PAPER / "generated/quantitative_claims.tex"
    if (
        not generated.exists()
        or "\\ArteryADARProFOne}{0.8869}" not in generated.read_text(encoding="utf-8")
    ):
        raise AssertionError("generated quantitative macros are absent or stale")


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
    with (PAPER / "source_tables/figure_source_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        if any(
            row["artifact"].startswith("paper/figure_assets/")
            for row in csv.DictReader(handle)
        ):
            raise AssertionError(
                "historical image-model assets entered the active figure build"
            )

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

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "| Artery | 0.9238 | 0.9477 | 0.9262 | 0.9485 | 0.8671 | 0.8869 |" not in readme:
        raise AssertionError("README canonical table is not synchronized")


def verify_svg_figures() -> None:
    """Require SVG placement, vector-only content, and PNAS-matched typography."""
    placed: set[Path] = set()
    for manuscript in (PAPER / "main.tex", PAPER / "supplementary.tex"):
        text = manuscript.read_text(encoding="utf-8")
        if re.search(r"\\includegraphics[^\n]*\{figures/", text):
            raise AssertionError(
                f"{manuscript} places a figure through includegraphics"
            )
        names = re.findall(r"\\includesvg(?:\[[^]]*\])?\{([^}]+\.svg)\}", text)
        if not names:
            raise AssertionError(f"{manuscript} contains no SVG figure placements")
        placed.update(manuscript.parent / name for name in names)

    for figure in sorted(placed):
        if not figure.exists():
            raise AssertionError(f"missing placed SVG figure: {figure}")
        markup = figure.read_text(encoding="utf-8")
        if "<image" in markup or "data:image" in markup:
            raise AssertionError(f"SVG contains embedded raster content: {figure}")
        if "Nimbus Sans" not in markup:
            raise AssertionError(f"SVG does not use the PNAS-matched font: {figure}")
        cache = figure.with_name(f"{figure.stem}_svg-raw.pdf")
        if not cache.exists():
            raise AssertionError(f"missing svg.sty PDF derivative: {cache}")


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


def verify_main_pdf_page_bounds() -> None:
    """Reject rendered main-paper text that falls beyond a physical page edge."""
    executable = shutil.which("pdftotext")
    if executable is None:
        raise AssertionError(
            "pdftotext is required for the PDF page-boundary check "
            "(install poppler-utils)"
        )
    pdf = PAPER / "main.pdf"
    with tempfile.TemporaryDirectory(prefix="adarpro-pdf-bounds-") as tmpdir:
        bbox = Path(tmpdir) / "main.html"
        subprocess.run(
            [executable, "-bbox-layout", str(pdf), str(bbox)],
            check=True,
            capture_output=True,
            text=True,
        )
        root = ET.parse(bbox).getroot()

    violations: list[str] = []
    pages = [element for element in root.iter() if element.tag.endswith("page")]
    for page_number, page in enumerate(pages, start=1):
        width = float(page.attrib["width"])
        height = float(page.attrib["height"])
        for word in (
            element for element in page.iter() if element.tag.endswith("word")
        ):
            x_min = float(word.attrib["xMin"])
            x_max = float(word.attrib["xMax"])
            y_min = float(word.attrib["yMin"])
            y_max = float(word.attrib["yMax"])
            tolerance = 0.5
            if (
                x_min < -tolerance
                or x_max > width + tolerance
                or y_min < -tolerance
                or y_max > height + tolerance
            ):
                content = "".join(word.itertext()).strip()
                violations.append(
                    f"page {page_number}: {content!r} at "
                    f"({x_min:.2f}, {y_min:.2f})--({x_max:.2f}, {y_max:.2f})"
                )
    if violations:
        details = "\n".join(violations[:10])
        raise AssertionError(f"main-paper content exceeds page bounds:\n{details}")


def main() -> None:
    verify_pnas_fields()
    verify_claim_traceability()
    verify_quantitative_sources()
    verify_svg_figures()
    verify_template_provenance()
    verify_compilation()
    verify_main_pdf_page_bounds()
    print(
        "PASS: PNAS limits, claim traceability, quantitative source hashes, "
        "vector-only SVG placement, manuscript builds, and main-paper page bounds verified"
    )


if __name__ == "__main__":
    main()
