# PNAS manuscript package

The manuscript is formatted as a PNAS Research Report using the current 2025
Research Report template and the official Supporting Information template.
Template origins and hashes are recorded in `template/README.md`.

## Build

From the repository root:

```bash
make paper
```

This regenerates all quantitative figures, the numerical claim ledger and
LaTeX macros, and SI tables from retained authoritative artifacts, then builds:

- `paper/main.pdf`
- `paper/supplementary.pdf`

The PNAS class requires a normal TeX Live installation. On Debian/Ubuntu the
non-base dependencies are supplied by `texlive-latex-extra`,
`texlive-fonts-recommended`, and `lmodern`. `make paper-check` additionally
uses Poppler's `pdftotext` to verify that rendered text remains within every
main-paper page boundary (`poppler-utils` on Debian/Ubuntu).

## Traceability

- `source_tables/` contains the data used by every quantitative panel.
- `generated/` contains deterministic LaTeX tables.
- `numerical_claim_ledger.csv` records each reported value, display rounding,
  epistemic status, authoritative artifact, selector, and manuscript use.
- `claim_traceability.csv` maps principal numerical claims to campaign
  artifacts and labels their epistemic status.
- `EPISTEMIC_CLAIM_AUDIT.md` records the inferential boundary for every major
  scientific claim, and `TERMINOLOGY_AUDIT.md` fixes paper-facing vocabulary.
- `reference_audit.csv` records DOI-level bibliographic verification.
- `scripts/render_paper_figures.py` and `scripts/render_paper_tables.py`
  regenerate the publication outputs.

The manuscript deliberately separates pair-disjoint development OOF evidence
from the authoritative, one-use canonical held-out comparison. Development
deltas are never added to published test metrics.

## Author placeholders

Author names/order, affiliations, contributions, correspondence, funding,
acknowledgments, and the competing-interest declaration require confirmation
before submission. No values were invented for these fields.
