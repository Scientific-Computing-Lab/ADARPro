# Final repository and paper audit

Audit date: 2026-08-25 UTC

## Preserved scientific record

The repository contains the complete ADARPro campaign rather than only the
selected predictor: the pinned AdarEdit reproduction, canonical human inputs,
pair-disjoint development folds, append-only experiment registry, OOF
predictions and clustered statistics, biological and computational controls,
frozen checkpoints, preregistered unified-model result, single canonical-test
output, reports 00--20, and the final synthesis.

The selected model remains the unchanged 0.5/0.5 equal-logit ensemble of
independently trained pair-first whole-duplex and target-local components. The
unified end-to-end model was rejected by its preregistered development rule
before test access. The canonical test was authorized and evaluated once after
the final manifest was frozen.

The manuscript keeps the two evaluation regimes separate:

- the supplied-checkpoint canonical held-out benchmark is the authoritative
  final comparison with published AdarEdit;
- pair-disjoint OOF comparisons use an identical-fold internal reference for
  attribution and development selection only.

## Publication package

- PNAS source/PDF: `paper/main.tex`, `paper/main.pdf` (7 pages).
- SI source/PDF: `paper/supplementary.tex`, `paper/supplementary.pdf` (10 pages).
- Four main and six supplementary figures in PDF, SVG, and PNG.
- Ten generated SI tables and machine-readable source tables.
- A 375-row numerical claim ledger plus generated LaTeX macros.
- Thirty-five DOI-audited bibliography records; 30 are cited.
- Official 2025 PNAS template files with source URLs, versions, and hashes.

The abstract has 233 words, the Significance Statement 106 words, and five
keywords are supplied. Author-specific metadata remains visibly unresolved;
no identity, contribution, funding, or declaration was invented.

## Excluded artifacts

Third-party raw downloads, provider weights, large deterministic feature
matrices, exploratory fold checkpoints, caches, virtual environments, and
LaTeX intermediates are not committed. This reflects size, redistribution, or
rebuildability, not outcome selection. Each scientifically relevant exclusion
has a source identity, role, byte size, SHA-256 value, and recovery instructions
in the provenance manifests and `docs/DATA.md`. All small tables needed to
regenerate the paper are retained.

## Quality-control status

The publication-content commit
`12e01b24150944dffd9f07feac5e36ecad84d264` passed local verification:

- 884 substrates and zero within-split/cross-tissue leakage;
- 120 Python tests and a clean Ruff check;
- 128 regenerated authoritative evaluations;
- authoritative metrics, confidence intervals, freeze hashes, quarantine
  state, and registry integrity;
- deterministic regeneration of 10 vector figures, 10 SI tables, and 375
  numerical claims/macros;
- successful PNAS main-paper and SI compilation;
- static and live Crossref audit of all DOI records;
- visual inspection of every main and SI page.

The final release manifest and a fresh-clone verification are recorded by the
subsequent audit-only commit. A Git commit cannot embed its own identifier, so
the exact final repository HEAD is reported in the release handoff as well as
by Git history. No force push is used.

## Remaining submission placeholders

- author names, order, affiliations, and corresponding-author details;
- author-contribution statement;
- funding and acknowledgments;
- competing-interest declaration;
- journal-assigned DOI.

There are no unresolved scientific-result, data-provenance, figure-source, or
compilation placeholders. See `FINAL_PAPER_AUDIT.md` for the detailed
editorial, numerical, figure, and reference audit.
