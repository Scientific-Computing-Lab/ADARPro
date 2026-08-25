# Final repository and paper audit

Audit date: 2026-08-25 UTC

## Scope and preserved scientific record

The release contains the complete ADARPro campaign rather than only the
selected model. It includes the pinned AdarEdit reproduction, canonical human
inputs, development folds, 128-row append-only experiment registry,
development OOF predictions and statistics, biological and computational
controls, frozen selected checkpoints, the preregistered unified-model result,
the single canonical-test output, chronological reports 00--20, and the final
scientific synthesis. `paper/claim_traceability.csv` maps every principal
manuscript claim to its authoritative artifact and records its epistemic
status.

The manuscript preserves two distinct performance regimes:

- the canonical held-out benchmark is the authoritative final comparison with
  published/supplied-checkpoint AdarEdit;
- pair-disjoint development OOF comparisons are same-fold instruments for
  attribution and model selection, not replacements for the published
  benchmark.

The selected predictor remains the frozen 0.5/0.5 equal-logit ensemble of
independently trained pair-balanced and target-local components. The unified
end-to-end model was rejected on preregistered development evidence before
test access. The canonical test was consumed once after the final manifest was
frozen.

## Publication package

- PNAS Research Report source and compiled PDF: `paper/main.tex` and
  `paper/main.pdf` (7 pages).
- Supporting Information source and compiled PDF: `paper/supplementary.tex`
  and `paper/supplementary.pdf` (19 pages).
- Four main and six supplementary figures in PDF, SVG, and PNG.
- Machine-readable source tables and deterministic scripts for every
  quantitative figure and table.
- Thirty-five DOI-audited bibliography records; all cited keys pass static and
  live Crossref checks. Preprints are explicitly labeled.
- Official 2025 PNAS Research Report and SI templates, support files, source
  URLs, versions, and SHA-256 values under `paper/template/`.

The abstract is 230 words, the Significance Statement is 105 words, and five
keywords are supplied. Author-specific metadata remains visibly unresolved;
no author identity or declaration was invented.

## Intentionally excluded artifacts

Third-party raw downloads, provider model weights, four large deterministic
feature matrices, exploratory fold checkpoints, caches, virtual environments,
and LaTeX intermediates are not committed. This is a size, redistribution, or
rebuildability decision, not outcome selection. Every scientifically relevant
exclusion has a source identity, role, byte size, SHA-256 value, and recovery
instructions in `provenance/source_workspace_inventory.csv`,
`provenance/excluded_artifacts.csv`, resource-specific provenance files, and
`docs/DATA.md`. All small source and result tables needed to regenerate the
paper are included.

## Quality-control status

The publication preflight passed in the pinned project environment. The same
checks passed from a fresh clone of the GitHub-hosted publication commit
`aa406d00fecb9df7246343b7574a0cd12fd06a42` at
`/tmp/adarpro-publication-XIuhkZ/ADARPro`:

- editable installation from repository metadata;
- 884-pair split and zero-overlap verification;
- authoritative canonical metrics, confidence intervals, freeze hashes,
  quarantine state, and registry integrity;
- complete Python tests and Ruff checks;
- release-manifest verification and secret/path scan;
- deterministic regeneration of paper figures and SI tables;
- PNAS main-paper and SI compilation;
- abstract, Significance Statement, keyword, citation, claim-traceability, and
  quantitative-source audits;
- visual inspection of every main and supplementary PDF page.

The result was 884 substrates with no within-split or cross-tissue leakage,
120 passing tests, a clean Ruff check, 128 regenerated evaluations, 10
regenerated vector figures, eight regenerated SI tables, two successful PNAS
LaTeX builds, 35 DOI-audited bibliography records, and passing scientific,
quarantine, provenance, figure-source, claim-traceability, and formatting
audits. The rebuilt main paper has seven pages and the SI has 19 pages. The
clone remained clean after regeneration, confirming that reported outputs are
reproducible byte for byte in the pinned environment.

The publication-content SHA above is the commit subjected to the complete
clean-clone audit. The final handoff also reports the subsequent audit-only
commit containing this record and the refreshed integrity manifest, because a
Git commit cannot embed its own identifier. No force push was used.

## Remaining submission placeholders

The following require information from the authors before journal submission:

- author names and order;
- affiliations and corresponding author;
- CRediT/author-contribution statement;
- funding and acknowledgments;
- competing-interest declaration;
- journal-assigned DOI.

There are no unresolved scientific-result, data-provenance, figure-source, or
compilation placeholders.
