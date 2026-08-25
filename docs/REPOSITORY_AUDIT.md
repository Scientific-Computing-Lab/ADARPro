# Repository consolidation audit

This document is completed automatically and manually at release milestones.

## Consolidation policy

- The original research workspace was not modified or repointed.
- The destination was populated through a clean curated export.
- Negative and terminated experiments are retained through reports, registry
  rows, result summaries, source tables, and OOF predictions.
- Excluded large artifacts are hash-addressed in a machine-readable manifest.
- Machine-local absolute paths were normalized; original and release hashes
  are both retained.
- No credential, API key, virtual environment, cache, or temporary build file
  is intended for release.
- No force-push is permitted.

## Required release checks

Run `make verify` and record the final commit, clean-clone directory, install
status, split result, test/lint result, result-table build, figure build, and
paper/SI compilation below. The publication release commit will update this
section after all checks pass.

## Phase I verification

The GitHub-hosted commit `c93dc8e` was cloned into a new temporary directory
and verified on 2026-08-25. The editable package installed in the pinned
environment; the 884-pair split check passed; all 120 tests passed; Ruff passed;
128 AdarEdit result evaluations were regenerated; the authoritative locked
result, freeze, quarantine, and 128-run registry passed their integrity checks;
the 1,724-file release manifest verified; and seven development figures plus
their source tables were regenerated outside the checkout. The clone remained
clean.

Status: **Phase I complete**. A second fresh-clone audit will be recorded after
the manuscript and Supplementary Information are added.

## Publication-package preflight

The PNAS main manuscript, Supporting Information, quantitative figure source
tables, deterministic renderers, DOI-audited bibliography, and claim-level
traceability table passed the pinned-environment preflight on 2026-08-25. The
preflight included all 120 tests, Ruff, the authoritative-result/freeze/
quarantine checks, generation of 128 evaluation rows, four main and six SI
figures, eight SI tables, both LaTeX builds, and static plus live Crossref
reference verification. A final clean-clone status is added only after the
publication commit has been fetched from GitHub and tested independently.
