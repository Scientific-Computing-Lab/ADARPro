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

Status: pending clean-clone verification.
