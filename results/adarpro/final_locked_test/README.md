# Final locked canonical-test artifacts

These files were produced by the single authorized invocation of
`python -m adarpro.evaluation.final_locked` on 2026-08-25.

- `metrics.csv`: reproduced published AdarEdit and frozen ADARpro metrics.
- `{context}_pair_bootstrap_ci.csv`: 95% pair-clustered confidence intervals
  for ADARpro minus published AdarEdit.
- `{context}_predictions.parquet`: ordered site-level predictions used for
  post-lock descriptive analysis.
- `{context}_pair_bootstrap.parquet`: the 2,000 clustered bootstrap draws.
- `summary.json`: candidate, manifest and result hashes.

The governing manifest is `provenance/final_pipeline_manifest.json` (SHA-256
`dd04f9786d52f57653da31b12dc777d10aa4247dfccd2b2c59ddff64ad395a1c`).
The quarantine records `evaluations_consumed = 1`; these outputs must not be
used to reopen model or threshold selection.
