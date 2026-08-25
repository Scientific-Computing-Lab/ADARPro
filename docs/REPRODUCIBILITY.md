# Reproducibility

## Immutable scientific identities

- Upstream AdarEdit: `Scientific-Computing-Lab/AdarEdit` at
  `5c8b57d70cdc2eb1c5e168431013275bcc9ed0c5`.
- Final selected family: `adarpro_pair_local_equal_logit_ensemble`.
- Fusion: fixed 0.5/0.5 average of component logits; weights were not searched.
- Frozen development code revision:
  `0f50fac33bec8bad325e6745d83ceaacebe42fc1`.
- Final pipeline manifest SHA-256:
  `dd04f9786d52f57653da31b12dc777d10aa4247dfccd2b2c59ddff64ad395a1c`.
- Locked metrics SHA-256:
  `ddd5352c48e84bb121e21d89e5d1777a3657caa06f28fa94ef1e8b3c58a74a0e`.
- Canonical-test state: `final_consumed`, one evaluation, consumed
  2026-08-25 04:28:30 UTC.

The original research workspace is preserved separately. The public release
normalizes only machine-local paths; the source and release hashes for every
such file are recorded in `provenance/source_workspace_inventory.csv`.

## Environment

`environment.yml` is the checkpoint-compatible environment inherited from the
published AdarEdit package. `pyproject.toml` installs the ADARPro library and
lightweight development/table dependencies. Exact resolved conda and pip
packages are retained in `provenance/`.

```bash
conda env create -f environment.yml
conda activate adaredit
python -m pip install -e '.[dev,tables]'
```

GPU reproduction of training used the hardware recorded in the provenance
files. Table, figure, statistical, and paper regeneration do not require a
GPU.

## Core verification

```bash
python data_construction/verify_split.py
python scripts/verify_authoritative_results.py
python results/build_results_tables.py
pytest -q
ruff check src tests scripts
python scripts/build_release_manifest.py --verify
```

`make verify` runs this sequence. The split check must report zero pair overlap.
The authoritative-result check validates the final metrics, pair-bootstrap
direction, false-positive counts, frozen manifest/checkpoint hashes,
single-use quarantine record, unified-model decision, and registry uniqueness.

## Regenerating reported outputs

- Original AdarEdit result matrices:
  `python results/build_results_tables.py`.
- Development campaign figures:
  `python -m adarpro.figures --help` and the commands documented by each
  source-table report.
- Locked descriptive figures:
  `python -m adarpro.evaluation.final_figures --help`.
- Paper source tables: `make paper-tables`.
- Paper figures: `make paper-figures`.
- Main paper and SI: `make paper`.
- Complete manuscript audit: `make paper-check`.

Paper scripts read retained authoritative CSV/Parquet/JSON artifacts and never
rerun model selection. Quantitative figure panels are traceable through
`paper/source_tables/figure_source_manifest.csv`.
Figure and manuscript PDF timestamps are fixed to the date of the locked test,
so repeated builds from an unchanged checkout are byte-reproducible.

The PNAS build requires a TeX Live installation with the packages used by the
official `pnas-new` class. On Debian/Ubuntu, `texlive-latex-extra`,
`texlive-fonts-recommended`, and `lmodern` provide the required dependencies.
The repository vendors the PNAS class/style files and records their origin and
hashes in `paper/template/README.md`; it does not vendor general TeX packages.
The reference audit can also resolve every recorded DOI against Crossref with
`python scripts/verify_paper_references.py --online`.

## Leakage controls

The 884 graph-buildable Alu-pair substrates share one global partition:
566 train, 139 validation, and 179 test pairs. Pair identity, not site row, is
the grouping unit.

During development:

- canonical test labels were quarantined;
- all learned preprocessing, selection, checkpointing, thresholds, and
  hyperparameters excluded the outer pair fold;
- out-of-fold predictions were produced before residual analysis;
- feature and model comparisons used identical pair folds;
- uncertainty resampled whole Alu pairs;
- natural-prevalence analyses retained the published pair assignment;
- the unified model was preregistered and rejected without test access.

After development, the five context-specific component pairs were retrained on
published training pairs, epochs and thresholds were selected on published
validation only, all artifacts were hashed, and a single authorization was
consumed. No post-test tuning occurred.

## Two performance regimes

The repository intentionally retains two views:

- **Canonical held-out benchmark:** supplied-checkpoint AdarEdit versus frozen
  ADARPro on the original test. This is the final performance claim.
- **Same-fold development comparison:** cross-fitted candidates versus the
  identical-fold internal reference. This attributes incremental signal but is
  not an independent final test and does not replace the published baseline.

Development deltas are never added arithmetically to published test metrics.

## Integrity and audit trail

- `provenance/final_pipeline_manifest.json`: frozen model identity.
- `provenance/test_quarantine.json`: state and input hashes.
- `provenance/test_access_audit.jsonl`: chained access events.
- `provenance/model_selection_policy.json`: ensemble selection policy.
- `provenance/unified_model_selection_policy.json`: preregistered unified gates.
- `provenance/unified_development_decision.json`: development-only decision.
- `provenance/source_workspace_inventory.csv`: source-to-release inventory.
- `provenance/release_manifest.sha256`: curated checkout integrity.
