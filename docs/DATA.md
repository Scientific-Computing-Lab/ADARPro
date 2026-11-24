# Data and artifact guide

## Retained in Git

The repository retains the canonical human AdarEdit JSONL/metadata benchmark,
global pair split, supplied checkpoints, small raw Alu-pair input, authoritative
site master and compact derived feature tables, OOF predictions and summaries,
frozen ADARPro checkpoints, locked-test predictions/bootstrap tables, and every
table needed for the paper.

The canonical binary task defines edited sites as EditingIndex ≥15% and
nonedited sites as EditingIndex <1%; 1–15% observations are excluded from the
canonical task but used in prespecified secondary richer-supervision analyses.
The four biological tissues are Artery Tibial, Brain/Cerebellum, Liver, and
skeletal muscle. `Combined` is a pooled GTEx endpoint.

## External resources not stored in Git

Downloaded references and provider model files remain external because of
size, redistribution terms, or both. Their exact cached hashes are listed in
`provenance/source_workspace_inventory.csv`; resource-specific provenance is
retained in JSON/CSV files under `provenance/` and `data/derived/**/provenance`.

Principal resources include:

- GRCh38 primary assembly FASTA (Ensembl release 84), source URL and SHA-256 in
  `provenance/grch38_reference.json`;
- AdarEdit GTEx per-site EditingIndex tables, source and five SHA-256 values in
  `provenance/gtex_editing_tables.json`;
- ENCODE eCLIP and H3K36me3 datasets, accessions/checksums in
  `context/rbp/encode_eclip_manifest.csv`, `provenance/eclip_pipeline_sop.json`,
  and `provenance/encode_h3k36me3.json`;
- PrismNet resources, download record and feature provenance under
  `provenance/` and `data/derived/features/`;
- RBPNet models and generated-feature provenance in
  `data/derived/rbp/rbpnet/provenance.json`;
- RiNALMo micro/mega checkpoints, provider identities and hashes in
  `provenance/rinalmo_micro.json` and `provenance/rinalmo_mega.json`;
- k100 Umap mappability track, identity in
  `data/derived/genomic_mapping/umap100.provenance.json`;
- Reformer RNA-binding models (Shen et al. 2025, *Patterns* 6(1):101150,
  doi:10.1016/j.patter.2024.101150). Code: https://github.com/xilinshen/Reformer
  (revision used: `6f051cfbf3493a27fe5a50689f2db85e949483ab`). Weights: Hugging
  Face `XLS/Reformer` (public; per-base binding, used here; `model.bin` SHA-256
  `1a4e731f1a0d4a21d8e88df8e38820e316eae89f6c1853aa7cf99f63ff0e5c94`) and
  `XLS/Reformer-BC` (not used here). The Reformer README's install section names
  these `sxl/Reformer` and `sxl/Reformer-BC`, which are not publicly reachable;
  the `XLS` copy holds the identical `model.bin` (same SHA-256).
  Training data: 225 eCLIP-seq experiments covering 155 RNA-binding proteins
  across 3 cell lines, doi:10.5281/zenodo.14021440. See "Reformer binding
  features" below.

Place acquired files beneath `data/external/` following those provenance
records. `data/external/README.md` is the only tracked file in that cache.

## Reformer binding features

The SI Reformer screen reads two retained files. Each has a `.summary.json` with
its source, SHA-256 and the producing scripts' paths and SHA-256.

- `data/derived/features/reformer_context.parquet`: one row per canonical
  train/valid site of the four tissues (39,190 sites; key `crop_uid` =
  `pair_local_key`). 38 outcome-independent features.
- `data/derived/features/reformer_context_pca_loadings.csv`: the loadings
  behind PC1–PC5 (225 models × 5).

No test-set site is included.

To regenerate the features:

1. **Window.** For each site, take hg38 [pos−256, pos+256) around the edited A,
   reverse-complemented on the minus strand. The A sits at index 255 (+) or
   256 (−). There is one window per site.
2. **Scores.** Score every window with all 225 per-base Reformer models
   (`XLS/Reformer`; the same `model.bin` as `sxl/Reformer`).
3. **Per-model values.** `focal` is the score at the A; `max25` is the maximum
   within ±25 nt of it.
4. **Panel features.** Keep `focal` and `max25` for the 16 dsRNA-binder models
   (ILF3, STAU2, DGCR8, DROSHA, SND1, TROVE2, DHX30, DDX21, DDX24).
5. **Aggregates.** `total_binding` is the mean `max25` over all 225 models.
   `PC1`–`PC5` come from an SVD of the z-scored 225-model `max25` matrix, each
   PC oriented so its loadings sum positive.

The scoring scripts and the cluster job identities are recorded in
`reformer_context.summary.json`. The scripts were run outside this repository
and are not ported into it.

`python -m adarpro.evaluation.reformer_error_screen` checks the retained
copies:

- `import-features` and `import-loadings` check identities and shapes;
- `describe-pcs` re-derives the PC summary;
- `run` reproduces the screen in `experiments/runs/r8_reformer/`.

## Large rebuildable matrices

Four deterministic derived matrices were excluded from Git because each is
176–301 MB:

- `data/derived/features/predicted_rbp_context_long.parquet`;
- `data/derived/features/predicted_rbp_context_wide.parquet`;
- `data/derived/features/residual_model_matrix.parquet`;
- `data/derived/rbp/rbpnet/rbpnet_site_features.parquet`.

Their exact sizes and SHA-256 identities are in
`provenance/excluded_artifacts.csv`; feature-building code, compact summaries,
input identities, and downstream results are retained.

## Exploratory checkpoints and upstream archive

Exploratory per-fold `.pth` checkpoints are excluded because configurations,
seeds, selected-checkpoint identities, metrics, and predictions are retained;
the frozen final checkpoints are included. Legacy AdarEdit analyses,
non-human data, and its manuscript are not duplicated here. They remain
available from the pinned upstream revision documented in `docs/UPSTREAM.md`.

## Complete inventory

`provenance/source_workspace_inventory.csv` has one row per scientific file in
the preserved workspace: path, artifact class, epistemic status, byte size,
source SHA-256, release SHA-256, inclusion status, reason, symlink flag, and
duplicate information. `provenance/excluded_artifacts.csv` is its exact
excluded subset. This is the authoritative answer to what was and was not
published in Git.
