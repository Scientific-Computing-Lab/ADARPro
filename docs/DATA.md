# Data and artifact guide

## Retained in Git

The repository retains the canonical human AdarEdit JSONL/metadata benchmark,
global pair split, supplied checkpoints, small raw Alu-pair input, authoritative
site master and compact derived feature tables, OOF predictions and summaries,
frozen ADARPro checkpoints, locked-test predictions/bootstrap tables, and every
table needed for the paper.

The canonical binary task defines edited sites as EditingIndex ≥15% and
non-edited sites as EditingIndex <1%; 1–15% observations are excluded from the
canonical task but used in prespecified secondary richer-supervision analyses.
The four biological tissues are Artery Tibial, Brain/Cerebellum, Liver, and
Muscle Skeletal. `Combined` is a pooled GTEx endpoint.

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
  `data/derived/genomic_mapping/umap100.provenance.json`.

Place acquired files beneath `data/external/` following those provenance
records. `data/external/README.md` is the only tracked file in that cache.

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
