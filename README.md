# ADARPro

ADARPro is a leakage-controlled study of **where predictive information for
endogenous A-to-I RNA editing resides**. Starting from the published AdarEdit
human inverted-Alu benchmark, the project tested biological context,
supervision, representation, transfer, pretraining, and integration
hypotheses under pair-disjoint development evaluation. The study retains both
successful and negative experiments.

The evidence supports a hierarchical picture. Whole-duplex sequence and
structure encode intrinsic substrate suitability. Genomic geometry and pair-level models
identify reproducible variation in substrate editing burden, consistent with a
realization or engagement dimension, although current bulk context proxies do
not resolve the underlying cellular state well enough to improve the leading
site classifier. Pair-first learning respects the substrate as the statistical
unit, while a target/helix-partner-aligned representation recovers
nucleotide-level information diluted by whole-duplex pooling.

The frozen predictor is an unchanged 50:50 equal-logit ensemble of two
independently trained models: a pair-first whole-duplex-only graph model and a
second pair-first graph model internally augmented with a target/helix-partner
local encoder. A separately preregistered unified end-to-end architecture was
developed before test access and rejected on development data alone.

## Locked canonical result

The canonical test was evaluated once after the complete pipeline was frozen.
The table below compares the reproduced supplied-checkpoint AdarEdit baseline
with the frozen ADARPro ensemble on the original published task and split.

| Context | AdarEdit AUROC | ADARPro AUROC | AdarEdit AUPRC | ADARPro AUPRC | AdarEdit F1 | ADARPro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Artery | 0.9238 | 0.9477 | 0.9262 | 0.9485 | 0.8671 | 0.8869 |
| Brain | 0.9301 | 0.9508 | 0.9286 | 0.9504 | 0.8689 | 0.8902 |
| Liver | 0.9092 | 0.9414 | 0.9242 | 0.9500 | 0.8541 | 0.8812 |
| Skeletal muscle | 0.8688 | 0.9144 | 0.8757 | 0.9207 | 0.8139 | 0.8541 |
| Combined | 0.9331 | 0.9491 | 0.9356 | 0.9513 | 0.8674 | 0.8852 |

Mean gains across the five reported benchmark endpoints were **+0.0277 AUROC,
+0.0261 AUPRC, and +0.0253 F1**.
Every endpoint had a positive pair-clustered 95% interval for all three
differences. In a post-test descriptive analysis at matched
reproduced-baseline recall, ADARPro avoided 126, 119, 158, 68, and 148 false
positives, respectively, with positive clustered intervals throughout.

These held-out results are the authoritative performance claim. The lower
absolute grouped out-of-fold scores in development are an internal research
instrument used to compare candidates under identical folds; they do
not replace or redefine the published benchmark.

## What the campaign established

- Pair-first expanded high/low supervision improved all five development
  endpoints under the same-fold internal comparison.
- Recovering all valid high/low labels and correcting pair weighting each made
  an independent contribution in the existing same-fold ablation.
- Inter-arm geometry and low-capacity pair models predicted substrate editing
  burden, but available static/bulk context did not resolve the latent cellular
  state well enough to strengthen site classification.
- A graph model augmented with an aligned target/helix-partner local view added
  complementary information. An otherwise identical model centered 96 nt away
  failed decisively, ruling out generic additional CNN capacity as the
  explanation.
- Fixed prediction-level integration outperformed the preregistered unified
  joint model at the available effective pair-level sample size.
- Tested eCLIP/PrismNet/RBPNet × bulk GTEx representations, current bulk
  H3K36me3 overlap, broad site-level annotations, frozen RNA foundation-model
  embeddings, generic extra GNN capacity, and several richer-task formulations
  did not provide sufficiently robust incremental information for promotion.
  These are statements about the tested measurements and models, not claims
  that the corresponding biology is absent.

The full decision trail is indexed in
[`docs/CAMPAIGN_INDEX.md`](docs/CAMPAIGN_INDEX.md). The scientific synthesis is
in [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md).

## Manuscript

The PNAS Research Report manuscript is available as
[`paper/main.pdf`](paper/main.pdf), with the complete campaign, controls, and
fold-level evidence organized in
[`paper/supplementary.pdf`](paper/supplementary.pdf). Their LaTeX sources,
official-template provenance, verified BibTeX, claim-to-artifact audit, and
programmatically generated figure and table sources are retained in
[`paper/`](paper/). Author, affiliation, contribution, funding, and
correspondence fields remain explicit placeholders pending author approval.

```bash
make paper-check
```

This regenerates every paper figure and SI table, compiles both PDFs, verifies
the bibliography and PNAS field limits, and checks quantitative and claim
traceability.

## Repository map

| Path | Contents |
|---|---|
| `src/adarpro/`, `code/` | ADARPro analysis/model code and pinned AdarEdit model code |
| `experiments/` | Append-only experiment registry, configurations, and retained run artifacts |
| `models/` | Frozen selected checkpoints, summaries, and development model artifacts |
| `results/` | Reproduced AdarEdit tables and the one-use locked-test result |
| `data/` | Canonical benchmark inputs and retained small derived tables |
| `figures/` | Reproducible figures and quantitative source tables |
| `reports/` | Chronological campaign reports 00–20 and final synthesis |
| `provenance/` | Input hashes, freeze manifests, audit chain, inventory, and exclusions |
| `paper/` | PNAS manuscript, SI, verified bibliography, and paper source tables |
| `docs/` | Campaign, data, reproducibility, and upstream documentation |

## Reproduce and verify

The exact checkpoint-compatible environment is defined in `environment.yml`;
the lightweight package/test dependencies are in `pyproject.toml`.

```bash
conda env create -f environment.yml
conda activate adaredit
python -m pip install -e '.[dev,tables]'
make verify
```

`make verify` checks the global pair split, result identities, registry
integrity, tests, lint, and the curated release manifest. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the complete paths from
retained artifacts to result tables, figures, and paper.

## Data and provenance

Canonical human benchmark files and the supplied AdarEdit checkpoints are
retained. External resources and four large rebuildable feature matrices are
not stored in Git; their source identities, SHA-256 hashes, roles, and
exclusion reasons are recorded in
[`provenance/source_workspace_inventory.csv`](provenance/source_workspace_inventory.csv),
[`provenance/excluded_artifacts.csv`](provenance/excluded_artifacts.csv), and
[`docs/DATA.md`](docs/DATA.md). No negative or terminated result was omitted
because it was unfavorable.

## License

The repository retains the GNU General Public License v3 supplied with the
AdarEdit codebase; see [`LICENSE`](LICENSE). External datasets and pretrained
weights remain governed by their original providers' terms.
