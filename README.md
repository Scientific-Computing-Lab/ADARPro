# Evidence-guided search maps predictive information across scales of human RNA editing

This repository contains a study of **where predictive
information for endogenous A-to-I RNA editing resides**. Starting from the
published AdarEdit human inverted-Alu benchmark, the project traverses three
biological information scales, a separate within-substrate dependence problem,
and four classes of model-sufficiency tests under pair-disjoint development
evaluation. The study retains supported, limited, unsupported, superseded, and
data-limited branches.

Whole-duplex sequence and predicted structure provide a strong reference.
Genomic organization adds between-substrate information about editing burden;
correctly centered local representations improve use of the target neighborhood.
Class-balanced pair-first learning reduces the influence of site-rich substrates
within each class. Current context proxies have not supplied a consistent
site-level correction in the retained comparisons. Neither substrate realization
nor an underlying cellular state was measured directly.

The selected predictor combines, for each endpoint, a pair-first whole-duplex
graph model and an independently trained pair-first local predictor (five seeds
each, averaged in log-odds) in a logistic second stage that also reads the
genomic spacing between the two arms. Its comparator is the equal-weight
average. A unified end-to-end architecture, run under preregistered decision
rules, did not surpass the two-model combination on development data.

## Locked canonical result

The complete selected predictor was frozen before the canonical held-out
evaluation, and the selected predictor and the equal-weight average were
scored once, in a single session. The table below compares the reproduced
supplied-checkpoint AdarEdit baseline with the frozen selected predictor on
the original published task and split.

| Context | AdarEdit AUROC | Selected predictor AUROC | AdarEdit AUPRC | Selected predictor AUPRC | AdarEdit F1 | Selected predictor F1 |
|---|---:|---:|---:|---:|---:|---:|
| Artery | 0.9238 | 0.9592 | 0.9262 | 0.9645 | 0.8671 | 0.8986 |
| Brain | 0.9301 | 0.9642 | 0.9286 | 0.9688 | 0.8689 | 0.9032 |
| Liver | 0.9092 | 0.9590 | 0.9242 | 0.9681 | 0.8541 | 0.9045 |
| Skeletal muscle | 0.8688 | 0.9411 | 0.8757 | 0.9513 | 0.8139 | 0.8756 |
| Combined | 0.9331 | 0.9639 | 0.9356 | 0.9687 | 0.8674 | 0.9051 |

Mean gains across the five reported benchmark endpoints were **+4.45 AUROC, +4.62 AUPRC, and +4.31 F1 percentage points**.
Every endpoint had a positive pair-clustered 95% interval for all three differences.
In a post-test descriptive analysis at matched reproduced-baseline recall, the selected predictor avoided 194, 188, 235, 105, and 223 false positives, respectively, with clustered intervals excluding zero throughout.

These held-out results are the authoritative performance claim. The lower
absolute grouped out-of-fold scores in development are an internal research
instrument used to compare candidates under identical folds; they do
not replace or redefine the published benchmark.

## What the campaign established

- Pair-first expanded high/low supervision improved all five development
  endpoints under the same-fold internal comparison.
- Recovering all valid high/low labels and correcting pair weighting made
  separable contributions in the muscle same-fold ablation.
- Inter-arm geometry predicted substrate editing burden. Added through a
  predicted burden (tested in skeletal muscle) it gave no reliable site-level
  gain; the measured spacing between the paired Alus, added to the site
  predictor's second stage, improved site ranking over a shuffled-spacing
  control during development.
- A local predictor (a CNN reading aligned target/helix-partner windows) added
  information beyond the whole-duplex graph model. A matched-capacity control
  centered 96 nt away, tested in muscle, performed worse, supporting correct
  centering; it does not isolate the predicted partner or establish a mechanism.
- A unified end-to-end model did not surpass the two-model combination on
  development data.
- Tested eCLIP/PrismNet/RBPNet × bulk GTEx representations, current bulk
  H3K36me3 overlap, broad site-level annotations, frozen RNA foundation-model
  embeddings, and several richer-task formulations
  did not provide sufficiently robust incremental information for promotion.
  These are statements about the tested measurements and models, not claims
  that the corresponding biology is absent.

The independent source-code audit found that early residual screens reused
pooled first-stage OOF scores in pair-disjoint second-stage fits. They are
exploratory, not fully nested evaluations. The supplied Bio-aware comparison also
changes features and architecture together and cannot isolate capacity. These
limits, corrected sampler/comparator descriptions, and the unaffected controlled
neural and final-test evidence are documented in
[`reports/29_independent_submission_audit.md`](reports/29_independent_submission_audit.md).

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
The [independent submission audit](reports/29_independent_submission_audit.md)
also identifies the residual-screen evidence gap and the unresolved PNAS
AI-assisted-graphics policy question. Biological illustrations now use
[credited DBCLS/TogoTV and Servier SVG artwork](paper/illustration_sources/README.md).
Historical GPT Image assets remain excluded from the active build;
see [archive provenance](paper/figure_assets/README.md).
The [licensed-illustration revision](reports/30_licensed_illustration_revision.md)
records source selection, figure changes, and verification.
These PDFs are revised author-review drafts, not certified submission-ready files.
The [sequential reader-comprehension walkthrough](reports/32_reader_comprehension_walkthrough.md)
records where an adjacent-field reader could lose the argument, the resulting
revisions, and the second-pass checks. It is an editorial simulation, not a
human-participant comprehension study.
The [final scientific-writing pass](reports/33_final_editorial_pass.md) and
[multilingual semantic audit](reports/34_multilingual_semantic_audit.md) record
the implemented readability revisions, Hebrew/German/Japanese/Spanish meaning
checks, quantitative preservation, and remaining author-completion items.

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
| `results/` | Reproduced AdarEdit tables and the locked-test results |
| `data/` | Canonical benchmark inputs and retained small derived tables |
| `figures/` | Reproducible figures and quantitative source tables |
| `reports/` | Campaign/editorial reports 00–34 and final synthesis |
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
