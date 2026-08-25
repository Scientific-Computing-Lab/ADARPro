# Final manuscript audit

> Historical pre-editorial audit. The submission-ready record superseding this
> snapshot is `FINAL_EDITORIAL_AUDIT.md`.

Audit date: 2026-08-25 UTC

## Publication identity

- **Title:** *Machine-learning dissection of endogenous A-to-I RNA editing
  reveals substrate- and target-level determinants of predictability*
- **Format:** official 2025 PNAS `pnasresearcharticle` template
- **Main manuscript:** `paper/main.tex` and `paper/main.pdf` (8 pages)
- **Supporting Information:** `paper/supplementary.tex` and
  `paper/supplementary.pdf` (11 pages)
- **Numerical source of truth:** `paper/numerical_claim_ledger.csv` (441 claims)
- **Publication-content commit:** `8b6dafbf5660dc2a1c77fd9fcaa0e9467d209b5a`

The publication-content commit is recorded rather than the audit commit because
a commit cannot contain its own identifier. The repository HEAD accompanying
this file is the final audit/manifest commit.

## Scientific revision

The manuscript now treats machine learning as a controlled instrument for
locating recoverable information rather than as an architecture contest. The
Results follow a biological and statistical hierarchy:

1. whole-duplex sequence and predicted structure encode intrinsic competence;
2. pair geometry identifies reproducible substrate-level variation in editing
   burden, consistent with a realization or engagement dimension that was not
   directly measured;
3. target/helix-partner alignment recovers local selectivity diluted by global
   pooling; and
4. the Alu pair is the effective unit for learning from correlated site rows.

The same-fold supervision ablation separates the gain from recovering
additional valid high/low labels from the gain due to pair-first weighting.
Biological-context results are stated at the resolution actually tested:
cell-line occupancy, tissue-average abundance, binary bulk chromatin overlap,
and broad static annotations did not add reproducible residual value. These
findings constrain the measurement problem without denying established RBP,
chromatin, transcript-processing, or cellular-state biology.

The Discussion distinguishes direct observations, development prediction
results, mechanistic interpretations, and future hypotheses. It ends with a
specific matched donor/cell experiment measuring editing, transcript state,
ADAR isoform/activity, regulator occupancy, chromatin, and in vivo RNA
conformation on the same substrates.

## Architecture-description correction

Every paper-facing description now gives the exact selected predictor:

- component 1 is an independently trained pair-first whole-duplex-only graph
  predictor;
- component 2 is an independently trained pair-first graph predictor internally
  augmented with and fused to an aligned target/helix-partner CNN; and
- the two logits are combined by a fixed 0.5/0.5 mean.

The manuscript and schematic no longer imply that component 2 is a standalone
local CNN. No weight search, stacker, or post-test fitting was used. The
preregistered unified end-to-end model remains a development-only negative
result and is not depicted as the selected architecture.

## Evidence taxonomy and terminology

The paper uses five scientific outcomes consistently: selected; supported but
not yet actionable; limited; not supported in the tested formulation; and
deferred/not validly tested. A sixth label, not pursued/superseded, distinguishes
branches closed after stronger lower-complexity evidence from hypotheses that
lacked a valid measurement.

Paper-facing terminology was standardized to `expanded row-wise sampling`,
`pair-clustered bootstrap interval`, `target/helix-partner`, `nonedited`, and
`skeletal muscle`. `Combined` is always identified as a pooled benchmark
endpoint rather than a fifth tissue. Matched-recall false-positive reduction is
explicitly labeled post-test descriptive; primary metrics use thresholds frozen
before test access. Machine-readable upstream identifiers remain unchanged for
provenance.

The detailed terminology and claim boundaries are recorded in
`paper/TERMINOLOGY_AUDIT.md` and `paper/EPISTEMIC_CLAIM_AUDIT.md`.

## Figures and Supporting Information

- Figure 1 is anchored on the biological hierarchy: intrinsic duplex properties
  and cell/transcript state converge on a latent realized substrate, followed by
  target-local selectivity and observed editing. The AI hypothesis map and
  leakage-safe funnel are visually secondary.
- Figure 2 uses a structured evidence matrix, rather than placing heterogeneous
  estimands on one axis, and surfaces the positive pair-geometry/editing-burden
  result.
- Figure 3 separates added labels from corrected pair weighting, presents the
  focal-to-aligned representation progression and 96-nt shifted control, and
  shows the exact two-component selected predictor.
- Figure 4 distinguishes the single-use canonical held-out comparison from the
  post-test descriptive matched-recall analysis and displays absolute F1,
  clustered improvement intervals, Brier reduction, and FP rescue versus TP
  damage.
- Figure S2 identifies every RBP, genomic, random-feature, and chromatin row.
  All main and supplementary figures use consistent, non-color-only status
  encodings and were inspected both at full resolution and in the compiled PDFs.
- Table S2 is a generated synthesis of 18 meaningful branches. Every row gives
  its own comparator, primary pair-clustered effect, secondary behavior,
  evidence status, and interpretation; heterogeneous estimands are not ranked
  against one another.

All quantitative figures and tables are generated from retained source
artifacts. The main paper contains four figures; the SI contains six figures
and ten generated tables.

## Numerical source-of-truth audit

`paper/numerical_claim_ledger.csv` contains 441 machine-generated rows with
exact values, displayed values, epistemic status, source artifact, selector,
manuscript use, and rounding rule. The same build emits 375 LaTeX macros in
`paper/generated/quantitative_claims.tex`. Figure and table manifests retain
source SHA-256 values.

The principal discrepancy resolved was artery canonical ADARPro F1. The frozen
artifact is `0.886949777...`, which rounds directly to **0.8869** at four
decimal places, not `0.8870`. Mean endpoint improvements, development deltas,
false-positive transitions, Brier differences, matched-recall counts,
natural-prevalence quantities, and unified-model effects were regenerated from
their authoritative artifacts rather than copied from prose reports.

Development OOF and canonical held-out results remain separate. The same-fold
internal reference is used only for attribution and model selection; the
supplied-checkpoint canonical benchmark is the authoritative final comparison.
No development delta is added to a canonical score.

## Reference audit

All 35 DOI-bearing bibliography records pass static and live Crossref checks;
all 30 cited keys resolve. A targeted status check on the 2026 references
confirmed that ADAR-GPT and Yi *et al.* are journal articles, whereas AdarEdit,
Cai *et al.*, and Park *et al.* remain preprints as of the audit date.
Bibliographic capitalization preserves RNA, ADAR, Alu, and model names.

## Simulated review passes

- **PNAS editor:** page 1 now states a broad biological question, an explicit
  AI-for-biology strategy, the information hierarchy, and the held-out result.
- **ADAR biologist:** engagement is never described as directly measured;
  negative context results are restricted to their bulk/static operationalization;
  Alu and predicted-structure scope is explicit.
- **Machine-learning reviewer:** same-fold attribution, pair grouping, label
  expansion versus weighting, matched-capacity shift, exact ensemble, unified
  comparator, and lack of weight search are all explicit.
- **Statistical reviewer:** pair IDs define folds and bootstrap clusters;
  Combined overlap and descriptive endpoint averaging are disclosed; threshold
  selection and post-test matched-recall status are unambiguous.
- **Figure editor:** every panel has one argument, vector artwork, consistent
  labels, readable intervals, and programmatic source tables; all pages were
  inspected after compilation.

## Verification status

- Abstract: 230 words; Significance Statement: 92 words; five keywords.
- Canonical split: 884 pairs with zero within-split or cross-tissue pair leakage.
- Main manuscript: 8 pages; Supporting Information: 11 pages.
- Canonical split, authoritative-result, freeze, quarantine, and registry checks:
  passed.
- Python tests: 120 passed; Ruff: passed.
- Deterministic figure/table generation, PNAS field limits, quantitative-source
  hashes, claim traceability, and manuscript compilation: passed.
- Static and live Crossref reference audit: passed for 35 records and 30 cited
  keys.
- PDF structural checks: passed for the main manuscript and SI. The PNAS
  first-page mode is explicitly terminated with the template-prescribed
  `\Parasplit` transition; a rendered bounding-box check confirms that no text
  falls beyond any edge of the 585-point-wide main-paper pages.
- Release manifest: 1,821 curated files verified.
- Fresh-clone reproduction: passed at publication-content commit
  `8b6dafbf5660dc2a1c77fd9fcaa0e9467d209b5a`. The remote clone passed
  `make verify` and `make paper-check`, regenerated the 8-page manuscript and
  11-page SI, passed the rendered page-boundary check, and remained clean.

The PNAS class emits its known `LastPage` duplicate-label and first-page
title-block box warnings. No manuscript-body overflow, unresolved citation, or
unresolved cross-reference is present.

## Remaining author-supplied fields

- author names, order, and affiliations;
- corresponding author and e-mail;
- author-contribution statement;
- funding and acknowledgments;
- competing-interest declaration; and
- journal-assigned DOI.

No scientific-result, model-selection, provenance, figure-source, or
compilation placeholder remains.
