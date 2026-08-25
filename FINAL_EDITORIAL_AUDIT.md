# Final editorial audit

Audit date: 2026-08-25 UTC

## Final publication identity

- **Title:** *Machine-learning dissection of endogenous A-to-I RNA editing
  reveals substrate- and target-level determinants of predictability*
- **Main manuscript:** `paper/main.tex` and `paper/main.pdf` (8 pages)
- **Supporting Information:** `paper/supplementary.tex` and
  `paper/supplementary.pdf` (11 pages)
- **Abstract:** 205 words; **Significance Statement:** 92 words; **keywords:** 5
- **Publication-content commit:**
  `7413212cb7cddabebfb6cb3be05abcef99ebd41e`
- **Manifest-bound clean-clone commit:**
  `de0ed65f57b8e6da1a80c655a87692bccdae6996`

The title was retained after comparison with biology-first alternatives. It is
accurate, memorable, broad enough for PNAS, and makes the AI contribution
visible without turning the manuscript into a model paper. The commit that
adds this audit necessarily follows the immutable publication-content and
manifest commits; its identifier is the repository HEAD accompanying this
file.

## Surgical language revision

The scientific structure was preserved. The Abstract and Significance
Statement were tightened around the biological question, substrate-level
editing burden, the two successful learning changes, and the one-use held-out
result. Secondary architecture details and lists of negative branches were
removed from the Abstract.

The Introduction, Results headings and transitions, and Discussion were edited
sentence by sentence to replace project-management and meta-scientific phrasing
with direct scientific claims. Repeated uses of *information dissection*,
*actionable information*, *promotion criteria*, and similar framework language
were removed. Experiment-diary transitions and duplicated caveats were also
removed. Interpretation limits now appear where they are needed rather than
after every result.

The final prose consistently distinguishes the observed substrate-level editing
burden from the inferred possibility of realization or engagement. It does not
claim to measure duplex formation, accessibility, ADAR encounter, or another
latent cellular state. Negative findings close a tested formulation and state
the better-matched measurement needed next.

## Architecture and evaluation language

Every paper-facing description gives the exact selected predictor:

1. an independently trained pair-first whole-duplex-only graph predictor;
2. an independently trained pair-first graph predictor internally augmented
   with an aligned target/helix-partner convolution, with graph and local
   embeddings fused before classification; and
3. a fixed 0.5/0.5 mean of the two logits.

No text or schematic describes the second component as a stand-alone local
CNN. No ensemble weight, stacker, or post-test parameter was fitted. The
preregistered connected model remains a development-only negative comparison.

Development OOF attribution, supplied-checkpoint canonical evaluation, frozen
validation-selected thresholds, and post-test descriptive matched-recall
rethresholding remain explicitly separated. Combined is described as a pooled
benchmark endpoint, never as a fifth tissue. No development effect was added
to a canonical score.

## Figures and tables

- **Figure 1:** retained the biologically correct convergent topology in which
  intrinsic duplex properties and cell/transcript state affect substrate
  availability before target-local selectivity. Text was shortened, the
  indirect status of the editing-burden/geometry readout was preserved, and
  spacing and arrow logic were rebalanced.
- **Figure 2A:** replaced the dense repeated-scale table with grouped sections,
  three short columns, larger type, alternating row shading, and symbols plus
  words for status. Interpretations are now brief labels; exact comparators,
  effects, and intervals remain in Table S2. Panels B and C continue to show
  the positive substrate-burden result.
- **Figure 3:** retained the supervision decomposition, replication, focal-to-
  aligned progression, shifted-window control, complementarity analysis, and
  exact selected architecture. Caption quantities now use generated macros.
- **Figure 4:** retained frozen confirmatory results separately from the panel
  labeled post-test descriptive matched-recall analysis.
- **Table S2:** retained all 18 meaningful branches, enlarged the type and row
  spacing, shortened interpretations, and kept every comparator explicit so
  heterogeneous estimands cannot be mistaken for a common ranking.
- **Table S10:** uses scientific disposition and stopping-criterion language;
  historical ranks and internal campaign-management terms remain only in
  repository provenance.

All ten figures remain available as vector PDFs and SVGs where applicable. All
quantitative plots and tables are regenerated from retained source artifacts.

## Evidence-status audit

Statuses were checked against the recorded predeclared decision rules rather
than reassigned for narrative convenience. In particular:

- pair geometry/editing burden remains **supported, not yet actionable**;
- multitask supervision, focal-node retention, shared-tissue learning, the
  engagement gate, and bulk H3K36me3 remain **limited**;
- the all-A decoder, larger/deeper graph capacity, and unified model remain
  **not supported in the tested formulation**; and
- matched donor/cell measurements, in vivo structure/isoform state, and
  benchmark-safe task-specific pretraining remain **deferred**.

Frozen RiNALMo remains **not supported**, not *limited*. The 150M confirmation
showed a small F1-only threshold effect without AUROC/AUPRC improvement and did
not satisfy the predeclared requirement to beat matched random features on both
ranking and F1. This reason is explicit in the Results, SI, Figure 2, Table S2,
and `paper/EPISTEMIC_CLAIM_AUDIT.md`.

## Terminology and numerical consistency

Paper-facing terminology is standardized to *expanded row-wise sampling*,
*pair-first weighting*, *pair-clustered bootstrap interval*, *target/helix
partner*, *nonedited*, and *skeletal muscle*. *Intrinsic competence* is avoided
in prose where it could sound like a measured phenotype; *intrinsic substrate
suitability* or *editability* is used instead. Deferred hypotheses are distinct
from branches that were not pursued because a simpler validated direction
superseded them.

The numerical ledger was regenerated without manual transcription. It contains
441 claims and emits 375 LaTeX macros. The shifted-window distance and fixed
ensemble weights in the Figure 3 caption now also resolve through generated
macros. The previously resolved artery F1 rounding remains 0.8869 from the
frozen artifact. No new numerical discrepancy was found; endpoint means, Brier
changes, false-positive transitions, natural-prevalence results, and unified-
model effects all trace to their authoritative tables.

## Bibliography and typography

All 35 DOI-bearing records pass the static and live Crossref audit; all 30 cited
keys resolve. Titles, years, journals, volumes, pages or article numbers, DOI
values, and publication status were checked against the audit record and
Crossref. AdarEdit, Cai *et al.*, and Park *et al.* remain labeled as 2026
preprints; ADAR-GPT, Yi *et al.*, and RiNALMo are labeled as journal articles.
The Bass title was protected so that `RNA` renders correctly. The compiled PDFs
show `Cell Reports` and `Cell Research` correctly; apparent split strings were
PDF text-extraction artifacts, not bibliography errors.

Every main and SI page was inspected as a rendered image. The PNAS first-page
transition uses the template-prescribed `\Parasplit`, and the automated
bounding-box check confirms that no main-paper text crosses a physical page
edge. No manuscript-body overfull box, unresolved citation, broken reference,
table overflow, or unreadable supplementary figure remains. The PNAS class
continues to emit its known title-block box and duplicate `LastPage` warnings.

## Verification

The manifest-bound revision was cloned into a new temporary directory and
verified independently:

- canonical split: 884 substrates, zero within-split or cross-tissue leakage;
- authoritative results, frozen manifest, quarantine, and registry: passed;
- Python tests: 120 passed (one third-party deprecation warning);
- Ruff: passed;
- curated release manifest: 1,821 files verified;
- figure, claim-ledger, and table regeneration: passed;
- static and live Crossref reference audits: passed;
- PNAS field limits, traceability hashes, deterministic compilation, and
  main-PDF page bounds: passed;
- clean-clone output: 8-page main paper and 11-page SI; worktree remained clean.

## Remaining author-supplied fields

- author names, order, affiliations, and lead-author designation;
- corresponding author and e-mail;
- author-contribution statement;
- funding and acknowledgments;
- competing-interest declaration; and
- journal-assigned DOI.

No scientific-result, model-selection, figure-source, reference, provenance,
or compilation placeholder remains.
