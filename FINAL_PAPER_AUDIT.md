# Final paper audit

Audit date: 2026-08-25 UTC

## Publication identity

- **Title:** *Systematic dissection of endogenous A-to-I RNA editing reveals substrate- and target-level determinants of predictability*
- **Format:** official 2025 PNAS `pnasresearcharticle` template
- **Main manuscript:** `paper/main.tex` and `paper/main.pdf` (7 pages)
- **Supporting Information:** `paper/supplementary.tex` and `paper/supplementary.pdf` (10 pages)
- **Publication-content commit audited:** `12e01b24150944dffd9f07feac5e36ecad84d264`
- **Release commit verified from a fresh clone:** `64b0affff2fcd6459a95d2e9104744077acae8db`

The publication-content commit is recorded rather than the audit commit because
a commit cannot contain its own SHA. The repository HEAD accompanying this
file is the final audit/manifest commit.

## Scientific and structural revision

The paper is organized around a three-level account of editing predictability:
intrinsic duplex competence, substrate realization/engagement, and
target-local selectivity. Pair-first learning is presented separately as a
statistical correction for correlated observations within substrates. The
benchmark gain is the locked validation of this account, not the paper's sole
claim.

The Results now follow biological scale rather than experiment chronology.
They distinguish selected contributions, supported but not yet actionable
findings, limited findings, formulations not supported by the present
measurements, and deferred hypotheses. The SI follows the same scientific
taxonomy and preserves the full campaign, including negative controls,
stopping decisions, and the preregistered unified-model result.

A pre-existing same-fold ablation was recovered without new post-test
experimentation. It separates the benefit of adding all eligible high/low
observations from the further benefit of pair-first weighting. The manuscript
also surfaces positive pair-level evidence that genomic geometry tracks
realized editing burden while making clear that current bulk/static proxies do
not estimate realization accurately enough to improve the leading site model.

## Figure and table audit

- Figure 1 now teaches the biological hierarchy first and places the
  leakage-safe campaign underneath it.
- Figure 2 uses an evidence matrix rather than ranking heterogeneous
  comparators on one effect-size axis; quantitative panels show substrate
  realization using commensurate pair-level estimands.
- Figure 3 presents supervision decomposition, cross-endpoint pair-first
  replication, target/focal/aligned controls, global--local complementarity,
  and the exact fixed 0.5/0.5 equal-logit predictor.
- Figure 4 separates effect-size intervals, absolute F1, calibration-sensitive
  error, frozen-threshold FP rescue/TP damage, and descriptive matched-recall
  FP reduction.
- Figure S2 labels every RBP, genomic, and chromatin comparison explicitly and
  places its metric legend outside the data region.
- Ten generated SI tables replace the former sparse administrative layout.
  The complete run registry remains machine-readable in the repository.

All four main and six supplementary figures were visually inspected at full
resolution and in the compiled PDFs. Every quantitative panel is regenerated
from retained source artifacts.

## Numerical source-of-truth audit

`paper/numerical_claim_ledger.csv` contains 375 machine-generated claims with
exact values, displayed values, epistemic status, source artifact, selector,
manuscript use, and rounding rule. The same build emits 375 LaTeX macros in
`paper/generated/quantitative_claims.tex`; manuscript quantities use those
macros rather than hand-entered copies. Figure and table manifests retain
source SHA-256 values.

The principal discrepancy resolved was artery canonical ADARPro F1. The
frozen artifact is `0.886949777...`, which rounds directly to **0.8869** at four
decimal places, not `0.8870`. The manuscript, README, reports, tables, and
figures now use the documented direct-rounding rule. Macro averages,
development deltas, false-positive transitions, matched-recall counts, and
natural-prevalence quantities were regenerated from their authoritative
artifacts rather than copied from prose reports.

Development OOF and canonical held-out results remain explicitly separate.
The lower same-fold internal reference is used only for attribution and model
selection; the supplied-checkpoint canonical benchmark is the authoritative
final comparison. No development delta is added to a canonical score.

## Reference and format audit

All 35 DOI-bearing bibliography records passed static checks and live Crossref
verification on 2026-08-25; all 30 cited keys resolve. The status audit confirms
that ADAR-GPT and Yi *et al.* are peer reviewed, whereas AdarEdit, Cai *et al.*,
and Park *et al.* remain identified as preprints. The abstract contains 233
words, the Significance Statement 106 words, and the manuscript has five
keywords.

## Verification status

- canonical split: 884 pairs, 64:16:20 assignment, zero pair or cross-tissue
  leakage;
- authoritative results, freeze, quarantine, and registry checks: passed;
- Python tests: 120 passed;
- Ruff: passed;
- main/SI deterministic figure and table regeneration: passed;
- manuscript/SI LaTeX compilation: passed;
- paper artifact and claim-traceability checks: passed;
- online bibliography audit: passed;
- visual inspection of all 17 compiled pages: passed.

The pushed release commit was cloned to
`/tmp/adarpro-final-clone.2oIHTb/ADARPro`, installed through the declared
project extras, and subjected to `make verify` and `make paper-check`. The
clone reproduced 1,817 manifest entries, 128 evaluation rows, 10 figures, 10
SI tables, 375 numerical claims/macros, and the 7- and 10-page PDFs, and its
working tree remained clean after regeneration.

The PNAS class emits its known `LastPage` duplicate-label and output-routine
box warnings; no body-text overflow, undefined citation, or undefined
cross-reference remains.

## Remaining author-supplied fields

Only information that cannot be inferred safely remains marked:

- author names, order, and affiliations;
- corresponding author and e-mail;
- author-contribution statement;
- funding and acknowledgments;
- competing-interest declaration;
- journal-assigned DOI.

No scientific-result, model-selection, provenance, figure-source, or
compilation placeholder remains.
