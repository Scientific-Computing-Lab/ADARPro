# Epistemic claim audit

Audit date: 2026-08-25 UTC

This audit separates observations, incremental prediction results,
interpretations, and future hypotheses. Quantitative provenance is recorded in
`claim_traceability.csv` and `numerical_claim_ledger.csv`.

| Manuscript claim | Epistemic class | Direct support | Wording boundary |
|---|---|---|---|
| The canonical split contains 884 graph-buildable pairs with zero overlap. | Reproduction observation | Upstream split verifier and canonical metadata | No claim beyond the reproduced benchmark. |
| Recovering additional valid high/low labels improves same-fold prediction. | Development OOF result | Existing row-wise expansion ablation | Not presented as independent held-out evidence. |
| Pair-first weighting adds value with the expanded label pool held fixed. | Development OOF result | Same-fold row-wise versus pair-first ablation | Interpreted as a statistical correction for correlated sites, not a molecular mechanism. |
| Inter-arm gap covaries with edited-site fraction across individual tissues. | Exploratory biological observation | One-row-per-pair development analysis | Does not establish that geometry causes editing. |
| Geometry plus graph summaries predicts pair-level burden beyond graph summaries. | Supported, not yet actionable | Pair-level cross-fitted models and pair-clustered intervals | Described as consistent with realization/engagement; no latent cellular state was directly measured. |
| Tested bulk RBP, transcript/repeat, and chromatin formulations do not add reproducible residual value. | Development OOF result | Same-fold residual models and negative controls | Applies only to the tested measurements and resolution; it does not deny established RBP or chromatin biology. |
| Focal-state and target/contact features localize missing information near the edited nucleotide. | Limited development evidence | Focal-node and low-capacity residual comparisons | Motivates target alignment without claiming direct measurement of ADAR chemistry. |
| The aligned local-augmented graph predictor adds target-centered information. | Selected development result | Same-fold comparison and liver replication | The second predictor remains a graph-plus-local model. |
| The 96-nt shifted control collapses despite matched capacity. | Mechanistic representation control | Existing aligned-versus-shifted OOF comparison | Supports spatial/structural target alignment, not a molecular binding mechanism. |
| Independently optimized whole-duplex-only and local-augmented predictors are complementary. | Selected development result | Fixed equal-logit comparison in muscle and liver | No claim that ensembling is universally superior. |
| The preregistered unified model did not outperform the frozen ensemble. | Preregistered development result | Immutable comparator and pair-clustered intervals | No untested explanation of its optimization behavior is asserted. |
| ADARPro improves AUROC, AUPRC, F1, and Brier score in every reported endpoint. | Canonical held-out result | Single-use final metrics and pair-clustered intervals | This is the authoritative benchmark claim. Combined is a pooled endpoint. |
| False-positive rescue exceeds true-positive damage at frozen thresholds. | Canonical held-out result | Frozen-threshold transition tables | Distinct from the matched-recall analysis. |
| ADARPro avoids false positives at matched baseline recall. | Post-test descriptive result | Final predictions rethresholded solely for recall matching | Explicitly labeled descriptive, not confirmatory or model-selecting. |
| Matched donor/cell measurements could test a two-stage editing model. | Future hypothesis | Synthesis of measurement gaps | Proposed as the next experiment, not as a conclusion from the benchmark. |

## Evidence-status vocabulary

- **Selected:** retained after controlled development comparison and used in
  the frozen predictor.
- **Supported, not yet actionable:** reproducible biological or information-
  level observation that did not strengthen the leading site classifier.
- **Limited:** small, metric-specific, context-specific, or insufficiently
  calibrated effect below the prespecified promotion criterion.
- **Not supported in the tested formulation:** no robust incremental value for
  the stated measurement and model; never generalized to biological absence.
- **Deferred / not validly tested:** benchmark-compatible measurements were
  unavailable.
- **Not pursued / superseded:** a branch was closed after stronger,
  lower-complexity evidence changed its value; this is not a negative result.
