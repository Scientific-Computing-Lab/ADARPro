# Epistemic claim audit

Audit date: 2026-09-10 UTC

## Independent source-code corrections (supersede earlier classifications)

The independent audit in `reports/29_independent_submission_audit.md` found
four differences between the earlier narrative and implementation. Pair-first
learning balances classes and samples pairs uniformly **within class**, then
sites; it is not uniform-pair/class-fallback sampling. The burden gate used
expanded row-wise supervision, whereas the later neural burden auxiliary used
pair-first supervision. Early RBP, genomic, chromatin, native-folding,
target-geometry, and initial substrate-profile screens reuse pooled first-stage
OOF logits and are exploratory, not fully nested. Finally, the supplied
Bio-aware comparison changes features and architecture together and cannot
isolate graph capacity. Its result is descriptive; graph-transformer expansion
was not pursued. The table rows below incorporate these corrected boundaries. No numerical result or historical decision was changed.

Structural-stratum gains are not confined to paired targets or one stem stratum;
they do not identify or exclude the model's use of any single structural feature.
The shifted-window control is muscle-only; liver repeats the local-versus-global
comparison. Brier improvements establish probability accuracy, not calibration
alone. Bootstrap intervals condition on retained fits and omit retraining and
adaptive-selection variation. The search taxonomy organizes an adaptive program;
only specific recorded branches had prospective decision rules.

This audit separates observations, incremental prediction results,
interpretations, and future hypotheses. Quantitative provenance is recorded in
`claim_traceability.csv` and `numerical_claim_ledger.csv`.

| Manuscript claim | Epistemic class | Direct support | Wording boundary |
|---|---|---|---|
| Prior predictors already recover editing-associated sequence and structural signal and examine biological features. | Prior-work scope | DeepRed, EditPredict, ADAR-GPT, AdarEdit, and the primary biochemical/perturbation literature; source audit in `reports/31_motivation_and_prior_work_revision.md` | Do not manufacture novelty by describing previous work as accuracy-only or claiming the first local window, graph representation, ablation, or predictive information probe. Published scores use different tasks and are not a common ranking. |
| The present contribution is the incremental comparison of substrate, target, and context information beyond a common endogenous reference, separating added measurements from representation and learning changes. | Study-design synthesis | Scale-organized coverage table, pair-level geometry comparison, target-centered and shifted controls, contextual screens, and statistical/model-sufficiency comparisons | Not a complete partition of biological variance or proof that all remaining errors require cellular measurements. The reference and pair-disjoint benchmark are inherited; the final test validates only the selected synthesis. |
| The study systematically traverses the major biologically defined information scales accessible to the benchmark. | Study-design synthesis | Search-space coverage table, experiment taxonomy, campaign index, and retained positive, limited, unsupported, superseded, and deferred branches | ``Accessible'' is essential: the claim does not include determinants for which no compatible measurement exists. |
| The secondary modeling search covers richer supervision, cross-context or multitask learning, more sophisticated representations, and greater capacity or integration. | Study-design synthesis | Search-space coverage table and registered campaign branches | Completed negative or limited experiments remain distinct from deferred and superseded directions. |
| Biologically and statistically targeted changes outperformed generic increases in modeling sophistication under the tested regime. | Development evidence synthesis | Pair-first and aligned-local selected comparisons; multitask, shared-tissue, foundation, decoder, and unified comparisons | Conditional on the tested data, representations, effective sample size, and controls; not a universal claim about future AI architectures. |
| The canonical split contains 884 graph-buildable pairs with zero overlap. | Reproduction observation | Upstream split verifier and canonical metadata | No claim beyond the reproduced benchmark. |
| Recovering additional valid high/low labels improves same-fold prediction. | Development OOF result | Existing row-wise expansion ablation | Not presented as independent held-out evidence. |
| Pair-first weighting adds value with the expanded label pool held fixed. | Development OOF result | Same-fold row-wise versus pair-first ablation | Interpreted as a statistical correction for correlated sites, not a molecular mechanism. |
| Inter-arm gap covaries with edited-site fraction across individual tissues. | Exploratory biological observation | One-row-per-pair development analysis | Does not establish that geometry causes editing. |
| Geometry plus graph summaries predicts pair-level burden beyond graph summaries. | Supported, not yet actionable | Pair-level cross-fitted models and pair-clustered intervals | Neither realization nor engagement was measured; the demonstrated outcome is editing burden. |
| Tested bulk RBP, transcript/repeat, and chromatin formulations lack consistent residual increments. | Exploratory pooled-score screens | Pair-disjoint second-stage models and controls; first-stage fits are not fully outer-isolated | Requires fully nested replication before independent measurement-limit claims; does not deny established RBP or chromatin biology. |
| Focal-state and target/contact features localize missing information near the edited nucleotide. | Limited development evidence | Focal-node and low-capacity residual comparisons | Motivates target alignment without claiming direct measurement of ADAR chemistry. |
| The aligned local-augmented graph predictor adds target-centered information organized by predicted pairing. | Selected development result | Same-fold comparison and liver replication | The second predictor remains a graph-plus-local model; the experiment does not isolate the predicted partner from the wider paired neighborhood. |
| The local input consists of target- and partner-centered lanes with nucleotide identity and pairing state. | Implementation fact, rechecked during the reader walkthrough | `local_window` and `encode` in `src/adarpro/models/multitask_oof.py`, reused by the final training/prediction pipeline | Twelve input channels; alignment is defined by window indexing. Earlier prose incorrectly listed separate focal/partner and signed-position channels. This is a description correction, not a model or result change. |
| The 96-nt shifted control collapses despite matched capacity. | Representation control | Existing aligned-versus-shifted OOF comparison | Supports correct target centering of the paired local representation, not a partner-specific or molecular binding mechanism. |
| Matched genomic sites vary across individual tissues, including strict edited/nonedited switches. | Post-hoc descriptive development observation | Matched continuous EditingIndex values from development pairs; whole-pair bootstrap | Demonstrates tissue-associated variation while holding site identity fixed; tissue identity is not a mechanism. |
| Pair-level editing rankings persist across tissues on identical site sets. | Post-hoc descriptive development observation | Pair summaries restricted to sites shared by each tissue pair; whole-pair bootstrap | Consistent with stable substrate-level propensity overlaid by tissue shifts; does not establish an invariant molecular state. |
| Independently optimized whole-duplex-only and local-augmented predictors are complementary. | Selected development result | Fixed equal-logit comparison in muscle and liver | No claim that ensembling is universally superior. |
| The fixed ensemble improves equal-pair within-duplex discrimination in muscle and liver. | Post-hoc development OOF diagnostic | Mean pair-level AUROC among mixed-label pairs; whole-pair bootstrap | Supports ensemble-level within-substrate ranking. The aligned-local component alone has a positive interval only in muscle, so no replicated component-specific claim is made. |
| Pair-first gains are largest among substrates with fewer eligible sites. | Post-hoc development OOF diagnostic | Per-pair log-loss contrasts by raw-site-count quintile | Supports redistribution of learning influence; quintiles are descriptive and were not a selection criterion. |
| Local and ensemble Brier gains occur across coarse target-structure strata. | Post-hoc development OOF diagnostic | Paired/unpaired and stem-extent means | Rules out a trivial single-stratum explanation descriptively; no subgroup-specific confirmatory claim is made. |
| The preregistered unified model did not outperform the frozen ensemble. | Preregistered development result | Immutable comparator and pair-clustered intervals | No untested explanation of its optimization behavior is asserted. |
| The selected predictor improves AUROC, AUPRC, F1, and Brier score in every reported endpoint. | Canonical held-out result | Frozen final metrics and pair-clustered intervals | This confirms the composite predictor on unseen pairs within the benchmark. It does not independently confirm every development-stage component attribution. Combined is pooled. |
| False-positive rescue exceeds true-positive damage at frozen thresholds. | Canonical held-out result | Frozen-threshold transition tables | Distinct from the matched-recall analysis. |
| The selected predictor avoids false positives at matched baseline recall. | Post-test descriptive result | Final predictions rethresholded solely for recall matching | Explicitly labeled descriptive, not confirmatory or model-selecting. |
| Matched donor/cell measurements could test a two-stage editing model. | Future hypothesis | Synthesis of measurement gaps | Proposed as the next experiment, not as a conclusion from the benchmark. |
| Transportability beyond the benchmark remains untested. | Evidence gap | The upstream RNAAtlas analysis assesses label concordance, not predictions from the selected framework | The canonical held-out split establishes generalization to unseen pairs under the same benchmark construction, not external-cohort transportability. |

## Evidence-status vocabulary

- **Selected:** retained after controlled development comparison and used in
  the frozen predictor.
- **Supported, not yet actionable:** reproducible biological or information-
  level observation that did not strengthen the leading site classifier.
- **Limited:** small, metric-specific, context-specific, or insufficient for
  the recorded selection criterion. Criteria were predeclared only where
  documented in the corresponding protocol; Brier error is not calibration alone.
- **Not supported in the tested formulation:** no robust incremental value for
  the stated measurement and model; never generalized to biological absence.
- **Deferred / not validly tested:** benchmark-compatible measurements were
  unavailable.
- **Not pursued / superseded:** a branch was closed after stronger,
  lower-complexity evidence changed its value; this is not a negative result.

## Evidence-status decision audit

The paper-facing status assignments were checked against the original decision
records rather than reassigned from the final narrative.

| Branch | Paper status | Historical basis |
|---|---|---|
| Pair geometry and editing burden | Supported, not yet actionable | Pair-level effects replicated across four tissues, but site-level gates did not improve F1 reliably and worsened Brier score. |
| Continuous/ordinal supervision | Limited | AUROC and Brier improved modestly; F1 was negligible and natural-prevalence integration was adverse. |
| Focal-node retention | Limited | AUROC/AUPRC improved beyond width controls; standalone F1 interval crossed zero. |
| Cross-tissue sharing | Limited | Ranking and F1 effects were positive but below both predeclared selection thresholds; mean Brier worsened. |
| Hierarchical engagement gate | Limited | AUPRC improved, but F1 crossed zero and Brier worsened; direct fusion stopped under the recorded rule. |
| Bulk H3K36me3 | Limited exploratory screen | Muscle threshold effects and liver AUPRC gains were metric-specific; pooled first-stage scores additionally limit interpretation. |
| Frozen RiNALMo | Not supported in the tested formulation | The fixed 150M confirmation reproduced a small F1-only threshold effect but no AUROC/AUPRC gain and failed the predeclared requirement to beat matched random features on both ranking and F1. |
| Substrate-wide all-A decoder | Not supported in the tested formulation | AUROC, AUPRC, and F1 were adverse in both completed folds, meeting the prespecified futility criterion. |
| Supplied Bio-aware variant | Descriptive, mixed | Features and architecture change together; this is not an isolated capacity experiment. Graph Transformer/GATv2 was not pursued. |
| Unified end-to-end architecture | Not supported in the tested formulation | The preregistered model was worse in AUROC/AUPRC and Brier than the immutable ensemble, so later stages remained closed. |
