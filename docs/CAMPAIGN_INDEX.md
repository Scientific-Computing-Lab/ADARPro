# ADARPro campaign index

This index maps the full campaign from hypothesis to evidence and decision.
It is organized scientifically rather than as a leaderboard. `Combined` is a
pooled GTEx endpoint, not a fifth tissue.

## Evaluation regimes

1. **Published reproduction.** Supplied AdarEdit checkpoints were evaluated on
   the canonical held-out split to verify the paper benchmark.
2. **Pair-disjoint development.** Train+validation pairs were divided into
   five outer folds. All model/feature/threshold choices were isolated from
   the held-out fold. These OOF results measure incremental information under
   identical conditions; their absolute scores do not redefine AdarEdit.
3. **Natural-prevalence development.** Secondary analyses retained all
   eligible high/low observations on development pairs without class
   downsampling.
4. **Preregistered development.** The unified architecture and stopping gates
   were fixed before its outcomes were inspected.
5. **Canonical held-out test.** The selected pipeline was frozen and evaluated
   exactly once. This is the authoritative performance claim.

## Evidence map

| Family | Hypothesis | Principal evidence | Decision | Scope |
|---|---|---|---|---|
| AdarEdit reproduction | Sequence/structure graph benchmark is reproducible without pair leakage | 884 graph-buildable pairs; 566/139/179 train/validation/test pairs; zero overlap; supplied metrics reproduced | Retain as authoritative reference | Published reproduction |
| OOF baseline/error atlas | Cross-fitted predictions can expose residual structure without self-training contamination | Five global pair folds and site master created; thresholds selected inside training data | Retain as internal instrument only | Development OOF |
| RBP occupancy/context | Local RBP binding and tissue abundance explain otherwise plausible false positives | ENCODE eCLIP, PrismNet, RBPNet, GTEx abundance, interactions, matched/random controls | Terminate current operationalization: no credible robust within-tissue increment | Development mechanistic/control |
| Cross-tissue switching | Binding potential × tissue abundance explains label changes for identical sites | Tissue identity nearly saturated switch direction; RBP terms did not add transferable signal | RBP mechanism unsupported at current resolution | Development paired analysis |
| IRAlu/transcript annotations | Repeat/transcript context supplies missing site-level information | Broad panel overfit; no stable residual increment | Terminate site-level panel | Development OOF |
| Pair geometry | Genomic arm separation and substrate geometry mark duplex realization | Actual inter-arm gap correlated with edited-site fraction (Spearman −0.33 to −0.48); pair models beat graph-inferable summaries | Retain biological insight; direct predictive integration insufficient | Development pair-level |
| Substrate engagement | A latent pair-level engagement layer gates site editing | Outer-isolated gate: small AUPRC improvement, nonrobust F1, worse Brier; neural auxiliary variant stopped after adverse folds | Do not promote | Development OOF/control |
| Chromatin | Cotranscriptional state, especially H3K36me3, adds substrate-realization information | Exact-tissue peak overlap small/inconsistent; Liver AUPRC signal did not generalize across metrics/endpoints; H2A.Z lacked fair matched coverage | Terminate current bulk peak-overlap formulation; finer matched data remain open | Development OOF/mechanistic |
| Expanded pair-balanced supervision | All eligible high/low observations are useful if site-rich pairs cannot dominate | AUROC/AUPRC/F1 improved in all five contexts; pair-clustered intervals favorable | **Advance and select** | Same-fold development OOF |
| Continuous/ordinal multitask | Intermediate and continuous EditingIndex enrich supervision | Small ranking/calibration signal; negligible standalone F1 beyond pair-balanced; ensemble reduced natural-prevalence F1 | Do not include in final model | Development OOF + natural prevalence |
| All-A substrate decoder | Jointly decoding all adenosines exploits within-pair structure | Prespecified two-fold futility: −6.64 AUROC, −6.73 AUPRC, −5.99 F1 points | Terminate | Development OOF |
| Within-pair rank | Relative site ordering captures substrate-wide editing propensity | Helped weaker original encoder; no value beyond pair-balanced or shuffled-rank control | Terminate as redundant | Development OOF/control |
| Shared cross-tissue model | Cross-context transfer helps low-data tissues | Macro +0.87 AUROC, +1.16 AUPRC, +0.75 F1 points; missed both +1-point promotion gates and worsened Brier | Retain secondary evidence; do not select | Development OOF |
| Focal-node representation | Explicit target state survives whole-graph pooling | Complementary development ranking signal | Retain as principle | Development OOF |
| Target/partner local encoder | Aligned local sequence/pairing patterns add information omitted by global mean pooling | Improved Muscle and Liver; centered +96-nt network failed; aligned-minus-shifted intervals strongly favorable | **Advance and select** | Development OOF/control |
| RNA foundation model | Frozen RiNALMo supplies missing transferable RNA representation | 35M and 150M embeddings failed to improve ranking beyond selected graph/matched controls | Terminate frozen-embedding formulation | Development OOF/control |
| IRAlu-specific pretraining | Large task-matched unlabeled corpus may enable future transfer | >288,000 benchmark-excluded opposite-orientation pairs found; isoform connectivity, foldability, and redundancy controls unresolved | Defer as a new-data program | Feasibility only |
| In-vivo structure | Cellular structural state explains realized versus predicted duplexes | No tissue/donor-matched scalable structural resource supports a powered benchmark test | Defer; not a negative result | Resource feasibility |
| Donor/sample state | Matched ADAR, interferon, splicing, and composition resolve tissue averaging | Editing outcomes are not available donor-matched at benchmark scale | Defer; not a negative result | Resource feasibility |
| Fixed prediction integration | Global pair-balanced and aligned-local predictors contain complementary errors | Prespecified 50:50 equal-logit fusion improved ranking/F1 and natural-prevalence performance; no weight search | **Select and freeze** | Development OOF/validation |
| Unified end-to-end model | Joint representations exploit the same principles better than fixed prediction fusion | Preregistered 65,698-parameter model underperformed ensemble in AUROC/AUPRC in every fold; worse Brier; repair gate not opened | Reject before test access | Preregistered development |
| Final canonical evaluation | Frozen selected model improves original published benchmark | Positive AUROC/AUPRC/F1 pair-bootstrap intervals in all five endpoints; matched-recall FP reductions | Final locked result | One-use canonical test |

## Report trail

| Report | Subject |
|---|---|
| `00_adaredit_reproduction.md` | Upstream pin, split integrity, supplied-checkpoint reproduction |
| `01_error_atlas.md` | Cross-fitted baseline predictions and residual atlas |
| `02_rbp_literature_and_resources.md` | RBP literature/resource survey and evidence ranking |
| `03_rbp_feature_audit.md` | eCLIP, predicted binding, expression, coverage, and controls |
| `04_cross_tissue_switch_analysis.md` | Matched sites, switches, and continuous tissue differences |
| `05_residual_pilot.md` | Low-capacity residual/adjudication experiments |
| `06_integrated_models.md` | Conditional integration decisions |
| `07_final_locked_test.md` | One-use canonical result and false-positive analysis |
| `08_mechanistic_interpretation.md` | Interpretation boundaries and case analyses |
| `09_performance_frontier_plan.md` | Ranked Biology/AI/Biology×AI opportunity tree |
| `10_chromatin_resource_audit.md` | H3K36me3/H2A.Z resource and leakage audit |
| `11_iralus_and_transcript_context.md` | Repeat, transcript, geometry, and engagement results |
| `12_ensemble_context_pilot.md` | Context/ensemble pilot |
| `13_donor_level_feasibility.md` | Sample-level outcome/resource feasibility |
| `14_invivo_structure_feasibility.md` | In-vivo structure feasibility |
| `15_rna_foundation_model_audit.md` | RiNALMo model/resource and empirical audit |
| `16_richer_supervision_and_substrate_context.md` | Expanded supervision, multitask, shared, all-A, focal/local results |
| `17_iralus_pretraining_feasibility.md` | Task-specific pretraining corpus audit |
| `18_unified_end_to_end_deferred_objective.md` | Deferred integrated-model objective |
| `19_unified_model_preregistration.md` | Frozen unified architecture, gates, and controls |
| `20_unified_end_to_end_result.md` | Unified result and development-only rejection |
| `FINAL_REPORT.md` | Complete scientific synthesis |

The append-only machine-readable record is
[`experiments/experiments.csv`](../experiments/experiments.csv). Each row records
input hashes, folds, seed, features, hyperparameters, artifact, metrics, and
canonical-test access.
