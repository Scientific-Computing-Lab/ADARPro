# 09 — Performance-frontier research plan

Frozen initial plan: 2026-08-21, before outcome analysis of the new genomic,
multi-task, substrate-wide, or chromatin branches. The strategy DOCX has SHA256
`6d112433b2329d95639e790271861ebcb18b047b0a2643951e49c39b66455145`.

## Current evidence

**Benchmark hierarchy:** published supplied-checkpoint canonical-test AdarEdit
is the authoritative baseline and final target. Grouped development OOF is only
a leakage-free instrument for measuring candidate deltas under the same folds;
its lower absolute scores are expected from reduced training data and are not
interpreted as a more truthful or preferable benchmark.

- The exact upstream revision is `5c8b57d`; its pair-disjoint split is valid
  (566/139/179 train/valid/test pairs) and supplied-checkpoint metrics reproduce.
- Baseline GAT remains the published model reference. The richer Bio-aware GNN does not
  uniformly improve, so capacity or biologically named intrinsic features alone
  are not a demonstrated route to the frontier.
- The supplied four-tissue/no-Combined joint Baseline already improves Muscle
  over its per-tissue Baseline by +.0251 F1 and +.0294 AUROC, and Liver by
  +.0106 F1. This makes expanded raw cross-tissue supervision a high-prior
  extension, while the existing joint model remains a required secondary
  comparator rather than a result to rediscover.
- All 79,135 balanced development examples have exact GRCh38 coordinates and
  five global pair folds. Muscle now has complete nested grouped-OOF
  development-reference predictions (F1 0.7783, AUROC 0.8456, 942 FP); the
  other contexts remain
  to be cross-fitted.
- The four-tissue natural experiment contains 1,221 strict switch sites. Tissue
  identity nearly saturates switch direction (AUROC 0.9991). No current
  eCLIP/PrismNet/RBPNet × bulk-GTEx formulation improves this ceiling.
- On 205,199 within-site continuous tissue differences, every current RBP
  formulation ties or worsens the tissue-pair baseline. This is strong evidence
  against that measurement strategy, not against RBP biology in general.
- Experimental eCLIP positives are sparse and repeat-censored. Bulk tissue TPM
  cannot represent protein activity, localization, isoform, or cell state.
- The raw development tables contain 275,569 individual-tissue observations,
  including 85,760 intermediate observations. This unused supervision is the
  most accessible AI opportunity.

## Directions below the continuation line

1. Generic CLIP/predicted binding × bulk GTEx expansion: terminated after the
   within-tissue residual experiment lost AUROC/AUPRC across every substantive
   feature block.
2. Strict-switch direction as a model-selection target: terminate because
   tissue identity creates a near-degenerate ceiling.
3. Unrestricted sparse RBP selection, deeper RBP fusion, or heterogeneous RBP
   graphs: terminate absent low-capacity incremental evidence.
4. Larger/deeper GAT alone: deprioritize; effective sample size is 705
   development pairs and existing added capacity has not generalized uniformly.
5. Tissue ID as mechanism: retain only as a negative/control baseline.

## Ranked remaining opportunities

The machine-readable research tree is
`experiments/performance_frontier_opportunities.csv`. The top rank is not the
most glamorous model; it is the best expected information-to-cost experiment.

1. **IRAlu architecture and transcript state (Biology).** Add outcome-independent
   pair gap/span, arm lengths/asymmetry/GC, RepeatMasker subfamily/divergence and
   local reverse-Alu neighborhood, plus GENCODE transcript/exon/intron/UTR,
   isoform compatibility and splice-distance features. First test is a nested
   low-capacity residual model against the cross-fitted baseline logit.
2. **Continuous + ordinal + binary multi-task learning (AI).** Use all absent,
   intermediate and high EditingIndex observations with a shared RNA encoder;
   compare against the same encoder and training budget using binary labels only.
3. **Substrate-wide multi-A prediction (AI × Biology).** Encode each duplex once,
   learn a shared engagement latent, and predict every candidate A jointly. Test
   whether substrate-level supervision adds value beyond site-independent GAT.
4. **Chromatin/co-transcriptional residual pilot (Biology).** Audit the July 2026
   H3K36me3/H2A.Z claim and obtain credible GRCh38 tracks. Start with fixed
   annotations and shifted-locus controls; do not interpret cell-line tracks as
   tissue state.
5. **Hierarchical engagement × chemistry model (Biology × AI).** Advance only
   after experiments 1–4 demonstrate a substrate/state signal. Engagement gets
   substrate/context variables; conditional chemistry gets local graph features.
6. **In-vivo/ensemble structure and mechanistic ADAR contact geometry.** Audit
   feasibility in parallel; scale only if mapped coverage and independence are
   adequate.
7. **Donor/sample-level multimodal learning.** Highest upside but high cost and
   controlled-data risk; require a written pair×donor leakage design and usable
   donor-level editing/omics before implementation.
8. **Frozen RNA language-model fusion.** Test only after cheap new-information
   branches, with pair-grouped evaluation and a frozen embedding cache.

## First five experiments and stop/advance rules

| Order | Experiment | Advance | Revise/combine | Kill |
|---:|---|---|---|---|
| 1 | Finish Muscle RBP residual | ≥0.5 F1 point, positive pair-bootstrap AUROC/AUPRC direction, and FP rescue at matched recall without comparable TP damage | Small consistent calibration/FP benefit | CI crosses zero with no matched-recall benefit, or controls match it |
| 2 | IRAlu + transcript residual | ≥1 F1 point or ≥0.5 point with lower clustered CI above zero on a ranking metric; stable fold direction; beats chromosome/random/shift controls | Stable 0.5–1 point or mechanistically specific FP rescue | <0.5 point, inconsistent folds, or random/control parity |
| 3 | Continuous/ordinal/binary multi-task GAT | ≥1 F1 point with paired pair-bootstrap support and no material calibration/recall damage | 0.5–1 point stable gain or continuous task clearly improves substrate latent for experiment 4 | No gain over identical binary-only encoder/training budget |
| 4 | Substrate-wide joint multi-A model | ≥1 F1 point and improved pair-level/FP metrics across folds | ≥0.5 point plus evidence the latent predicts substrate editing burden | No gain over site-independent model, or gains vanish when pair weighting is correct |
| 5 | Chromatin residual pilot | ≥0.5 F1 point or substantial FP reduction at matched recall, positive clustered CI, correct perturbation-supported direction, and superiority to shifted tracks | Cell-line-specific signal requiring better tissue match | No incremental signal, shifted-track parity, or coverage/context mismatch dominates |

No branch is promoted solely by a point estimate. Feature construction is
outcome-independent; all learned preprocessing and selection stay inside the
training fold. The canonical test remains sealed until the full successor is
frozen, and a negative portfolio is an acceptable scientific endpoint.

## Deferred additive objective recorded 2026-08-24

After the currently committed campaign reaches its next natural decision
boundary, re-audit all development evidence and separately test whether the
surviving technological principles can be learned in one low-capacity,
connected, jointly optimized end-to-end architecture. This does not replace or
alter the selected prediction-level ensemble, which remains a first-class
result and baseline. The full requirements are recorded in
`reports/18_unified_end_to_end_deferred_objective.md`.

Before any canonical-test consumption, explicitly remind the user of this
deferred objective and obtain the sequencing decision. No canonical-test result
may inform the unified architecture, its ablations, or its selection.

## Deferred objective resolution

The user selected development-first sequencing. The unified model was
preregistered and completed with the canonical test still sealed. It lost to
the frozen ensemble on AUROC and AUPRC in every Muscle outer fold, did not
improve F1, and worsened Brier score. Its conditional repair rule was not met.
The campaign therefore terminates this branch and retains the 50:50
pair-balanced + target-local ensemble as the final development winner; see
reports 19–20.
