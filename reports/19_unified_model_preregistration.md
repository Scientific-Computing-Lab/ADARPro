# 19 — Unified ADARPro evidence audit and preregistration

Frozen before unified-model outcomes: 2026-08-24 at repository revision
`a6fd4923df68a1dad35dcf29e0764f392a46b3cb`.

## Evidence that survives the completed campaign

Four technological principles retain credible development support:

1. **Pair-first expanded high/low supervision.** This is the only intervention
   with positive pair-bootstrap AUROC, AUPRC and F1 intervals in every one of
   the five endpoints. It supplies the unified model's training distribution.
2. **Whole-duplex graph representation.** The compact AdarEdit GAT remains the
   strongest validated intrinsic substrate backbone and contributes prediction
   errors that are complementary to the target-local model.
3. **Explicit focal representation.** Concatenating the focal-node GAT state
   with global pooling replicated as a ranking improvement in Muscle and Liver,
   although its standalone F1 evidence was not sufficient for selection.
4. **Target/helix-partner-aligned local representation.** The aligned local CNN
   replicated in Muscle and Liver, transported to natural prevalence, and
   decisively beat the identical 96-nt-shifted control. Its fixed equal-logit
   combination with the pair-balanced graph is the strongest frozen
   prediction-level model.

The following are not included: continuous/ordinal auxiliary objectives,
cross-tissue heads, the tested substrate-engagement objective, static genomic
context, bulk H3K36me3 peaks, eCLIP/PrismNet/RBPNet × bulk expression, frozen
RNA-language-model embeddings, and larger generic message passing. Each either
failed its promotion rule, worsened a relevant endpoint, or lacks an adequately
matched measurement. This exclusion prevents the unified model from becoming
an omnibus concatenation of weak ideas.

## Frozen comparator

The comparator is the completed 50:50 logit ensemble of independently trained
`binary_pair_balanced` and `binary_pair_balanced_local` models. It is immutable.
The Muscle and Liver OOF prediction hashes are recorded in
`provenance/unified_model_selection_policy.json`; the published-train/
validation checkpoints, thresholds, hashes and summaries remain untouched in
`models/frozen/` and `models/frozen_ensemble/`.

## Primary architecture

`unified_target_local_interaction_aux_v1` is one connected network trained from
scratch end-to-end. One compact three-layer GAT produces a whole-duplex mean
embedding and the final embedding of the marked focal nucleotide. The existing
aligned ±32-nt target/helix-partner CNN produces a third view. Each view is
projected to 32 dimensions. The final representation concatenates the three
projected views and their three elementwise pair interactions, then passes
through a 64-unit fusion layer to one binary logit.

A training-only global binary head receives the whole-duplex embedding with a
fixed loss weight of 0.25. It is not an inference-time prediction and is not
averaged with the final logit. Its purpose is to preserve the independently
validated global signal while the main loss jointly trains the shared GAT,
focal view, local CNN and interaction fusion. All components receive gradients
in the same optimization process. The architecture is therefore neither a
frozen-model wrapper nor prediction stacking or logit ensembling.

Training otherwise exactly preserves the successful pair-balanced protocol:
all eligible high/low observations from non-held-out development pairs, equal
class sampling, uniform pair then site sampling, the canonical-training-row
update budget, 1,000 epochs, Adam at 1e-3 with 1e-4 weight decay, batch size 256,
and the established five outer/inner pair folds.

## Staged evaluation and stop rules

1. Complete five-fold Muscle OOF and compare it directly with the immutable
   pair-plus-local ensemble on identical rows and folds.
2. Advance only for credible ensemble-relative value: preferably a positive
   clustered F1 interval without ranking damage, or positive clustered AUROC
   and AUPRC intervals with nonnegative F1, acceptable Brier calibration and a
   favorable false-positive/true-positive tradeoff.
3. If the primary candidate advances, run three explanatory controls: no
   global auxiliary loss, first-node instead of focal-node view, and the same
   local encoder shifted 96 nt from the target. These isolate auxiliary
   supervision, focal identity and target alignment.
4. Replicate an advancing architecture in Liver. It cannot be promoted on a
   Muscle-only result.
5. If the primary fails only through an auxiliary-loss optimization conflict
   while its ranking remains complementary, the no-auxiliary model is the only
   prespecified repair. If both fail, terminate rather than search architectures.
6. A surviving family must be confirmed development-only across all five
   endpoints before a global selection decision and frozen train/validation
   fitting.

All uncertainty is clustered by `pair_id`. Natural-prevalence transport is
required for a promoted candidate. No development delta will be projected onto
the published test score. The canonical test remains sealed until the unified
campaign and final model-family decision are complete.

## Outcome pointer

The primary experiment was completed after this document was frozen. It failed
the promotion rule, and the conditional repair/ablation/replication stages were
not triggered. The full result and rationale are recorded without changing
this preregistration in `20_unified_end_to_end_result.md` and
`provenance/unified_development_decision.json`.
