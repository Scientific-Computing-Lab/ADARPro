# 20 — Unified end-to-end ADARPro result

Status: development campaign complete; unified family not promoted; canonical
test remained sealed throughout.

## Question and frozen comparator

This experiment asked whether the campaign's independently validated
technological principles could interact more effectively inside one jointly
trained model than in the immutable 50:50 pair-balanced + target-local
equal-logit ensemble. The ensemble, its weights, OOF predictions, final
checkpoints, thresholds and selection policy were not changed. Its Muscle OOF
table retained SHA-256
`85fbc8826efdce06cd061f87f482409ad597f015aca0be769b775bc0ebeeb8f7`.

The evidence audit retained pair-first expanded high/low supervision, the
compact whole-duplex GAT, an explicit focal-node representation and the
target/helix-partner-aligned local view. It excluded the campaign's failed or
sub-threshold RBP, chromatin, multitask, substrate-engagement, joint all-A,
cross-tissue, generic larger-GNN and frozen-foundation-model formulations.

## Architecture and training

`unified_target_local_interaction_aux_v1` is one connected graph trained from
scratch. A shared three-layer GAT produces a whole-duplex mean representation
and the marked focal nucleotide's final node state. A two-layer dilated CNN
encodes aligned ±32-nt windows around the target and its helix partner. The
global, focal and local views are projected to 32 dimensions; the three views
and all three elementwise pair interactions enter a 64-unit fusion layer and
one final binary head. A training-only global binary head has fixed loss weight
0.25. It is not used at inference. Gradients flow from the joint loss through
all representation modules.

The model has 65,698 trainable parameters, less than the 80,138 parameters
across the ensemble's two independently trained components. Training used the
frozen pair-first, class-balanced expanded high/low sampler; the canonical-row
epoch update budget; five outer pair folds; inner fold `(outer + 1) mod 5` for
checkpoint and threshold selection; 1,000 epochs; Adam at 1e-3 with 1e-4
weight decay; batch size 256; and seed 20260821. All outer predictions were
generated before outcome comparison.

## Development result

On the identical 5,967 Muscle development rows from 585 Alu pairs:

| Model | AUROC | AUPRC | F1 | Precision | Recall | Specificity | MCC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen pair + local ensemble | 0.9142 | 0.9040 | 0.8347 | 0.8273 | 0.8422 | 0.8299 | 0.6720 | 0.1168 |
| Unified end-to-end | 0.9072 | 0.8922 | 0.8317 | 0.8074 | 0.8575 | 0.8022 | 0.6603 | 0.1372 |

Pair-clustered 2,000-replicate differences for unified minus ensemble were:

| Metric | Point difference | 95% pair-bootstrap interval |
|---|---:|---:|
| AUROC | -0.0070 | [-0.0129, -0.0015] |
| AUPRC | -0.0118 | [-0.0202, -0.0043] |
| F1 | -0.0029 | [-0.0125, +0.0066] |
| Precision | -0.0198 | [-0.0300, -0.0099] |
| Recall | +0.0153 | [+0.0021, +0.0291] |
| Specificity | -0.0277 | [-0.0396, -0.0160] |
| Brier (lower is better) | +0.0203 | [+0.0135, +0.0276] |

AUROC and AUPRC were lower in every outer fold. Foldwise AUROC differences
were -0.0146, -0.0052, -0.0117, -0.0087 and -0.0023; AUPRC differences were
-0.0198, -0.0050, -0.0115, -0.0147 and -0.0066. F1 was mixed
(-0.0127, +0.0066, -0.0178, +0.0010, +0.0080), while Brier was worse in all
five folds.

At the frozen fold-specific thresholds, the unified model rescued 105 of the
ensemble's 516 false positives but introduced 189 new false positives. It
damaged 107 of 2,471 ensemble true positives and recovered 152 false negatives.
The net tradeoff therefore increased both false positives and total errors.

Against the standalone target-local component, the unified model had only a
small, uncertain AUROC difference (+0.0026 point estimate; bootstrap median
+0.0024, 95% interval -0.0031 to +0.0086), no AUPRC or F1 improvement, and
worse Brier. Thus it did not uncover a representation-level interaction that
the prediction-level ensemble missed.

## Prespecified decision

The primary candidate fails the frozen promotion rule. Its ranking is worse
than the ensemble in every fold, both ranking intervals exclude zero in the
adverse direction, F1 does not improve, and calibration worsens. The sole
no-auxiliary repair was permitted only if the primary failed through an
auxiliary-loss optimization/calibration conflict **while retaining
complementary ranking**. That condition was not met. Accordingly no repair,
conditional ablation family, Liver replication, all-context confirmation or
natural-prevalence transport was triggered. Running those experiments would
have converted a preregistered staged screen into post-outcome architecture
search.

The unified campaign is terminated. The frozen 50:50 pair-balanced +
target-local ensemble remained the global development winner and was later
evaluated in the one-use canonical test, where it improved AUROC, AUPRC, F1 and
Brier score in all five endpoints (report 07). This negative unified result is
informative: with the
available effective sample size, complementarity between global and aligned
local predictors is exploited more reliably by fixed prediction-level fusion
than by this low-capacity joint representation-learning formulation.

Machine-readable decision provenance is in
`provenance/unified_development_decision.json`; full-size artifacts are
hash-addressed there and registered in `experiments.csv`.
