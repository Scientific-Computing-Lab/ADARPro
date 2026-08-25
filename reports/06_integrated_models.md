# 06 — Integrated models

Status: complete. Development selection, frozen train/validation fitting, the
additional unified end-to-end campaign and the one-use canonical test are all
complete.

The published supplied-checkpoint AdarEdit canonical-test result remains the
authoritative benchmark. All results below are same-fold incremental evidence
from the train+validation development portion only.

## First outer-isolated hierarchical gate

Expanded high/low supervision established a stronger site model without adding
inputs. A separate nested pair-level analysis then showed that actual IRAlu
geometry predicts substrate edited-site burden across every tissue. The first
integration therefore used two explicitly separated stages:

1. an expanded-supervision compact GAT estimates site chemical editability;
2. a ridge model estimates pair engagement from outcome-independent genomic
   geometry and intrinsic pair summaries;
3. a two-feature logistic gate combines their logits/scores.

For every outer fold, both base models excluded the outer and inner pairs. Pair
ridge regularization was selected only within the three training folds; gate
regularization and threshold were selected on the inner fold. Recomputed GAT
probabilities matched the saved outer-fold predictions within 1.14e-05.

On Muscle canonical development rows, the unchanged expanded GAT scored AUROC
0.8659, AUPRC 0.8520 and F1 0.7924. The gate scored 0.8724, 0.8604 and 0.7965.
It made eight more true-positive calls and 19 fewer false-positive calls. Its
pair-bootstrap AUPRC gain was +0.0085 (95% interval +0.0024, +0.0151); AUROC
improved in every fold, while the F1 interval crossed zero. It beat both a
graph-intrinsic engagement score and a shuffled-engagement control on AUPRC,
and beat the shuffled control on AUROC and specificity. Brier score worsened,
so calibration remains unresolved.

Decision: the gate establishes incremental site-ranking information and
unlocks direct integration, but is not the final successor.

## Implemented next model

Pair-balanced expanded supervision is now the leading site model: under the
same development folds it improves AUROC, AUPRC and F1 by 0.0429, 0.0473 and
0.0426 over the internal AdarEdit reference, with all corresponding clustered
intervals above zero. These deltas are development evidence only; the published
held-out benchmark remains the final target.

The direct model retains the compact GAT and pair-balanced expanded high/low
supervision, adds a fold-standardized 16-variable substrate-context encoder,
and adds an auxiliary Huber loss for pair edited-site fraction. This tests the
biological decomposition directly while ensuring that observed outcome burden
is a training target—not an inference-time feature. The implementation also
saves held-out pair-engagement predictions so site and substrate objectives can
be audited separately.

Prespecified controls are:

- pair-balanced expanded GAT without context;
- pair context without the engagement auxiliary loss;
- graph-intrinsic versus genomic-geometry engagement;
- shuffled engagement targets;
- the already completed low-capacity gate.

## Direct model futility result

The direct model was stopped under its prespecified futility rule after three
complete outer folds. On the identical 3,579 rows, it reached AUROC 0.9008,
AUPRC 0.8945 and F1 0.8145, versus 0.8947, 0.8806 and 0.8206 for the
pair-balanced model. Pair-bootstrap median differences were +0.0061 AUROC
(95% interval -0.0040, +0.0165), +0.0138 AUPRC (+0.0006, +0.0284), and
-0.0065 F1 (-0.0190, +0.0072). Recall decreased -0.0209 (-0.0402, -0.0005)
and Brier score worsened by +0.0028 (-0.0067, +0.0120).

The AUPRC direction is worth retaining as evidence that pair context can alter
ranking, but it is driven mainly by the first fold: AUROC/F1 differences for
folds 1-3 were (+0.0185, +0.0146), (-0.0060, -0.0343), and (-0.0007,
-0.0007). Two of three folds therefore failed to improve the primary ranking
and threshold metrics, and the pooled model damaged recall. This is not a
material or reproducible advance beyond pair-balanced supervision.

**Decision:** stop direct context/engagement training for futility and preserve
the three completed outer folds. Do not spend another full ablation family on
this formulation. The simpler outer-isolated gate remains mechanistic evidence,
not a successor. Revisit hierarchical integration only if a stronger and
independent substrate-state measurement becomes available.

At the final locked boundary, the selected prediction-level ensemble—not the
failed unified candidate—was evaluated once on the canonical test. It improved
AUROC/AUPRC/F1 in all five endpoints with positive pair-bootstrap intervals;
full results are in report 07. No integrated candidate was selected or tested.

## Selected integrated successor

The subsequent performance-frontier campaign identified two independently
useful components under the same pair-disjoint folds:

- pair-balanced supervision over every eligible high/low observation on
  training pairs;
- a target/helix-partner-aligned local CNN fused with the compact whole-duplex
  GAT.

A fixed 50:50 logit ensemble of the pair-balanced graph model and target-local
fusion model passed its preregistered ladder in Muscle and Liver, including the
96-nt-shifted local-window control and natural-prevalence transport. In Muscle,
ensemble-minus-pair differences were +0.0256 AUROC, +0.0308 AUPRC and +0.0139
F1. In Liver they were +0.0182, +0.0214 and +0.0179, with all pair-bootstrap
intervals favorable. No ensemble weights were searched.

The four-tissue masked multi-label model produced real but smaller transfer
gains over independently trained pair-balanced models: macro AUROC +0.0087,
AUPRC +0.0116 and F1 +0.0075. It missed both frozen +0.01 promotion gates and
worsened mean Brier, so it is not advanced and no joint-local expansion is
opened.

**Development decision:** the globally selected family is the fixed equal-logit
pair-balanced plus target-local ensemble. It was subsequently trained
separately for all five published contexts using published train pairs, with
checkpoints and thresholds selected only on published validation. This decision
does not use or estimate canonical-test performance; the published AdarEdit
test remains the authoritative target.

## Frozen train/validation completion

The selected family has now completed its prespecified 1,000-epoch, seed-42
fits in all five endpoints. The following figures are the published-validation
selection summaries; because the same validation data select the epoch and
threshold, they are artifact-readiness checks rather than held-out performance
claims.

| Context | Threshold | AUROC | AUPRC | F1 | Sites | Pairs |
|---|---:|---:|---:|---:|---:|---:|
| Artery | 0.475 | 0.9384 | 0.9208 | 0.8711 | 3,876 | 121 |
| Brain | 0.425 | 0.9455 | 0.9339 | 0.8836 | 3,766 | 118 |
| Liver | 0.550 | 0.9407 | 0.9268 | 0.8591 | 3,024 | 102 |
| Muscle Skeletal | 0.225 | 0.9312 | 0.9340 | 0.8693 | 1,288 | 120 |
| Combined | 0.350 | 0.9477 | 0.9412 | 0.8842 | 3,793 | 139 |

All ten component histories contain exactly one row for each epoch 1–1,000.
Every saved best epoch, threshold, validation F1, checkpoint SHA-256,
context/mode configuration and sealed-development flag agrees across history,
checkpoint and summary. Recalculation from each saved validation-prediction
table exactly reproduced the fixed 50:50 logit average, thresholded predictions,
AUROC, AUPRC and F1. After final reporting, the experiment registry contains
128 unique run IDs and no duplicates. The full test/lint gate passes (120
tests; Ruff clean).

The deferred unified end-to-end objective was surfaced and completed before
the final freeze. Its primary joint model lost to the immutable ensemble on
AUROC and AUPRC in every Muscle outer fold, failed its clustered promotion
criteria and did not trigger its conditional repair or replication stages.
Report 20 contains the complete result. The ensemble remained the final
development winner and subsequently improved canonical-test AUROC, AUPRC and
F1 in all five endpoints; report 07 records the single consumed evaluation.
