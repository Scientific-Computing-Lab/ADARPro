# 05 — Residual pilot

## Muscle Skeletal decisive experiment

The exact compact Baseline GAT was cross-fitted across the five global pair
folds. Every outer prediction came from a model trained on three other folds;
the fourth fold selected epoch and threshold. Across 5,967 balanced canonical
development examples (585 pairs), the raw OOF development reference achieved
AUROC 0.8456, AUPRC 0.8256 and F1 0.7783, with 942 false positives and 465
false negatives.
These are internal development-reference values from models trained on only
three of five development folds. They are not the AdarEdit benchmark. The
published supplied-checkpoint canonical test performance in report 00 remains
the authoritative baseline and final target.

Low-capacity L2 logistic adjudicators then used the OOF development-reference
logit plus prespecified context blocks. C and threshold were selected on an
inner pair fold, and all reported predictions are outer-fold predictions. Pair-bootstrap
differences use the cross-fitted development-reference logit recalibrator,
so feature gains cannot be confused with logistic recalibration.
Only within-protocol deltas are used for advancement decisions; a frozen
successor will ultimately be compared directly with published AdarEdit under
the original canonical held-out conditions.

| Added information | ΔAUROC median (95% pair CI) | ΔAUPRC median (95% pair CI) | ΔF1 median (95% pair CI) |
|---|---:|---:|---:|
| eCLIP | −0.0103 (−0.0223, −0.0017) | −0.0215 (−0.0388, −0.0073) | −0.0025 (−0.0114, +0.0040) |
| RBPNet sequence profiles | −0.0092 (−0.0171, −0.0006) | −0.0153 (−0.0268, −0.0037) | +0.0005 (−0.0081, +0.0096) |
| PrismNet | −0.0200 (−0.0304, −0.0115) | −0.0288 (−0.0413, −0.0176) | −0.0089 (−0.0200, −0.0000) |
| binding + bulk expression | −0.0340 (−0.0471, −0.0213) | −0.0445 (−0.0614, −0.0285) | −0.0176 (−0.0304, −0.0047) |
| biological suppressors | −0.0026 (−0.0074, +0.0026) | −0.0070 (−0.0147, −0.0002) | −0.0006 (−0.0073, +0.0061) |
| matched random features | −0.0277 (−0.0331, −0.0225) | −0.0295 (−0.0365, −0.0226) | −0.0241 (−0.0323, −0.0167) |
| eCLIP shifted +1 Mb | −0.0012 (−0.0022, −0.0002) | −0.0039 (−0.0072, −0.0006) | −0.0008 (−0.0025, +0.0010) |

At matched-recall thresholds, no context model improved specificity relative
to that recalibrator. Apparent raw FP rescue was accompanied by
TP damage and/or additional false positives. True-locus eCLIP performed worse
than its +1 Mb genomic-shift control, which argues against a substrate-local
occupancy explanation in this operationalization.

**Decision:** terminate generic eCLIP/PrismNet/RBPNet × bulk-GTEx expansion.
The data do not justify a deeper fusion model. RBP work can re-enter only with
materially different evidence such as matched perturbation effects, activity,
localization, or tissue/cell-matched repeat-aware occupancy.

Artifacts: `experiments/runs/residual_pilot/MuscleSkeletal/`.
