# 01 — Development error atlas

## Muscle Skeletal grouped-OOF atlas

The first complete context contains 5,967 balanced canonical development
examples from 585 pairs. The cross-fitted Baseline GAT produces 942 false
positives, including 411 with OOF probability at least 0.8. These errors are
the development-only target for matched mechanistic analysis; no canonical
test errors were inspected.

This OOF atlas is an error-discovery instrument, not an alternative benchmark.
Its models use reduced training subsets; the published canonical AdarEdit test
result remains the authoritative performance reference.

## Natural prevalence

Applying the same five outer-fold checkpoints and validation-selected
thresholds to all eligible high/low sites on development pairs yields 54,289
Muscle observations from 597 pairs. Edited prevalence is 5.40%. AUROC remains
0.8478, but AUPRC is 0.2711 and precision is only 0.1351 at recall 0.8415:
2,469 TP, 15,804 FP, 35,551 TN and 465 FN. Thus the balanced benchmark masks a
large operational false-positive burden; later successors must report both
regimes.

Artifacts: `results/adarpro/error_atlas/` and
`experiments/runs/natural_prevalence/MuscleSkeletal/`.
