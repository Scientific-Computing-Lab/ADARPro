# 07 — Final locked test

Status: complete. The canonical test authorization was consumed exactly once.

## Frozen candidate and execution

Development selected the unchanged 50:50 logit ensemble of independently
trained `binary_pair_balanced` and `binary_pair_balanced_local` models. The
unified end-to-end candidate was completed first and rejected using development
data alone. No canonical row informed that choice.

The final manifest was frozen at git revision
`0f50fac33bec8bad325e6745d83ceaacebe42fc1` with SHA-256
`dd04f9786d52f57653da31b12dc777d10aa4247dfccd2b2c59ddff64ad395a1c`.
It binds the five pairs of checkpoint hashes, five validation-selected
thresholds, fixed weights, environment and data hashes, inference code, the
original selection policy, and the unified preregistration and decision. The
guard consumed its authorization on 2026-08-25 at 04:28:30 UTC and records
`evaluations_consumed = 1`. The final metrics file has SHA-256
`ddd5352c48e84bb121e21d89e5d1777a3657caa06f28fa94ef1e8b3c58a74a0e`.

## Canonical-test performance

| Context | Model | AUROC | AUPRC | F1 | Precision | Recall | Specificity | MCC | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Artery | Published AdarEdit | 0.9238 | 0.9262 | 0.8671 | 0.8345 | 0.9023 | 0.7930 | 0.7025 | 0.1077 |
|  | Frozen ensemble | 0.9477 | 0.9485 | 0.8870 | 0.8840 | 0.8899 | 0.8650 | 0.7553 | 0.0879 |
| Brain | Published AdarEdit | 0.9301 | 0.9286 | 0.8689 | 0.8371 | 0.9032 | 0.8021 | 0.7114 | 0.1050 |
|  | Frozen ensemble | 0.9508 | 0.9504 | 0.8902 | 0.8741 | 0.9069 | 0.8529 | 0.7622 | 0.0855 |
| Liver | Published AdarEdit | 0.9092 | 0.9242 | 0.8541 | 0.8368 | 0.8721 | 0.7788 | 0.6559 | 0.1196 |
|  | Frozen ensemble | 0.9414 | 0.9500 | 0.8812 | 0.8946 | 0.8683 | 0.8670 | 0.7324 | 0.0951 |
| Muscle Skeletal | Published AdarEdit | 0.8688 | 0.8757 | 0.8139 | 0.7767 | 0.8549 | 0.7152 | 0.5781 | 0.1526 |
|  | Frozen ensemble | 0.9144 | 0.9207 | 0.8541 | 0.8230 | 0.8876 | 0.7788 | 0.6730 | 0.1296 |
| Combined | Published AdarEdit | 0.9331 | 0.9356 | 0.8674 | 0.8334 | 0.9043 | 0.8026 | 0.7125 | 0.1043 |
|  | Frozen ensemble | 0.9491 | 0.9513 | 0.8852 | 0.8463 | 0.9279 | 0.8159 | 0.7510 | 0.0870 |

Across the five endpoints, the macro-average changes are +0.0277 AUROC,
+0.0261 AUPRC and +0.0253 F1. The F1 gain ranges from +0.0178 in Combined to
+0.0401 in Muscle Skeletal. Every context has a positive 95% pair-bootstrap
interval for AUROC, AUPRC and F1:

| Context | ΔAUROC (95% pair CI) | ΔAUPRC (95% pair CI) | ΔF1 (95% pair CI) |
|---|---:|---:|---:|
| Artery | +0.0239 [0.0186, 0.0294] | +0.0224 [0.0162, 0.0290] | +0.0199 [0.0106, 0.0297] |
| Brain | +0.0206 [0.0154, 0.0257] | +0.0217 [0.0130, 0.0323] | +0.0213 [0.0125, 0.0309] |
| Liver | +0.0322 [0.0252, 0.0395] | +0.0258 [0.0186, 0.0344] | +0.0271 [0.0147, 0.0400] |
| Muscle Skeletal | +0.0456 [0.0300, 0.0628] | +0.0450 [0.0266, 0.0679] | +0.0401 [0.0190, 0.0621] |
| Combined | +0.0160 [0.0109, 0.0212] | +0.0157 [0.0089, 0.0234] | +0.0178 [0.0098, 0.0258] |

Brier score improves in all five contexts, with every clustered interval below
zero. The result therefore improves discrimination, thresholded classification
and probability accuracy rather than trading one metric family for another.

## False-positive behavior

At the immutable validation-selected operating points:

| Context | Baseline FP | Ensemble FP | FP rescued | New FP | FP rescue rate | TP damage rate | Net TP change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Artery | 460 | 300 | 202 | 42 | 43.9% | 4.9% | -32 |
| Brain | 425 | 316 | 164 | 55 | 38.6% | 3.3% | +9 |
| Liver | 399 | 240 | 202 | 43 | 50.6% | 5.9% | -9 |
| Muscle Skeletal | 188 | 146 | 83 | 41 | 44.1% | 4.3% | +25 |
| Combined | 459 | 428 | 147 | 116 | 32.0% | 1.7% | +60 |

Across endpoint tables, false positives fall from 1,931 to 1,430 while true
positives rise from 9,499 to 9,552. These totals are descriptive because the
same biological sites can appear in multiple context tables.

At exactly the reproduced baseline recall, the ensemble avoids 126, 119, 158,
68 and 148 false positives in Artery, Brain, Liver, Muscle and Combined. The
within-resample matched-recall bootstrap interval excludes zero for every
context (reductions of 85–175, 75–166, 116–198, 33–98 and 95–195,
respectively). Thus the false-positive benefit is not an artifact of accepting
lower sensitivity.

## Interpretation boundary

The result supports pair-first use of expanded high/low supervision and an
explicit target/helix-partner-aligned representation as predictive advances.
It does not rescue the negative RBP result and does not establish a new
cell-environment mechanism. The post-lock analyses are descriptive and cannot
reopen model selection. Natural-prevalence evidence remains the prespecified
development-pair secondary analysis; no additional natural-prevalence test was
run after the one-use canonical evaluation.

Reproducible source tables and figures are in
`results/adarpro/final_locked_test/` and `figures/final/`.
