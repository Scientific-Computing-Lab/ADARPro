# 00 — AdarEdit reproduction

## Pinned upstream state

- Repository: `https://github.com/Scientific-Computing-Lab/AdarEdit.git`
- Branch and revision: `main`, `5c8b57d70cdc2eb1c5e168431013275bcc9ed0c5`
- Upstream commit time: 2026-08-10 21:08:15 UTC
- Checkout/audit date: 2026-08-21
- License reported by GitHub: GPL-3.0
- Untouched checkout: `../AdarEdit-upstream`
- ADARpro worktree: branch `adarpro`

The August repository is the strict reproducibility release: 884
graph-buildable inverted-Alu substrates and a single global pair assignment
shared across contexts. A March Research Square record still describes the
older 905-substrate/high-score analysis; it is not the benchmark used here.

## Environment

The unchanged upstream YAML was resolved with flexible channel priority after
the workstation's older classic Conda solver could not resolve the current R
stack under strict priority. Checkpoint-critical versions are Python 3.10.20,
PyTorch 2.7.0+cu126, PyTorch Geometric 2.6.1, NumPy 2.2.5, pandas 2.2.3, and
scikit-learn 1.6.1. Upstream checkpoints report PyTorch 2.7.0+cu128/PyG 2.6.1
on an H200. The local GPU is an RTX 3090 (24 GB). Exact local packages are in
`provenance/conda_packages_2026-08-21.json` and
`provenance/pip_freeze_2026-08-21.txt`.

## Split verification

`python data_construction/verify_split.py` passed all assertions:

- 884 unique, graph-buildable substrates;
- 566/139/179 train/validation/test pairs (64.0/15.7/20.2%);
- zero substrates assigned to more than one split;
- zero cross-context train-or-validation to test site overlap.

The canonical test artifacts are hashed and sealed in
`provenance/test_quarantine.json`. The table below was produced through the
aggregate-only supplied-checkpoint path. No row-level held-out output was
created by ADARpro.

## Supplied-checkpoint reproduction

| Context | Model | AUROC | AUPRC | F1 | Precision | Recall | Specificity | MCC | Threshold | TP | FP | TN | FN | Sites | Pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Artery | Baseline | .9238 | .9262 | .8671 | .8345 | .9023 | .7930 | .7025 | .450 | 2319 | 460 | 1762 | 251 | 4792 | 151 |
| Artery | Bio-aware | .9183 | .9228 | .8585 | .8313 | .8875 | .7916 | .6846 | .400 | 2281 | 463 | 1759 | 289 | 4792 | 151 |
| Brain | Baseline | .9301 | .9286 | .8689 | .8371 | .9032 | .8021 | .7114 | .375 | 2184 | 425 | 1723 | 234 | 4566 | 148 |
| Brain | Bio-aware | .9274 | .9305 | .8633 | .8495 | .8776 | .8250 | .7044 | .250 | 2122 | 376 | 1772 | 296 | 4566 | 148 |
| Liver | Baseline | .9092 | .9242 | .8541 | .8368 | .8721 | .7788 | .6559 | .525 | 2046 | 399 | 1405 | 300 | 4150 | 130 |
| Liver | Bio-aware | .9100 | .9269 | .8543 | .8536 | .8551 | .8093 | .6646 | .375 | 2006 | 344 | 1460 | 340 | 4150 | 130 |
| Muscle Skeletal | Baseline | .8688 | .8757 | .8139 | .7767 | .8549 | .7152 | .5781 | .350 | 654 | 188 | 472 | 111 | 1425 | 140 |
| Muscle Skeletal | Bio-aware | .8782 | .8954 | .8161 | .7874 | .8471 | .7348 | .5874 | .100 | 648 | 175 | 485 | 117 | 1425 | 140 |
| Combined | Baseline | .9331 | .9356 | .8674 | .8334 | .9043 | .8026 | .7125 | .475 | 2296 | 459 | 1866 | 243 | 4864 | 179 |
| Combined | Bio-aware | .9030 | .9097 | .8357 | .8261 | .8456 | .8056 | .6521 | .250 | 2147 | 452 | 1873 | 392 | 4864 | 179 |

All AUROC, AUPRC, F1, precision, recall, specificity, and accuracy values agree
with supplied checkpoint summaries within `2e-5`. The baseline is the stronger
fair default in three of four individual tissues by F1 and in the pooled
Combined endpoint; the Bio-aware model is marginally stronger in Liver and
Muscle Skeletal. This supports using the compact Baseline GAT as the primary
intrinsic-editability reference while retaining Bio-aware results as a
sensitivity comparison.

Machine-readable results: `results/adarpro/baseline_reproduction.csv`.

## Existing joint-model control

The upstream release also supplies a four-output joint model trained across
the individual tissues without treating `Combined` as a tissue. Its published
aggregate summaries are an important secondary comparator for any ADARpro
model that shares information across contexts:

| Context | Joint encoder | AUROC | AUPRC | F1 |
|---|---|---:|---:|---:|
| Artery | Baseline | .9229 | .9254 | .8646 |
| Artery | Bio-aware | .9266 | .9319 | .8664 |
| Brain | Baseline | .9214 | .9213 | .8635 |
| Brain | Bio-aware | .9262 | .9273 | .8644 |
| Liver | Baseline | .9128 | .9269 | .8647 |
| Liver | Bio-aware | .9209 | .9351 | .8577 |
| Muscle Skeletal | Baseline | .8982 | .9014 | .8390 |
| Muscle Skeletal | Bio-aware | .9058 | .9121 | .8352 |

This does not replace the authoritative per-tissue supplied-checkpoint
benchmark above. It does show that shared cross-tissue supervision is already
valuable for data-limited Muscle (Baseline: +.0251 F1 and +.0294 AUROC) and
Liver (+.0106 F1 and +.0036 AUROC). Therefore a successor using the much larger
raw high/low pool must be compared with this existing joint control as well as
with the headline per-tissue Baseline; otherwise it could merely rediscover an
effect already present in AdarEdit.

These supplied-checkpoint held-out results are the authoritative AdarEdit
benchmark and the final performance target for ADARpro. Later grouped OOF
development scores are used only to estimate incremental feature/model value
under identical folds; they do not redefine or supersede this table.

Status: supplied-checkpoint evaluation reproduced. Development-only baseline
retraining and five-fold grouped OOF generation are complete for all five
endpoints; those OOF predictions remain an internal incremental-comparison
instrument and do not replace the published benchmark above.
