# 15 — RNA foundation-model feasibility

## Candidate audit

RiNALMo is the cleaner first executable candidate. The official implementation
(`lbcb-sci/RiNALMo`, audited commit `2c2c5c14a5ae609d8c560a5d9ca32e51e0288955`)
provides nucleotide embeddings from models pretrained on 36 million
non-coding RNA sequences. Official Zenodo record `15043668` supplies 35M,
150M and 650M parameter checkpoints (134 MB, 592 MB and 2.60 GB), with CC BY
4.0 model parameters; code is Apache-2.0. Rotary positions avoid a documented
short maximum that would truncate the roughly 600-nt AdarEdit substrates.

ERNIE-RNA is scientifically attractive because it learns
structure-enhanced attention and exposes token embeddings and attention maps.
The official repository (`Bruce-ywj/ERNIE-RNA`, audited commit
`43bc06de1088ed03ffd7de918ad4b2c2a3346a43`) is MIT-licensed, but its reference
environment is tied to older Fairseq/PyTorch and its checkpoints are hosted on
Google Drive. That adds more environment and provenance risk than RiNALMo for
the first test. Maintained MultiMolecule conversions exist for both model
families, but they are third-party reimplementations and are not the primary
scientific reference.

## ADARpro formulation

The foundation representation will be complementary, not a replacement for
the explicit RNA graph. Only development substrates are embedded during model
selection. The first low-variance test caches one embedding per unique
substrate, then uses:

- the target-token representation;
- a whole-substrate masked mean;
- a target-centered local mean;
- fold-fitted PCA or another fixed low-dimensional projection;
- the cross-fitted AdarEdit logit as the reference feature.

All PCA, scaling, regularization and thresholds must be fitted inside the
training/inner folds. Caching an outcome-independent frozen embedding is safe;
fitting a projection on all development labels or all-fold covariance is not.
Because there are only 585 independent Muscle pairs, an unrestricted
650M-model fine-tune or thousands of raw embedding coordinates would be
statistically indefensible.

## Staged decision

Start with the official 35M RiNALMo checkpoint to validate token alignment,
memory, cache identity and nested residual signal. Advance to 150M only if the
micro representation shows a stable ranking/F1 direction or a clear
target-local signal; reserve 650M for confirmation of a promising family.
Terminate if fold-fitted low-dimensional embeddings do not improve the shared
development reference and matched random/PCA controls perform similarly.

The official code and micro checkpoint were pinned and retrieved on 2026-08-22.
The checkpoint's published MD5 matched exactly; its local SHA256 is recorded in
`provenance/rinalmo_micro.json`. This preparation did not access labels or the
canonical test partition and does not promote the branch ahead of the active
target-local, independent-tissue and cross-tissue-supervision experiments.
The 33,491,074-parameter model passed a CUDA inference smoke test with finite
12-token × 480-dimensional output. Two API-only compatibility changes were
needed for FlashAttention 2.8.3 and are checksum-recorded in the provenance
file; neither changes model parameters or attention semantics.

The predeclared 150M (`mega`) confirmation checkpoint was retrieved only after
the micro decision was frozen. Its published MD5 matched exactly; byte count,
local SHA256, source URL, code commit and compatibility patch are recorded in
`provenance/rinalmo_mega.json`.

The frozen micro model was then applied once to the 585 unique Muscle
development substrates. The cache contains aligned final-layer target-token,
whole-sequence mean and prespecified ±16-nt local-mean embeddings for 5,967
canonical development sites (480 dimensions per view, stored as float16).
Construction was outcome-independent and rejected any sealed split. The cache
and index checksums are recorded in the experiment registry. No predictive
claim is made until the fold-fitted projection/residual experiment and its
matched-dimensional controls are complete.

## RiNALMo micro residual result

The outer-isolated residual used the fixed pair-balanced plus focal ensemble as
its RNA-graph base. For every outer fold, both neural components were trained
only on the other three training folds, the RiNALMo PCA projections were fitted
only there, and logistic regularization and threshold were selected on the
disjoint inner fold. The primary foundation representation concatenated 16
fold-fitted principal components from each of target-token, ±16-nt local-mean
and whole-substrate-mean embeddings. Shuffled and same-dimensional Gaussian
controls used the identical gate.

The combined representation did not improve ranking: AUROC/AUPRC/F1 were
0.9038/0.8962/0.8337 versus 0.9088/0.8996/0.8298 for base-logit
recalibration. Pair-bootstrap differences were -0.0049 AUROC (95% interval
-0.0129, +0.0019), -0.0031 AUPRC (-0.0114, +0.0045), and +0.0041 F1
(-0.0055, +0.0125). It did not beat the random control.

The target-token view alone was the only weak positive. It reached AUROC
0.9089, AUPRC 0.8993 and F1 0.8364. Against the unmodified fixed base ensemble,
its F1 difference was +0.0059 (+0.0000, +0.0121), driven by +0.0130 recall;
AUROC and AUPRC intervals crossed zero. Against the matched-random control,
ranking and F1 intervals also crossed zero, although Brier improved by 0.0057
(0.0025, 0.0090 reduction). It did beat the within-fold shuffled-embedding
control on AUROC (+0.0085), AUPRC (+0.0083), F1 (+0.0079), and Brier, with all
intervals favorable. Target-versus-recalibrated-base AUROC and AUPRC moved in
the favorable direction in all five folds despite the near-zero pooled OOF
difference, indicating fold-to-fold score-scale variation.

**Decision:** this is below the material-gain threshold and does not justify
foundation-model integration. It narrowly satisfies the predeclared condition
for one fixed 150M-model confirmation because the target-aligned view, but not
the broader views, showed consistent fold direction and a small F1/calibration
effect. The 150M run must reuse the same target-only 16-component formulation;
no window, PCA-dimension or regularization expansion is allowed. Failure to
beat the matched-random control on ranking and F1 terminates the branch.

## Fixed 150M confirmation

The 150M model was evaluated exactly once with the frozen target-token,
16-component formulation. It reached AUROC 0.9082, AUPRC 0.8972 and F1 0.8369.
Relative to the unchanged graph ensemble (0.9092/0.8986/0.8304), AUROC changed
by -0.0011 (95% pair-bootstrap interval -0.0046, +0.0026), AUPRC by -0.0014
(-0.0087, +0.0052), and F1 by +0.0065 (+0.0009, +0.0118). Thus the small F1
effect replicated, but no ranking gain did.

The matched-random gate reached AUROC 0.9057, AUPRC 0.8961 and F1 0.8270.
RiNALMo exceeded it on F1 by +0.0099 (+0.0030, +0.0165), but the AUROC
(+0.0024; -0.0036, +0.0083) and AUPRC (+0.0010; -0.0063, +0.0084) intervals
crossed zero. It therefore fails the predeclared requirement to beat random
features on both ranking and F1. Scaling from 35M to 150M parameters did not
produce a material or control-robust ranking improvement.

**Decision:** terminate frozen RNA-foundation residual fusion. Do not test the
650M model or another foundation architecture on these development labels.
The result does not exclude future task-specific pretraining on a much larger,
benchmark-excluded IRAlu corpus, but generic frozen embeddings are not part of
the successor candidate.

This branch is below expanded high/low supervision, cross-tissue sharing and
target-centered structural features in the current queue, but it is fully
feasible on the local 24 GB GPU.

Sources audited 2026-08-21:

- `https://github.com/lbcb-sci/RiNALMo`
- `https://zenodo.org/records/15043668`
- `https://github.com/Bruce-ywj/ERNIE-RNA`
- RiNALMo DOI `10.1038/s41467-025-60872-5`
- ERNIE-RNA DOI `10.1038/s41467-025-64972-0`
