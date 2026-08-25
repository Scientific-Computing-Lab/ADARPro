# 10 — Chromatin/co-transcriptional resource audit

Cai et al. 2026 (bioRxiv DOI 10.64898/2026.07.27.740859) reports that local
H3K36me3 is the strongest negative predictor of editing frequency across its
human-cell-line and mouse-embryonic-tissue multi-omics analyses, whereas
H2A.Z.1 is positively associated. Perturbation provides directionally stronger
evidence than association: H2AFZ knockdown decreased 76.3% of high-frequency
sites (mean Δ −0.07), while SETD2 knockout/loss of H3K36me3 increased 86.9% of
low-frequency sites (~2.7-fold).

This is mechanistically attractive because chromatin is upstream information
that sequence plus an RNAfold MFE graph cannot uniquely infer. It is not yet a
license to attach arbitrary cell-line epigenomes to GTEx tissues.

An ENCODE API audit on 2026-08-21 found released H3K36me3 experiments for
liver (12), skeletal muscle tissue (3), and tibial artery (4), but none for the
exact cerebellum biosample. H2AFZ/H2A.Z experiments exist in proxies such as
myotube, skeletal muscle myoblast, HepG2, astrocyte, and endothelial cells, not
the four exact ADARpro tissues.

Decision:

1. audit and aggregate exact-tissue H3K36me3 tracks for Artery, Liver, and
   Muscle using a prespecified replicate rule;
2. treat H2AFZ tracks only as generic/cell-proxy annotations unless a genuinely
   matched source is found;
3. include shifted-locus and chromosome controls;
4. never assign an unmatched track a tissue-specific mechanistic meaning;
5. evaluate first as a low-capacity grouped residual layer.

Machine-readable availability is in `context/chromatin/resource_audit.csv`.

## Exact-track acquisition and feature audit

The file-level audit retained 11 current preferred GRCh38 pseudoreplicated
peak tracks: four tibial artery, four liver, and three skeletal-muscle donors.
Eight older liver experiments were excluded explicitly because they have no
valid preferred GRCh38 peak artifact. Checksums, accessions, pipeline versions,
and local paths are frozen in `provenance/encode_h3k36me3.json`. There is no
exact cerebellum track and no chromatin vector was fabricated for `Combined`.

Peak-derived features were prespecified at the focal site, ±250 nt, and whole
IRAlu-pair span, then aggregated across donors. A +1 Mb shifted-locus panel has
identical dimensionality. In Muscle Skeletal, 6.4% of site positions, 16.7% of
±250 nt windows, and 22.8% of pair spans overlap a peak on average across
donors; shifted values are 4.3%, 5.9%, and 7.6%, respectively.

## Muscle development residual pilot

The published supplied-checkpoint test result remains the authoritative
AdarEdit benchmark. This pilot uses the reduced-training grouped OOF reference
only to estimate an incremental feature delta.

Pair-span H3K36me3 is the first context block to pass a continuation rule. The
same-fold baseline-logit-plus-pair model has a clustered median ΔF1 of +0.0122
(95% CI +0.0005 to +0.0244) and Δspecificity of +0.0443 (+0.0269 to +0.0619).
At its inner-fold-selected thresholds it rescues 17.3% of baseline false
positives while damaging 4.3% of baseline true positives. The effect is not a
general ranking gain: ΔAUROC is −0.0009 (−0.0109 to +0.0076) and ΔAUPRC is
−0.0079 (−0.0184 to +0.0021).

The richer multiscale block has median ΔAUROC +0.0047 but its interval crosses
zero and its AUPRC is flat to negative. Both negative controls lose: the +1 Mb
shifted block has median ΔF1 −0.0017 and the matched random block −0.0003.
With thresholds selected on each inner fold to target the reference recall,
the pair block retains a median Δspecificity of +0.0391 (95% CI +0.0239 to
+0.0550); the realized outer-fold Δrecall is −0.0093 (−0.0274 to +0.0059) and
ΔF1 +0.0095 (−0.0011 to +0.0189). Thus the false-positive benefit persists,
but exact recall matching does not transfer perfectly and the F1 interval
crosses zero under this operating rule.
The strongest stable coefficient is biologically concordant: mean pair-span
H3K36me3 signal is negative in all five training fits, while distance from the
mark is positive in all five. Correlated overlap terms themselves are less
stable and are not interpreted separately.

**Decision:** advance only the six-variable pair-span formulation to an
independent-context replication once Liver grouped OOF is complete. Do not yet
download multi-gigabyte signal tracks or integrate chromatin into the neural
model. A Liver failure, loss under a stronger matched-recall analysis, or a
directional reversal will terminate this branch; replication will justify a
chromatin × substrate-engagement integration test.

Artifacts: `data/derived/features/chromatin_h3k36me3.parquet` and
`experiments/runs/chromatin_residual/MuscleSkeletal/`, including the saved
foldwise standardized coefficients.

## Substrate-engagement test

After a learnable pair-level engagement target was established, the exact same
H3K36me3 annotations were tested at their more natural substrate-level endpoint:
fraction of high sites per pair. A nested ridge model compared 16 intrinsic and
genomic-geometry variables alone with the same variables plus the six pair-span
H3K36me3 features. The +1 Mb shifted features remained the negative control.

H3K36me3 did not improve the geometry model in Artery or Liver. Median paired
differences for H3K36me3 were respectively -0.0017 and -0.0122 R², -0.0033 and
-0.0103 Spearman, and -0.0053 and -0.0058 any-high AUROC; every interval included
zero or favored the reference. Muscle point estimates were +0.0315 R², +0.0569
Spearman and +0.0232 AUROC, but all 95% pair-bootstrap intervals crossed zero.
The shifted control was null or negative.

## Liver site-level replication and decision

The prespecified Liver residual analysis is complete on 15,509 canonical
development rows from 508 pairs. The multiscale occupancy block added median
+0.0050 AUPRC (95% pair interval +0.0011, +0.0096), +0.0019 AUROC (-0.0010,
+0.0047), and +0.0037 F1 (-0.0003, +0.0078) relative to the cross-fitted
development-reference logit recalibrator. Specificity increased +0.0205
(+0.0136, +0.0275), but recall decreased -0.0080 (-0.0141, -0.0021). The
AUPRC effect was positive in four folds and reversed in fold 5 (-0.0113).

This is a weak site-ranking signal, not a material predictive gain. Together
with the null Liver and Artery pair-engagement results, threshold instability,
fold heterogeneity, and lack of clear superiority to the shifted-locus control,
it does not meet the prespecified advance rule.

**Decision:** terminate the current bulk exact-tissue ENCODE H3K36me3
peak-overlap formulation and do not integrate it into the neural successor.
This does not refute the perturbation biology; donor-matched quantitative
chromatin state, rather than pooled peak presence, remains a scientifically
distinct future measurement.

Additional artifacts: `experiments/runs/pair_engagement_chromatin/` and
`experiments/runs/chromatin_residual/Liver/`.
