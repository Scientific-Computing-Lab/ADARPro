# 11 — IRAlu architecture and transcript context

## Feature audit

Outcome-independent annotation of all 705 development pairs uses UCSC hg38
RepeatMasker (1,282,821 Alu records; SHA256 in provenance) and GENCODE v46.
The panel contains 9 physical geometry, 35 Alu identity/divergence/neighborhood,
20 pair/site transcript-state, and 13 engineered intrinsic-control features.

All arms map completely to RepeatMasker Alu records and are oppositely
oriented, as expected by construction. Useful variation remains: median
inter-arm genomic gap is 470 nt (range 1–12,540), median span 1,058 nt,
same-subfamily pairs are 6.8%, mean milliDiv is 114.5, and the median number of
additional Alus within the ±5 kb neighborhood is seven. GENCODE reports at
least one compatible same-strand transcript for 96.0% of pairs; transcript
features remain isoform-sensitive and are not treated as observed splicing.

## Linear residual pilot: Muscle Skeletal

The published canonical AdarEdit test result remains the authoritative
benchmark. This pilot uses the lower, reduced-training grouped OOF reference
only to estimate feature deltas without leakage.

The broad architecture block does not improve the internal reference. Median
pair-bootstrap deltas are ΔAUROC −0.0103 and ΔF1 −0.0062, with wide intervals;
RepeatMasker and transcript blocks also lose. A matched 64-dimensional random
block loses ΔF1 −0.0076, confirming substantial overfitting pressure from only
585 context-specific development pairs.

The nine physical geometry features are the sole weak exception: median
ΔAUROC +0.0016 (95% CI −0.0079 to +0.0105) and ΔF1 +0.0047 (−0.0056 to
+0.0141). F1 improves in four of five folds, but one fold has negative AUROC
and the clustered intervals cross zero. This is below the prespecified
advancement threshold and is not a material gain.

## Prespecified nonlinear geometry revision

A shallow, inner-fold-selected boosted tree tested the possibility that the
distance relationship is real but nonlinear. Relative to its same-fold
baseline-logit reference, the clustered median deltas were ΔAUROC +0.0003
(95% CI −0.0065 to +0.0072), ΔAUPRC −0.0095 (−0.0211 to +0.0026), and ΔF1
+0.0030 (−0.0043 to +0.0114). F1 increased in four folds, but AUROC fell in
two folds and the pooled AUPRC was lower. This revision therefore also fails
the prespecified advancement rule.

**Decision:** terminate IRAlu architecture/transcript context as a standalone
performance branch. Retain the nine inexpensive geometry variables only as
prespecified auxiliary inputs to a later substrate-engagement model; any such
combined model must demonstrate its own cross-fitted incremental value. No
positive standalone claim is supported.

Artifacts: `data/derived/features/genomic_context.parquet`,
`experiments/runs/genomic_residual/MuscleSkeletal/`, and
`experiments/runs/genomic_residual_v2/MuscleSkeletal/`.

## Subsequent substrate-level result

The retained geometry hypothesis was later tested at the biological level it
was intended to represent. A nested pair-level model predicts the fraction of
highly edited sites on a substrate in all four tissues, and the information
ablation shows that genomic geometry—not graph-inferable summary statistics or
adenosine count—carries most of that signal. Actual inter-arm genomic gap is
the dominant univariate correlate (Spearman -0.33 to -0.48 across tissues).

This direction and magnitude have strong independent precedent. Bazak et al.,
*Nucleic Acids Research* 2014 (PMID 24829451; DOI 10.1093/nar/gku414), analyzed
203,798 genic Alus and reported distance to the closest reverse-oriented Alu as
the strongest Alu-level editability determinant: average editability decayed
approximately exponentially with a characteristic distance near 800 bp, and
distance alone explained 28% of variance. That result was not used to choose a
window or transform here; ADARpro's prespecified raw and log1p gap variables
were frozen before the pair-level outcome analysis. Its agreement with the
observed negative gap correlation supports a duplex-realization interpretation
without converting the development association into a causal claim.

This does not reverse the negative site-wise result above. It sharpens it:
geometry is not an effective ad hoc correction to individual-site logits, but
is a credible predictor of the shared substrate-engagement state. See
`16_richer_supervision_and_substrate_context.md` for the cross-tissue results
and the direct neural-fusion decision.

Literature source audited 2026-08-22 through PubMed/OpenAlex:
`https://doi.org/10.1093/nar/gku414`.
