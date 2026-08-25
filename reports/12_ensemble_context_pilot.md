# 12 — Native-context ensemble folding pilot

## Question and data

AdarEdit receives one isolated-substrate MFE graph. The upstream
`context_folding_results.csv` provides an outcome-independent alternative:
each IRAlu pair was refolded with its natural inter-Alu genomic gap and ±200 nt
flanks, including partition-function thermodynamics. The precomputed table was
joined to development pairs by the original BED row and validated against all
six genomic coordinate/strand fields. It covers 674/705 development pairs and
563/585 Muscle Skeletal development pairs; missing pairs exceed the original
3,500 nt folding cap.

The supplied-checkpoint canonical test remains the authoritative AdarEdit
benchmark. No test label or prediction was used here. Nested pair-grouped OOF
models use only the reduced-training development reference to measure feature
deltas.

## Result

The best block is normalized ensemble thermodynamics (context MFE, ensemble
free energy, ΔΔG, and mean inter-arm probability). Its clustered median deltas
are ΔAUROC +0.0031 (95% CI −0.0053 to +0.0117), ΔAUPRC −0.0008 (−0.0113 to
+0.0097), and ΔF1 +0.0039 (−0.0055 to +0.0137). Fold directions are mixed.

The context inter-arm dsRNA fraction alone increases specificity by a median
+0.0231 (95% CI +0.0120 to +0.0353), but loses AUPRC (median −0.0087) and has
no stable F1 or AUROC gain. The broader structural-fraction block degrades F1;
a matched nine-dimensional random block also degrades.

## Decision

Terminate native-gap/flank-aware ensemble folding as a standalone performance
branch. Its cheap thermodynamic block may be retained as a prespecified
auxiliary variable in a later engagement model, but it has not earned a
positive claim. This result does not falsify genuinely in-vivo DMS/icSHAPE
state, which measures protein- and cell-conditioned structure absent from this
calculation; that higher-cost direction remains contingent on finding a
well-matched, sufficiently covered resource.

Artifacts: `data/derived/features/ensemble_context.parquet`,
`provenance/ensemble_context_features.json`, and
`experiments/runs/ensemble_context_residual/MuscleSkeletal/`.
