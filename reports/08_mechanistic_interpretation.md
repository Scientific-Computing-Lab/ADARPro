# 08 — Mechanistic interpretation

## Development evidence

The published AdarEdit held-out benchmark remains the authoritative reference.
Model and mechanism selection below used only pair-disjoint development OOF
predictions; the final locked test was consumed only after those decisions were
frozen.

The original generic RBP-occupancy hypothesis is not supported by the available
eCLIP, PrismNet, RBPNet and bulk-GTEx operationalizations. Those features do not
explain within-tissue residuals or matched cross-tissue changes beyond their
controls. This is a negative result about the measurements, not proof that RBP
regulation is absent.

The strongest missing assumption found instead has two parts:

1. **Pair-first supervision.** The canonical balanced subset discards many
   valid high/low observations, and site-wise sampling overweights site-rich
   substrates. Using every eligible training-pair observation while sampling
   pairs uniformly produces significant ranking and F1 gains across Artery,
   Brain, Liver, Muscle and the separately modeled pooled `Combined` endpoint.
2. **Target-centered representation.** The compact GAT marks the target but
   reduces hundreds of nucleotide embeddings to a global mean. A small CNN
   over the target and its helix-aligned partner recovers information that this
   pooling under-resolves. The same CNN centered 96 nt away performs far worse,
   so added capacity or generic local composition does not explain the result.

## False-positive rescue and cost

On the canonical Muscle development rows, target-local fusion corrected 307 of
764 pair-balanced false positives (40.2%) while creating 221 new false
positives. It damaged 160 of 2,574 pair-balanced true positives (6.2%) but
recovered 168 false negatives, yielding eight additional true positives and 86
fewer false positives overall. The fixed pair-plus-local ensemble corrected 302
of the same false positives while creating only 54, at a cost of 103 net true
positives relative to pair-balanced training. Relative to the internal
AdarEdit OOF reference, the ensemble corrected 541 of 942 false positives,
created 115, damaged 206 true positives and recovered 208 false negatives.

At natural prevalence, the ensemble's operating point is especially relevant:
relative to pair-balanced training it removes 4,196 false positives at a cost
of 103 true positives; relative to the internal AdarEdit OOF reference it
removes 7,559 false positives while adding two true positives. These are
development robustness results, not estimates of canonical-test performance.

## What the local encoder appears to capture

Descriptive stratification shows favorable Brier differences for both paired
and unpaired targets and across every prespecified stem-extent bin. The largest
local-model log-loss improvements occur for very short (1–5 nt) and long
(>20 nt) contiguous target stems, but the relationship is not monotonic and
the global local-minus-pair log-loss interval crosses zero. The fixed ensemble
has a robust log-loss improvement (median -0.0598; 95% pair interval -0.0727,
-0.0460) and improves every structural stratum.

The responsible information is therefore not reducible to one hand-coded
paired/unpaired flag or stem-length threshold. A more defensible interpretation
is that target/partner-aligned multi-scale patterns—base identity, interruptions
and helix continuity—must remain spatially resolved until classification. This
is a modeling mechanism, not yet a claim about one molecular regulator.

Saved audit tables and candidate/counterexample records are in
`experiments/runs/target_local_interpretation/MuscleSkeletal/`.

## Final status

The selected ensemble improved canonical-test AUROC, AUPRC and F1 in every
context, with positive pair-bootstrap intervals. At the frozen operating
points it rescued 32.0–50.6% of published-baseline false positives while
damaging 1.7–5.9% of baseline true positives. It also reduced false positives
at exactly matched baseline recall in all five contexts, with every clustered
interval excluding zero. This confirms that the target-centered formulation
transfers beyond development, while remaining evidence for a modeling
mechanism rather than a specific molecular regulator. All post-lock
interpretation is descriptive and cannot reopen development.
