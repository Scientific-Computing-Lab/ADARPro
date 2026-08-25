# 04 — Cross-tissue switch analysis

## Development-pair atlas

The coordinate and raw-level atlas uses only the 705 pairs belonging to the
canonical train+validation union. It does not expose the 179 held-out pairs.
Every site is mapped by both `(pair_id, local_position)` and
`chr:1-based-position:strand`; all 79,135 canonical development examples agree
with the strand-aware construction formula and the GRCh38 transcript-oriented
focal nucleotide is adenosine at all 46,133 unique canonical sites.

For the matched natural experiment, all raw coverage-qualified sites on those
development pairs were recovered from Artery Tibial, Brain/Cerebellum, Liver,
and Muscle Skeletal. Combined was excluded because it is a pooled endpoint.

- 275,569 site × tissue observations;
- 83,185 unique biological sites;
- 74,758 sites observed in at least two individual tissues;
- 189,809 high-confidence binary observations;
- 85,760 intermediate (1–15%) observations retained for continuous/ordinal work;
- 1,221 sites with a strict switch: ≥15% in at least one tissue and <1% in
  another.

Among the 74,758 matched sites, 3,115 were edited in every observed tissue,
32,432 were non-edited in every observed tissue, 37,990 involved an
intermediate state without a strict edited/non-edited switch, and 1,221 were
strict switchers.

These counts demonstrate that the fixed-intrinsic-state comparison is large
enough to be informative. They do not identify an RBP mechanism: abundance,
binding, mappability, and negative-control analyses remain required.

## Strict-switch direction pilot

Binding features were available for 1,059 of the 1,221 strict switchers (289
Alu pairs), yielding 1,524 edited-versus-nonedited tissue comparisons. A/B
ordering was randomized deterministically; models were selected and evaluated
with the canonical pair folds, and each biological site received total weight
one regardless of how many tissue comparisons it contributed.

The dominant result is a nearly deterministic tissue hierarchy:

- Artery was edited over Muscle Skeletal in 702/702 comparisons and over Liver
  in 134/134;
- Brain was edited over Muscle Skeletal in 396/397;
- Liver was edited over Muscle Skeletal in 190/190;
- the only appreciable reversals were Artery/Brain (3/34 Brain-edited) and
  Brain/Liver (11/67 Liver-edited).

Consequently, tissue identity alone achieved AUROC 0.99914 and weighted
accuracy 0.99323. PrismNet interactions reached AUROC 0.99924, a paired
pair-bootstrap median difference of only +0.00008 (95% CI -0.00006 to
+0.00041), with identical accuracy. eCLIP was effectively tied (+0.00001
median AUROC); RBPNet, the biologically selected suppressor panel, the broad
panel, shuffled expression, permuted RBP identities, negative-control RBPs,
and matched random features did not improve on tissue identity. Binding
without tissue abundance was near chance (AUROC 0.538).

This does not support the claim that the currently operationalized RBP layer
explains strict switch direction. It does show that binding potential alone
cannot determine direction and that the four-tissue binary switch task has a
severe tissue-identity ceiling. The more informative follow-up is continuous
editing-index residual analysis after removing tissue-pair mean effects, plus
focused descriptive analysis of the 15 hierarchy-reversing comparisons.

## Continuous paired EditingIndex analysis

All high, intermediate, and absent editing levels were next retained for the
46,133 annotated canonical sites. This produced 205,199 within-site tissue-pair
differences across 43,197 sites and 705 Alu pairs. Weighted Ridge models used
the same outer/inner pair folds; the baseline learned one mean effect for each
of the six tissue pairs.

The tissue-pair baseline had RMSE 12.535 EditingIndex percentage points, MAE
8.246, R² 0.0704, and Pearson correlation 0.2653. No RBP formulation improved
it. Pair-bootstrap median RMSE changes (negative would be better) were:

- negative-control RBPs: +0.0002 (95% CI -0.0125 to +0.0135);
- eCLIP interactions: +0.0036 (-0.0060 to +0.0186);
- matched random features: +0.0039 (+0.0018 to +0.0058);
- PrismNet interactions: +0.0084 (-0.0094 to +0.0275);
- biologically selected suppressors: +0.0131 (+0.0077 to +0.0193);
- RBPNet interactions: +0.0141 (+0.0007 to +0.0295);
- all RBP modalities: +0.0291 (+0.0042 to +0.0559).

Thus neither the strict-switch nor continuous matched analysis supports
incremental explanatory value for the present eCLIP/PrismNet/RBPNet × bulk
GTEx abundance formulation. Plausible bottlenecks include cell-line/tissue
mismatch, bulk rather than cell-state abundance, eCLIP repeat censorship,
unmeasured protein activity/localization, and binding predictors trained on
CLIP signal rather than ADAR competition.

Artifacts are generated under `data/derived/cross_tissue/`; checksums are in
its `summary.json`. Matched model outputs and clustered bootstrap intervals are
under `experiments/runs/cross_tissue_switch/`.
