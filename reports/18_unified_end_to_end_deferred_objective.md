# 18 — Deferred unified end-to-end ADARPro objective

Recorded: 2026-08-24. Activated after the frozen ensemble campaign completed
and the user selected development-first sequencing. The development experiment
is now complete; see `19_unified_model_preregistration.md` and
`20_unified_end_to_end_result.md`.

## Scope and sequencing

The current prediction-level integration campaign remains unchanged. Complete
the selected equal-weight pair-balanced + target-local logit ensemble, its
validation, frozen training, comparison, canonical evaluation under the
existing protocol, and reporting. Preserve that ensemble as a first-class
result and comparison baseline.

At the first natural decision boundary after the currently committed work has
finished, re-audit the complete development evidence and add a separate
experiment asking whether the technological principles that still have credible
independent or complementary signal can be translated into one new, jointly
trained neural architecture. Do not treat the present list of ideas as
mandatory: retain only principles that survive the final evidence audit, add
later discoveries when justified, and exclude candidates that fail stronger
comparisons.

If the canonical test is still sealed at that boundary, explicitly remind the
user of this objective **before consuming the test** so the unified candidate
can, if chosen, be designed, compared, selected, and frozen entirely on
development data. If the canonical test has already been consumed, its outcomes
must not inform unified-model design or tuning; that candidate then requires a
fresh independent evaluation protocol.

## Model-level integration requirement

The unified candidate must be one connected computational graph whose relevant
trainable components are optimized jointly, with representation-level
interaction before the final prediction and gradients flowing through the
integrated architecture. It may use shared representations, tissue-specific
specialization, substrate-level auxiliary objectives, richer supervision, and
global/focal/partner-aligned modules only where the final development evidence
supports them.

It must be intentionally principled and appropriately low-capacity for the
pair-level effective sample size. It is explicitly not:

- a wrapper around frozen predictors;
- probability stacking;
- late logit averaging;
- a mixture of final predictions; or
- a larger ensemble.

The experiment must include leakage-safe pair-grouped development comparison,
appropriate controls and ablations, and direct comparison with the strongest
prediction-level ensemble of separately trained models. Its question is:

> Can a single jointly trained end-to-end architecture, constructed from the
> technological principles independently validated during the campaign,
> outperform the strongest ensemble of separately trained models?

If it wins rigorously, it is the preferred candidate for the name ADARPro. If
it does not, retain the ensemble as the strongest predictor and report that the
observed complementarity was easier to exploit at prediction level than through
joint representation learning.

## Resolution

The preregistered primary unified model did not outperform the frozen ensemble:
its Muscle OOF AUROC/AUPRC/F1 were 0.9072/0.8922/0.8317 versus
0.9142/0.9040/0.8347, and its Brier score was worse. AUROC and AUPRC decreased
in all five folds and both pair-bootstrap intervals excluded zero in the
adverse direction. The conditional repair and downstream replication gates
were therefore not opened. The frozen prediction-level ensemble remains the
development winner.
