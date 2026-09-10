# 21 — Focused post-analysis plan

## Guardrails

- Use only pair-disjoint development OOF predictions, matched development-site
  tables, and already completed descriptive diagnostics.
- Do not read, subgroup, or rethreshold canonical held-out predictions.
- Treat every new result as post-hoc descriptive evidence unless it reproduces
  an already specified comparison; do not use it for model selection.
- Add nothing to the paper unless the result clarifies a central information-
  scale claim more directly than the current presentation.

## Ranked analyses and stop rules

### 1. Within-duplex discrimination (highest value)

**Scientific value.** Directly tests whether selected representations rank an
edited site above a nonedited site within the same duplex, where substrate
identity is held fixed. This is the cleanest available complement to the
existing between-duplex geometry result.

**Existing artifacts.** Pair-disjoint canonical-development OOF predictions
for the internal AdarEdit reference and expanded pair-first model in all five
endpoints; focal, aligned-local, and fixed-ensemble OOF predictions where
retained (Muscle Skeletal and Liver). No canonical-test artifact is required.

**Estimand.** Restrict to pairs containing at least one positive and one
negative canonical site. Compute each pair's within-pair AUROC (positive-versus-
negative score concordance), average pairs with equal weight, and obtain paired
model contrasts by resampling whole pairs. Report the number and size
distribution of eligible pairs and the distribution of pair-level contrasts,
not only the aggregate mean.

**Stop rule.** Do not promote the result if fewer than 100 mixed-label pairs
are available in a tissue, if the aligned-local or fixed-ensemble contrast
against pair-first has a 95% pair-bootstrap interval spanning zero in either
Muscle Skeletal or Liver, or if the aggregate gain is visibly driven by a small
tail of pairs. A main-text panel is justified only if the result is positive in
both retained local-model tissues and makes the between- versus within-duplex
distinction clearer than the current figures.

### 2. Matched-site tissue variation (high biological value)

**Scientific value.** Positively demonstrates tissue-dependent editing while
site identity is held fixed, complementing the negative result that tested
bulk and cell-line proxies do not resolve the responsible state.

**Existing artifacts.** The 74,758 development sites observed in at least two
individual tissues, raw continuous EditingIndex values, and the existing
strict switch matrix. Combined is excluded.

**Estimand.** For each site, calculate the range and standard deviation of
EditingIndex across observed tissues; summarize with equal site weight. Build
an individual-tissue matrix of strict edited/nonedited switches and disclose
the coverage denominator for each tissue pair.

**Stop rule.** Retain only if the continuous variation and strict-switch counts
form a coherent positive observation after denominators are shown. Promote to
the main paper only if it materially changes the interpretation of the bulk-
proxy negative result; otherwise use one compact SI panel.

### 3. Existing pair-size and structural-strata diagnostics (SI only)

**Scientific value.** Tests whether pair-first gains redistribute performance
toward underweighted small substrates and whether local/ensemble gains persist
across coarse structural strata.

**Existing artifacts.** Completed per-pair pair-balance tables, raw-site-count
quintiles, target-paired/stem-extent/helix-integrity summaries, and pair-level
bootstrap results.

**Inference validity.** These are post-hoc development diagnostics of retained
OOF predictions; no retraining or test access is needed.

**Stop rule.** Add at most two compact SI panels. Omit a panel if it merely
duplicates prose, cannot remain legible at one-column width, or suggests a
confirmatory subgroup claim unsupported by stratum-specific uncertainty.

### 4. Matched-site cross-tissue substrate propensity (optional)

**Scientific value.** Tests whether stable pair-specific editing propensity is
detectable beneath tissue shifts.

**Existing artifacts.** Raw development observations for four individual
tissues with pair and site identifiers.

**Estimand.** For each tissue pair and Alu pair, restrict both tissues to the
same observed sites, then compare mean EditingIndex and high-editing fraction
across Alu pairs. Report Spearman correlation with whole-pair bootstrap
intervals and matched-site/pair counts.

**Stop rule.** Retain only if all six tissue-pair correlations are positive,
their intervals exclude zero, each comparison has at least 100 eligible Alu
pairs, and conclusions agree for continuous mean EditingIndex and binary edited
fraction. Otherwise omit the branch entirely.

### 5. Partner-only perturbation (stopped before execution)

**Scientific value.** Would separate correct target centering from explicit
target-to-predicted-partner correspondence.

**Required artifact and validity.** Valid evaluation requires each development
outer-fold checkpoint so that only its held-out fold is rescored after a
deterministic partner perturbation. The repository retains OOF predictions and
histories, but not the development `best.pth` checkpoints; those paths are
intentionally ignored. Frozen final checkpoints are not substitutes because
validation observations influenced their selection and they do not reproduce
the OOF attribution regime.

**Stop rule.** Do not retrain the five-fold local models and do not use the
final checkpoints for this question. Preserve the narrower claim that the
matched shifted-window experiment supports correctly centered local
information organized by predicted pairing, not a partner-specific effect.
