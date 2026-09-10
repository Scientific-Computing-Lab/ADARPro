# 22 — Focused post-analysis decisions

## Scope and evidence status

All new computations used pair-disjoint development OOF predictions or matched
raw observations from development pairs. The canonical held-out test was not
read, subgrouped, rethresholded, or used for selection. Every retained result
is labeled post-hoc and descriptive.

## 1. Within-duplex decomposition

**Decision: informative, SI only.** The analysis included only duplexes with at
least one edited and one nonedited canonical site, calculated one within-pair
AUROC per model, weighted pairs equally, and resampled whole pairs.

- The eligible set contained 414–646 mixed-label pairs per endpoint.
- Expanded pair-first prediction improved mean within-pair AUROC over the
  internal AdarEdit reference in all five endpoints; every pair-bootstrap
  interval was above zero.
- The fixed ensemble improved mean within-pair AUROC over pair-first prediction
  in muscle by 0.0178 (95% interval 0.0121–0.0240) and in liver by 0.0138
  (0.0106–0.0170).
- The aligned-local component alone improved muscle by 0.0141
  (0.0049–0.0234), but its liver contrast was 0.0018
  (-0.0047–0.0081).

The fixed ensemble therefore improves within-substrate ranking in both retained
contexts, but the local component does not independently replicate that
specific decomposition in liver. This misses the prewritten main-figure stop
rule and is presented transparently in SI Figure S8A rather than promoted as a
new headline result.

## 2. Partner-only perturbation

**Decision: stopped before execution; retain narrower wording.** A valid
perturbation requires each outer-fold development checkpoint so that only the
corresponding held-out fold is rescored. The repository retains the OOF
predictions and histories but not those ignored `best.pth` checkpoints. The
frozen final checkpoints are not valid substitutes because validation data
influenced their selection and they do not recreate the OOF attribution
regime. Retraining would reopen the modeling campaign.

The manuscript therefore continues to claim only that correctly target-centered
local information organized by predicted pairing matters beyond matched generic
capacity. It does not claim that the predicted partner alone is uniquely or
causally responsible.

## 3. Matched-site tissue context

**Decision: materially strengthens the biological story.** Among 74,758 sites
observed in at least two tissues, 1,221 undergo a strict edited/nonedited switch.
Among 50,162 sites observed in all four tissues, 54.6%, 32.0%, and 17.4% span at
least 1, 5, and 15 EditingIndex percentage points, respectively. This is a
direct positive observation of tissue-associated variation with site identity
held fixed.

The optional matched-site substrate analysis also passed its retention rule.
Across all six tissue pairs, pair-level rank correlations were 0.727–0.921 for
mean EditingIndex and 0.633–0.894 for high-editing fraction; all whole-pair
bootstrap intervals excluded zero. These results support stable substrate
rankings overlaid by tissue shifts, without identifying tissue identity or any
bulk proxy as a mechanism. They are shown in SI Figure S7 and summarized in one
main-text paragraph.

## 4. Existing diagnostics promoted to SI

**Decision: include one compact robustness figure.** SI Figure S8 combines:

- the within-duplex analysis;
- the completed pair-size audit, in which pair-first learning most reduces log
  loss for low-site-count substrates and the two largest quintiles favor
  row-wise sampling; and
- the completed target-structure stratification, in which local and ensemble
  Brier differences are favorable for paired and unpaired targets and every
  target-stem extent.

The latter panels are explicitly descriptive and do not make confirmatory
subgroup claims.

## 5. Further analysis

**Decision: no further analysis is scientifically justified from the retained
artifacts.** Additional model perturbation would require missing development
checkpoints or retraining, and further subgrouping would add post-hoc complexity
without closing a central evidential gap. Independent human-cohort
transportability remains the important evidence gap and cannot be solved by
reanalysis of this benchmark.

## Strict verdict

**One or more analyses materially strengthen the main paper.** The matched-site
tissue analysis supplies the missing positive evidence that context changes
editing at fixed sites, while the stable pair rankings sharpen the multiscale
interpretation. The within-duplex and existing diagnostic analyses add useful,
properly bounded SI support but do not justify a new main figure.
