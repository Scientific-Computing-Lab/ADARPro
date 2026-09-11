# Final multilingual semantic and submission-writing audit

Completed 2026-09-11 UTC. Starting repository revision:
`1a40646942bfa9ee386e84cc5c4cf621108d60d7`.
This report completes the editorial work in [report 33](33_final_editorial_pass.md)
and gives the counts for the final PDFs. No scientific analysis was rerun.

## Thesis and editorial method

**The comparisons separate substrate-level information added by genomic
organization, target-local information recovered from existing RNA inputs, and
statistical weighting gains, while leaving tissue-associated variation unresolved
by the tested contextual measurements.**

The complete main paper, SI, figures, captions, and supporting audits were read
before the initial revisions recorded in report 33. The multilingual pass then
tested 48 semantic units covering the front matter, Introduction, Results
questions and transitions, Discussion, difficult Methods/SI passages, and caption
interpretations. Each unit was rendered separately into Hebrew, German, Japanese,
and Spanish. English was reconstructed from those drafts without displaying the
source passage during reconstruction. Quantitative tokens were checked separately
rather than translated stylistically.

The records are [language drafts](editorial_checks/34_language_routes.md),
[English reconstructions](editorial_checks/34_english_reconstructions.md), and
[rechecks of revised anchors](editorial_checks/34_revision_recheck.md).
These are meaning translations and one editor's diagnostic routes, **not four
independent reviewers, native-speaker validation, or a human-reader experiment**.
Translation-specific mistakes were corrected in the drafts; they did not justify
changing otherwise clear English. Wording was revised only when comparison with
the source and retained evidence identified a real ambiguity.

## Changes that matter

- Made the biological subject explicit: **editing at the same genomic site**
  varies across tissues. The site itself does not change. Cross-tissue substrate
  stability now refers to **correlated rankings**, not constant editing levels.
- Sharpened the reason for the study in the Abstract and Introduction. Overall
  predictive accuracy does not identify which gains come from additional
  measurements and which from better use of existing RNA inputs. These are
  tested routes, not a claim that the experiments establish the necessity of
  new measurements or the sufficiency of a future assay set.
- Specified the outcome and comparator behind the contextual negatives:
  **site-level prediction**, and, for the separate matched-site analyses,
  references using **tissue identity alone**. Checked this wording against
  report 04 and the switch-direction/continuous-difference implementations.
  The tissue-pair mean reference uses tissue identities, not a measured cellular
  state. Exploratory pooled-score dependencies remain explicit.
- Replaced ambiguous metric language with **reduced Brier error, without a
  clear F1 gain**. This preserves a small nonzero F1 estimate with an interval
  spanning zero; it does not turn that finding into an exact null or confuse
  probability error with calibration alone.
- Explained training and validation as the roles of distinct pair sets.
  Replaced “confounded feature/capacity changes” with the concrete fact that
  those variants changed features and architecture together. The SI now asks
  whether local retention improves on whole-duplex pooling, without implying
  that the original graph lacked a target marker.
- Retained the earlier whole-paper improvements in report 33: a distinct
  Significance Statement, motivated transitions between biology and statistics,
  functional descriptions of modeling hypotheses, and a Discussion that
  separates the scientific meanings of different gains. No new catalogue,
  repeated caveats, or branded model framing was added.
- During PDF inspection, prevented AUROC/AUPRC from being hyphenated, including
  an AUROC split across columns. The coordinated table-spacing and SI-reference
  corrections from report 33 remain; no figure or table value changed.

## Sequential reader and specialist recheck

On rereading in order, the first page explains why predicting editing is not
enough to choose the next biological measurement. The reference section defines
class probabilities and ranking before their interpretation is needed. Geometry
then concerns the fraction of eligible edited sites **between duplexes**;
matched-site comparisons concern editing differences **between tissues**.
Weighting changes learning influence, whereas target-local encoding reorganizes
information already supplied to the predictor. The final test concerns their
frozen combination on unseen benchmark pairs. The Discussion returns to the
unresolved measurement question without assigning all error to cellular state.

The specialist recheck preserved the noncausal interpretation of tissue and
geometry associations, muscle-only shifted-window control, weaker liver
local-component within-duplex replication, class-conditional pair sampling,
pooled Combined endpoint, and conditional model-sufficiency conclusion.
The test does not independently confirm every adaptive component attribution
or establish transportability to an independent human cohort. These meanings
also survived the language-route reconstructions; no further rewrite was
justified merely by stylistic variation among translations.

## Final verification

- All **398 generated macro definitions**, **127 quantitative macro uses**,
  **67 citation-key occurrences**, literal numerical claims, and mathematical
  expressions are preserved relative to the starting revision. Bibliography,
  numerical/claim ledgers, experiment registry, source data, and frozen models
  are unchanged. The SI's added Table S1 callout corrects its old reference to
  a development figure; multilingual edits changed no further references.
- **Abstract: 217 words. Significance: 103 words.** Introduction through
  Discussion: **2,938 to 2,846 words**, excluding captions and Methods under the
  same repository counting rule. Clarity, not minimum length, governed edits.
- Both PDFs were compiled and inspected: **11 main pages, 16 SI pages**.
  All pages were reviewed during the complete pass; changed pages were checked
  again after reflow. Figures remain vector-only SVG placements. Fonts are
  embedded and no page-edge clipping or unresolved references was found.
- Artifact/source/claim, reference, authoritative-result/freeze/registry, PDF
  integrity, and page-boundary checks passed. **140 tests passed**; Ruff passed.
  Reference checks cover **38 DOI records and 34 cited keys**. Existing PNAS
  class box/compatibility and Python typing warnings remain as documented in
  report 33; successful compilation was not substituted for visual inspection.

## Human completion and remaining scope

Authors must complete names/order, affiliations, contributions, correspondence,
competing interests, acknowledgments, and funding. They must confirm the complete
AI-tool/version disclosure and journal eligibility of AI-assisted diagram code.
The active biological illustrations are credited licensed SVGs; GPT Image
assets remain excluded. Publisher DOI/volume/article metadata must not be invented.

An independent human cohort, fully nested confirmation of the early contextual
screens, and matched cellular/structural measurements are genuine evidence gaps,
not prose defects. No new test evaluation, training, or marginal analysis was
performed to obscure them. **Verdict: Send to Review after author completion and
graphics-policy clearance.** Further general rewriting would now mostly create
stylistic churn.
