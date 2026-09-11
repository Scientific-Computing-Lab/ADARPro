# Final scientific-writing pass

Starting revision: `1a40646942bfa9ee386e84cc5c4cf621108d60d7`.
Date: 2026-09-10 UTC.

## Decisions after reading, before editing

The biological question and evidence hierarchy survive independent rereading.
The principal contribution is the separation of additional substrate-level
measurements, better use of target-local RNA information, and statistical
weighting effects, with tissue-associated variation left unresolved by the
tested contextual proxies. The final predictor tests the selected combination
within the benchmark, not every development-stage attribution.

Read the complete main source and rendered paper, SI text and rendered pages,
all six main and eight SI figures/captions, Tables S1--S10, coverage and claim
ledgers, terminology/epistemic audits, and editorial reports 23, 29, 31, and 32.
Checked the local-input/pooling implementation and retained tissue/within-pair
summaries where interpretations depend on them. Earlier audit conclusions are
not treated as independent evidence.

High-value revisions: distinguish the Significance Statement from the Abstract;
reduce repeated search announcements; improve the transitions between biology,
weighting, representation, and model sufficiency; give modeling tests functional
descriptions; synthesize the Discussion instead of repeating its Results;
clarify standalone captions without redesigning scientific figures. Preserve
first-use outcome/metric definitions and the established biological opening.

Two specific SI issues require correction: a canonical-comparison sentence
incorrectly points to the development figure, and fewer training pairs are
asserted to cause lower OOF performance without a matched sample-size test.
Separate the observed score difference from that untested causal explanation.

## Implemented changes

- Rebuilt the Abstract around the biological puzzle, what overall prediction
  accuracy leaves unanswered, and the comparison of added measurements with
  better use of existing inputs. The Significance Statement now states the
  biological outcomes concretely and explains their relevance to the next
  experiment. Neither requires prior knowledge of AdarEdit.
- Reduced repeated search announcements in the Introduction. Preserved the
  biological opening and fair credit to earlier predictors; made the separate
  role of statistical weighting explicit.
- Reworked the Results transitions from substrate geometry to contextual
  measurements, weighting, local representation, and model sufficiency.
  Functional descriptions now explain the modeling hypotheses before naming
  their implementations. The negative graded-supervision integration result
  explicitly identifies the affected metric and evaluation pool.
- Reorganized the Discussion around different routes to improvement:
  information absent from the RNA input, information already present but better
  represented, and changes in the influence of correlated observations.
  Consolidated residual-error alternatives and retained the external-validity
  and adaptive-development qualifications. The last paragraph specifies how
  matched biological measurements would be judged, without claiming sufficiency.
- Clarified standalone captions, including development versus held-out roles
  and the interpretation of reliability curves. Corrected the SI's canonical
  comparison reference to Table S1, retaining Fig. S1 for development. Removed
  the unsupported causal explanation of lower OOF scores by training-set size.
  Added spacing between crowded columns in Tables S1 and S4 through their
  generator; every table cell is unchanged.
- Aligned Methods, terminology, and reproducibility documentation on adaptive
  selection and exploratory pooled-score screens. Repaired a stale verification
  count: the registry has 128 original entries plus three development-only
  diagnostics added in revision `9fb11dc`. The check now verifies their exact
  identities, evidence types, and lack of canonical-test access. Seven regression
  tests guard this correction; the registry itself was not edited.

The title, scientific ordering, first-use outcome and metric definitions,
quantitative evidence, and all six main/eight SI figure assets were preserved.
No new scientific result, model fit, or canonical-test evaluation was produced.

## Sequential reader and adversarial checks

This is an editorial simulation, not a study of human readers. After each
checkpoint, the revised text supports the following reconstruction without
requiring information introduced later:

| Checkpoint | Reader's usable understanding |
|---|---|
| First page | Editing helps prevent self-RNA immune activation, yet varies among substrates, sites, and tissues. Prediction success does not identify which additional information is useful or what should be measured next. |
| Reference section | The inherited predictor ranks sites well. Its scores are not editing percentages or explained variance; training in one context and testing in another has a descriptive transfer boundary. |
| Context section | Geometry adds pair-level burden information. Matched sites differ among tissues, but tested contextual proxies add no consistent correction. Early residual screens remain exploratory. |
| Target-local section | Weighting changes how observations influence learning; local encoding changes how existing sequence/pairing information is used. The shifted control tests centering of the full neighborhood, not partner identity alone. |
| Final Results | The fixed combination improves prediction on unseen benchmark pairs. It does not independently confirm every adaptive component attribution or establish external-cohort transportability. |
| Discussion | Added measurements, improved representation, and weighting have distinct interpretations. Matched biological measurements are motivated, while the sources of residual error remain unassigned. |

The adversarial check retained the muscle-only shifted control, limited
liver local-component within-duplex replication, pooled Combined endpoint,
noncausal tissue associations, Brier/calibration distinction, and confounded
capacity characterization. None was weakened or promoted to stronger evidence.

## Preservation and verification

- All **398 generated macro definitions**, **127 main/SI quantitative macro
  occurrences**, literal numerical values, mathematical expressions, and
  **67 citation-key occurrences** are unchanged. The bibliography, numerical
  ledger, claim ledger, experiment registry, source data, and frozen model
  artifacts are unchanged. The sole added table callout corrects the SI error.
- Abstract: **213 words**; Significance: **110 words**, using the repository
  checker. Introduction through Discussion decreased from **2,938 to 2,826
  words**, excluding captions and Methods under the same counting rule.
- Both PDFs were rebuilt: **11 main pages**, **16 SI pages**. Every page and
  figure was inspected visually, including reduced page views and the final
  reflowed pages. Two SI column-spacing defects were corrected and rerendered.
  Fonts are embedded; SVG placements and vector-only assets pass verification.
- Artifact/quantitative-source, reference, authoritative-result/freeze/registry,
  PDF-integrity, and page-boundary checks passed. All **140 tests** passed;
  repository-wide Ruff and `git diff --check` passed. Reference verification
  covers **38 DOI records and 34 cited keys**; no new citation was introduced.
- Nonfatal PNAS-class first-page box warnings and microtype/caption compatibility
  warnings remain. Visual inspection and page-boundary checks found no clipping
  or unresolved references. PyTorch Geometric emits existing Python typing
  deprecation warnings during tests; they do not affect these manuscript builds.

## Remaining human completion and scientific scope

Authors must supply names/order, affiliations, contributions, correspondence,
competing interests, funding, and acknowledgments. They must also confirm the
complete AI-tool/version disclosure and resolve PNAS eligibility of AI-assisted
diagram code. Active illustrations use credited licensed vectors and exclude
GPT Image assets; licensing alone does not settle the journal's broader policy.
DOI and volume/article placeholders belong to publisher metadata, not invented
author information.

Editing cannot supply an independent human cohort, fully nested confirmation
of the early contextual screens, or direct matched cellular/structural
measurements. The text states these boundaries without making them repeated
conditions on every positive result. No further general rewrite or marginal
analysis is recommended. **Editorial verdict: Send to Review**, after author
metadata and graphics-policy clearance; this is not an acceptance prediction.
