# Independent scientific and submission audit

Date: 2026-09-10. Starting revision: `8b539d1`.

## Independent reconstruction, before editing

**Thesis.** An adaptive, scale-organized search locates additional predictive
signal in genomic organization between Alu substrates and in correctly centered
local representations, while the tested context proxies and model extensions
leave variation unresolved at the benchmark's resolution.

The strongest advance is the controlled separation of learning, representation,
and measurement questions. It is not discovery of Alu-distance dependence or
local ADAR selectivity themselves. The title, three biological scales, separate
statistical axis, four model-sufficiency classes, and subordinate role of the
frozen predictor should remain. The search was adaptive, not one prospectively
registered factorial experiment; branch-level preregistrations must not be
generalized to the whole program.

Reasons to send for review: (1) a coherent multiscale question with positive and
negative evidence; (2) label-recovery/weighting and aligned/shifted controls that
distinguish useful explanations; (3) a frozen synthesis with consistent
within-benchmark held-out gains and retained provenance.

Strongest risks: (1) the contextual-screen evidence is weaker than the blanket
nested-evaluation language implied; (2) reuse of one benchmark and the absence
of independent human-cohort prediction leave external validity unresolved;
(3) known biology could be mistaken for a new mechanism if the contribution is
not kept at the level of controlled predictive recoverability.

## Source-level findings requiring correction

- `sample_epoch_indices` in `src/adarpro/models/multitask_oof.py` allocates
  equal class counts, then samples uniformly among pairs eligible for that
  class, then among their sites. The final trainer imports this function.
  It does **not** sample an arbitrary pair, then a class with fallback. Pair
  weights are uniform within class, not necessarily over all substrates.
- `hierarchical_gate/binary_natural` compares the burden gate with expanded
  row-wise supervision (`expanded_reference`), not the later pair-first model.
  The direct auxiliary neural experiment does use the pair-first comparator.
- `residual.fit_cross_fitted` reuses pooled `baseline_oof_logit` features.
  Although each row's first-stage prediction excludes its own pair and each
  second-stage fit is pair-disjoint, first-stage models generating meta-training
  features can include the current meta-test fold. These screens are not fully
  nested end-to-end. Affected families are RBP, genomic, chromatin, native-context
  folding, target geometry, and the initial substrate-profile residual screen.
  Their shared-reference contrasts remain exploratory; no direction or size of
  the resulting bias is assumed. Direct neural comparisons, pair-level geometry,
  matched-site tissue analyses, explicitly outer-isolated later gates/foundation
  fusion, and the frozen final test have distinct implementations.
- The supplied Bio-aware comparison changes features and architecture. It does
  not isolate graph capacity and has no common development OOF effect estimate.
  Graph Transformer/GATv2 was not pursued; that is not a negative experiment.
- Native-gap/flank-aware ensemble folding was completed but omitted from the
  latest search-space coverage table. Add the retained limited/exploratory result.
- The shifted-window experiment is in muscle. Liver repeats the aligned-versus-
  whole-duplex comparison, not the shifted-window control.
- Structural-stratum Brier improvements show that gains are not confined to one
  stratum. They do not rule out the use of a binary pairing feature.
- Development bootstraps used 1,000 or 2,000 replicates, with 5,000 for the
  post-hoc within-duplex analysis; the blanket 2,000-replicate Methods statement
  must be restricted to the final test.
- Brier score measures probability accuracy, combining discrimination and
  calibration. A lower Brier score alone is not proof of better calibration.

These corrections change descriptions and evidence strength, not trained models,
retained predictions, thresholds, data, or numerical results. Historical reports
and the immutable experiment registry remain unchanged; this audit supersedes
their overstrong interpretations where specified.

## Prose and visual decisions

Preserve strong results-forward paragraphs and stable technical names. Reduce
repeated search announcements and future-measurement lists. Academic Humanizer
governs evidence and scholarly register; HPL is used only for locally repetitive
syntax, cadence, and metadiscourse, not informality or detector evasion. Both
complete public rule documents were retrieved and read for this pass.

The consulted rule sources were Academic Humanizer `SKILL.md` at revision
`94b88b23703bed7df507acae7d6d5876209a0cdf` and HPL `PROMPT.md` at
`a8b13dbc610d10d035837970d60c59687390c86a`. Evidence anchoring, stable technical
terms, and calibrated empirical verbs took priority. Sentence variation was
used to separate results, interpretation, and methods—not to introduce casual
language or artificial irregularity. Repeated search announcements and the SI's
duplicate closing summary were compressed; necessary scope qualifications stayed.

All six main and eight SI figures were inspected in the existing rendered PDFs.
Keep quantitative ordering and values. Replace image-model-derived inserts;
remove the decorative tissue banner; correct the benchmark workflow's floating
site markers and invented-looking positive interval; replace truncated bars in
the local comparison with points; enlarge the coverage table; add a legend to
the calibration figure; narrow structural-stratum titles. The conceptual search
anchor and benchmark/evaluation diagram retain different jobs.

## Policy and author action

The official PNAS Editorial and Journal Policies page was retrieved on
2026-09-10 through a text proxy to the publisher page. It states: “AI tools for
creating images or graphics may not be used unless the software is the subject
of the work under consideration.” Disclosure alone does not exempt this paper.
Source: https://www.pnas.org/author-center/editorial-and-journal-policies#artificial-intelligence

Generated-image-derived components in all six main figures were removed
from the manuscript build, not merely retraced. Historical assets stay archived
for provenance. Replacement code-defined diagrams must be reviewed by the
authors; because the policy wording is broad, this audit does not certify that
AI-assisted diagram code is exempt. Obtain editorial confirmation or have an
author/illustrator independently produce the submitted schematic artwork.

Author list/order, affiliations, corresponding author/email/ORCID, contributions,
competing interests, acknowledgments/funding, and the complete AI-tool/version
disclosure require human confirmation. No names or declarations will be invented.

## Scientific work not undertaken

No training, architecture search, canonical-test inference, threshold selection,
or new subgroup analysis is authorized by this audit. A fully nested rerun of
the early residual screens would resolve a genuine methodological limitation,
but requires first-stage refits and is not replaced by cosmetic qualification.
No repository artifact supplies frozen predictor validation in an independent
human cohort; upstream RNAAtlas label concordance is not that validation.

## Final editorial decisions

The intellectual center survives, with a narrower negative-evidence claim:
the study maps **recoverability in specified comparisons**, not biological
absence or a universal limit. Geometry supplies an additional measurement;
target-centered encoding reorganizes already available information. Pair-first
weighting addresses learning influence, not a newly discovered molecular scale.

The selected title, three biological axes, separate model-sufficiency question,
positive pair-level geometry result, local shifted-window control, fixed
integration, held-out values, and major result order were retained. No result,
threshold, model weight, equation, citation identity, or numerical macro was
changed. Main prose excluding figures/captions became about 85 words shorter
under the same source-text counting rule; the SI stayed close to its original
length after removing a repeated closing summary. The abstract and Significance
Statement remain below 250 and 120 words.

Details belong in the SI: the exact sampler, gate comparator, pooled-score
dependency, bootstrap counts, and newly restored native-context/profile search
rows. Main text retains the material evidence boundary once in Results and
at the Methods/Limitations scope. The supplied richer model is no longer
presented as an isolated capacity-negative experiment. Descriptive structural
strata no longer exclude a binary-feature explanation.

### Figure changes and visual checks

1. Search overview: code-defined RNA/context/local motifs; corrected modeling
   status, larger matrix labels, and an explicit exploratory-screen caption.
2. Benchmark: site symbols lie on one RNA substrate; the five-fold matrix is
   nonquantitative; matched comparisons contain no invented effect size. Train
   and validation feed the frozen specification while test pairs bypass fitting.
3. Scope: removed the decorative tissue/lineage banner. Endpoint ordering,
   pooled Combined, matrix values, and human/nonhuman separators are unchanged.
4. Evidence map: corrected capacity status and marked pooled-score screens;
   the integration cue is the actual fixed logit mean. Pair-level plots are
   unchanged.
5. Organizing principles: points replace truncated bars; parallel graph and
   graph-plus-local processing converge through explicit equal logit weights.
   The learning inset states within-class weighting.
6. Held-out synthesis: separate semantic tier motifs replace generated art;
   the F1 legend no longer overlaps the first paired observation. Brier is
   described as probability error, not isolated calibration.

All eight SI figures were inspected; the calibration plot gained a legend,
the contextual-screen title became explicitly exploratory, and the structural
strata title was narrowed. The coverage matrix uses larger type across pages.
The PDFs were checked page by page and figure panels at enlarged and reduced
scales. Native SVG masters and vector PDF derivatives remain reproducible.

### Submission packaging and remaining decisions

The current [PNAS submission instructions](https://www.pnas.org/author-center/submitting-your-manuscript)
allow Research Reports up to 12 pages (6 preferred); the main draft is 10.
They list PDF/EPS/TIFF/PPT figure uploads and instruct LaTeX authors to use
`includegraphics`. This repository deliberately retains the requested SVG
authoring workflow. For the journal upload, use its vector PDF derivatives
and prepare the corresponding `includegraphics` submission copy; SVG alone
is not the documented upload format. Do not confuse this packaging step with
the unresolved AI-graphics eligibility question.

Author metadata/declarations and AI-tool versions remain unfilled. DOI, issue,
and volume placeholders belong to the unpublished journal template; authors
must not invent them. The repeated SI title and duplicate `LastPage` label were
fixed without modifying the pinned upstream class. Template microtype warnings
and first-page internal overfull-box messages remain; rendered page boundaries
are checked separately rather than treating a successful compile as visual proof.

Ranked scientific vulnerabilities:

1. **Contextual-screen inference:** fully nested replication is still needed
   for a strong independently validated negative measurement claim. The current
   paper labels these screens exploratory; the code-level limitation is not
   repaired by that wording. Matched-site RBP tests provide separate support.
2. **External validity:** no zero-tuning prediction in an independently generated
   human cohort, and balanced extreme labels are not population prevalence.
   Do not manufacture proxy validation or delay for a marginal extra analysis.
3. **Novelty and selection scope:** known Alu geometry and local selectivity are
   being separated within an adaptively searched benchmark. Component effects
   are not multiplicity-adjusted confirmations or independent-cohort replications.

**Independent verdict: Major-Revision-Level but Promising.** The paper now
states a coherent, reviewable multiscale predictive contribution. I would not
certify the files for immediate submission until the authors explicitly resolve
the residual-screen claim strategy and graphics policy, and complete metadata
and disclosures. No further architecture search is justified. A decision to
submit the bounded, exploratory-screen account can be defensible; claiming
independently established measurement limits cannot.

## Verification record

- Final PDFs: 10-page main paper and 15-page SI; all scientific figure
  placements are vector-only SVG with vector PDF build derivatives.
- Six main figures and eight SI figures inspected at publication scale;
  enlarged panels and reduced-size views checked. The SI bibliography no
  longer leaves two entries on an otherwise empty final page.
- `make PYTHON=.venv/bin/python paper-check` passed. The final SI-only
  pagination edit was recompiled and `verify_paper_artifacts.py` rerun.
- 126 tests passed; Ruff passed. Existing PyTorch/Python deprecation warnings
  do not arise from this revision.
- 463 numerical claims, 397 generated macros, and all retained results,
  checkpoints, experiment files, and freeze provenance are unchanged from
  the starting commit. Only evidence descriptions and figure presentation changed.
- Static bibliography audit: 35 DOI records, 30 distinct cited keys. Primary
  AdarEdit text and the Yi publisher article were checked; Cai's DOI metadata
  and deposited abstract independently confirm the H2AFZ/SETD2 perturbation
  rationale. This is not a claim to have revalidated every cited experiment.
- `qpdf --check` passed for both PDFs; main-page bounds and font embedding
  were checked. No raster image is embedded in the main manuscript PDF.
- The source-artifact manifest now excludes independently generated table
  files, fixing a stale-hash failure when tables change after figure rendering.
  A guard rejects archived image-model assets in the active figure manifest.
