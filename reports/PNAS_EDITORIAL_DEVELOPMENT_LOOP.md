# PNAS editorial-development loop

Date: 2026-09-09

## Independent reconstruction

- **Central question:** Which measured information explains endogenous A-to-I editing at the duplex, cellular-state, and nucleotide scales, and where does prediction saturate?
- **Main answer:** Whole-duplex sequence and predicted structure recover intrinsic suitability, substrate geometry marks pair-level editing propensity, and target/helix-partner alignment recovers nucleotide-level information; current bulk and cell-line annotations do not resolve transcript-specific cellular realization.
- **Minimum evidence chain:** pair-disjoint baseline and transfer; pair-level geometry analysis; controlled context tests; pair-first supervision ablation; aligned-versus-shifted target control; saturation tests; single-use canonical held-out validation.
- **Strongest evidence:** the matched-capacity shifted-window control localizes the target-alignment gain; pair-first gains replicate across all endpoints; geometry adds pair-level information in every tissue; the frozen predictor improves all reported canonical endpoints with pair-clustered intervals above zero.
- **Enabling method:** ADARPro operationalizes the selected statistical and representational principles but is not the headline scientific contribution.
- **Unresolved questions:** which donor-, cell-, transcript-, and conformation-resolved measurements account for the remaining error; why cross-lineage transfer varies; whether the information hierarchy transports to an independent cohort and non-Alu substrates.

## Editorial triage

- **One-sentence advance:** Endogenous A-to-I editing exposes separable predictive information at the duplex and nucleotide scales, whereas transcript-specific cellular realization remains beyond the resolution of current bulk measurements.
- **Why a broad reader should care:** The work uses controlled prediction as an empirical assay of biological information, distinguishing what present measurements explain from what requires matched cellular-state data.
- **Unknown before:** The relative predictive contribution and practical limits of substrate organization, bulk cellular context, substrate-aware learning, and target/partner alignment had not been compared within one leakage-controlled framework.
- **Known after:** Intrinsic duplex and aligned target information are recoverable and complementary; static geometry captures pair-level burden; present bulk context does not resolve individual-substrate state; generic model or feature expansion reaches diminishing returns.
- **Contribution type:** A biological information map supported by computation, with an improved predictor as validation and deliverable.
- **Without the model name:** The geometry, context-resolution, transfer, target-alignment, and saturation results still form a coherent map of determinants and limits.
- **Strongest desk-rejection risk:** The study could be read as a benchmark paper without direct perturbational validation. The revision addresses this by centering the empirically bounded information hierarchy, separating prediction from mechanism, and stating the matched-measurement agenda without causal overreach.

## Reviewer-level classification

### A. Acceptance-critical

No unresolved internal-validity problem was identified. Pair isolation, fold-specific fitting, pair-clustered uncertainty, frozen thresholds and checkpoints, single-use canonical evaluation, and the nonindependence of Combined were all made explicit. Independent-cohort and perturbational validation remain scope limitations rather than defects in the reported experiments.

### B. Presentation and framing

- The earlier opening still devoted too much space to benchmark procedure. It was replaced by the biological information hierarchy and shortened methods pointer.
- Substrate geometry and contextual negative results appeared after model-development material. They now precede sampling and representation results.
- The plateau and frozen-validation sections read as optimization history. They were compressed to the evidence needed to establish the empirical boundary and validate the selected principles.
- The Discussion repeated results and caveats. It now synthesizes the information map, explains measurement resolution, states limitations once, and ends with a matched-measurement agenda.

### C. Supporting Information or Limitations

Alternative thresholds, natural-prevalence analyses, matched-recall diagnostics, hand-engineered feature variants, detailed comparator criteria, unavailable H2A.Z data, architecture variants, and resource-specific controls remain in SI. Benchmark scope, structure surrogacy, heterogeneous transfer, repeat-mapping limits, and absence of an independent cohort are consolidated in one Limitations paragraph.

## Iteration record

### Iteration 1

1. **Thesis:** Editing contains separable duplex-, cellular-, and target-scale information; current bulk measurements leave the cellular-state layer unresolved.
2. **Top risks:** model-paper appearance; limited direct mechanism; optimization history obscuring the biology.
3. **Changes:** reorganized Results around scientific questions; moved substrate geometry and context before model refinements; compressed protocol scaffolding.
4. **Moved:** thresholds, fold details, evidence taxonomy, and quarantine implementation to Methods/SI.
5. **Length:** shorter.
6. **Verdict:** Borderline Send to Review.

### Iteration 2

1. **Thesis:** A multiscale information hierarchy defines both recoverable determinants and the present predictive boundary of editing.
2. **Top risks:** benchmark-only interpretation; residual model-first language; excessive development detail.
3. **Changes:** revised title, Significance Statement, abstract, Introduction ending, plateau section, and frozen-validation section.
4. **Moved:** endpoint-by-endpoint metric lists and post-test diagnostic detail to SI.
5. **Length:** shorter.
6. **Verdict:** Send to Review.

### Iteration 3

1. **Thesis:** Duplex and aligned target information are reproducibly recoverable, whereas measured bulk context does not identify transcript-specific substrate state.
2. **Top risks:** predictive association mistaken for mechanism; multiple development comparisons; heterogeneous transfer interpretation.
3. **Changes:** softened causal wording without weakening empirical conclusions; added the multiple-comparison boundary; clarified pooled Combined and the matched-capacity target control.
4. **Moved:** unavailable-resource and secondary diagnostic detail to SI.
5. **Length:** approximately unchanged.
6. **Verdict:** Reviewable but Major Revision.

### Iteration 4

1. **Thesis:** The study maps what is recoverable at substrate and nucleotide scales and identifies transcript-resolved cellular state as the principal missing measurement scale.
2. **Top risks:** lack of independent cohort; restricted substrate scope; unresolved source of cross-lineage transfer loss.
3. **Changes:** shortened the Discussion, sharpened subsection conclusions, removed repeated caveats, and compressed figure captions.
4. **Moved:** engineered-feature contingencies, H2A.Z availability, production provenance, and matched-recall specifics to SI/Methods.
5. **Length:** shorter.
6. **Verdict:** Send to Review.

### Iteration 5

1. **Thesis:** Endogenous editing is governed by separable, recoverable information at the duplex and target scales, while transcript-specific cellular realization remains unresolved by current bulk measurements.
2. **Top risks:** predictive rather than perturbational inference; transportability beyond the benchmark; causally unresolved transfer failures.
3. **Changes:** no further manuscript changes after the stop-rule audit.
4. **Moved:** no additional material.
5. **Length:** unchanged from Iteration 4 and approximately 700 source words shorter than the pre-loop manuscript.
6. **Verdict:** Send to Review.

## Acceptance test

- The title states the scientific scope and boundary without naming ADARPro.
- The Significance Statement and abstract follow problem, systematic interrogation, principal findings, boundary, and implication.
- Figure 1 presents the three information scales and the leakage-safe design.
- Every Results subsection opens with a question or conclusion that advances the central thesis.
- The final Discussion paragraphs distinguish the empirical boundary, scope limitations, practical validation, and next measurement scale.
- Demonstrated findings, predictive inferences, and unknown mechanisms are explicitly separated.
- No evidence, measurement, or quantitative result was added.

## Author-required submission fields

Author names and order, affiliations, contributions, corresponding-author details, funding, acknowledgments, and competing-interest language remain placeholders and require author confirmation before submission.
