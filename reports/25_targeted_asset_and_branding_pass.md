# Targeted figure-asset and model-branding pass

## Scope

This pass addressed only the remaining weak visual inserts and the manuscript's
product-like use of the model name. Quantitative panels, retained values, model
identities in machine-readable artifacts, and scientific claims were unchanged.

## Asset generation and curation

GPT Image 1.5 generated 33 candidates across six concepts: pooled human tissue,
whole-duplex graph view, pair-first weighting, target-local context,
substrate/cellular context, and complementary global/local views. Candidate
selection was based on scientific legibility at full and reduced size. Outputs
that resembled canonical DNA helices, glossy molecular tokens, balance-scale
clip art, or decorative cellular scenery were rejected. The selected assets and
their prompts, hashes, and selection rationales are recorded in
`paper/figure_assets/generation_manifest.json` and
`paper/figure_assets/selection_manifest.json`.

## Figure changes

- **Figure 1C:** Replaced the segmented classroom-style split bar with one
  pair-level source, a single allocation junction, and three parallel evidence
  lanes. Train, validation, and test now map directly to develop, lock, and
  evaluate with consistent arrows and complete-pair counts.
- **Figure 2 header:** Replaced the small sigma badge for Combined with a curated
  four-tissue pooled endpoint asset and matched its scale and baseline to the
  neighboring tissue illustrations.
- **Figure 3A:** Replaced the four placeholder mini-diagrams with distinct,
  curated information-bearing assets for pair-first learning, target-local
  representation, substrate/cellular context, and complementary integration.
  Transparent content trimming improves their legibility without enlarging the
  evidence-map column.
- **Figure 4E:** Replaced the generic node cloud and local duplex insert with
  curated whole-duplex graph and target-local RNA views. Rebuilt both encoder
  modules as clipped scientific process nodes with internal layer rails,
  embedded each logit in the pipeline, labeled the fixed 0.5 branch weights,
  and routed both branches into one compact summation operator with an explicit
  equation and output score.
- **Figure 5:** Retained the already resolved tier-specific synthesis assets;
  only the final-predictor display terminology changed.

All generated components remain label-free and nonquantitative. Labels,
connectors, equations, panel structure, and data graphics are vector-rendered.

## Manuscript terminology

The model name was removed from the Abstract, Results, main captions,
Supporting Information prose and captions, display legends, and evidence
audits. The study is introduced descriptively as a multiscale A-to-I framework;
the final system is called the selected predictor or frozen two-view predictor.
The repository URL and internal artifact/model identifiers remain unchanged for
reproducibility.

## Visual audit

The revised figures were inspected as complete high-resolution renders, as
panel crops, and in a reduced five-figure contact sheet. The audit checked
arrow termination, junction ownership, icon scale, optical baselines, generated
asset semantics, label clearance, encoder alignment, mathematical flow, and
legibility after reduction.
