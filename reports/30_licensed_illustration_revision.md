# Licensed scientific-illustration revision — 2026-09-10

## Scope

Replace weak biological figure inserts with credited, reusable scientific SVG
artwork. No image-generation model, new analysis, model fit, inference, or
canonical-test access was used. The scientific narrative, numerical claims,
captions' conclusions, and quantitative panels were retained.

## Curation and licensing

Selected three original SVGs, documented in
[`paper/illustration_sources/README.md`](../paper/illustration_sources/README.md):

- DBCLS/TogoTV's *pri-miRNA* structural illustration, designed by @hakushijaco
  and edited by Hiromasa Ono, DOI 10.7875/togopic.2024.065, CC BY 4.0. The primary
  record was inspected to verify the designer, editor, date, and license.
- Servier Medical Art's cell-membrane and nucleus illustrations, pinned to
  Bioicons revision `d29e766ea7580b8063c4f47b29e872db40a4d979`, under the source
  files' CC BY 3.0 license. The membrane file's history dates to April 2021.
  Servier's current terms also permit sharing and modification of its medical
  illustrations under CC BY 4.0; the downloaded files retain their stated license.

Candidates inspected but rejected included older DBCLS circle-and-line RNA
motifs (no visual improvement), the short miRNA duplex (less appropriate
overhang/duplex geometry), Servier's single-stranded RNA ribbon (not a paired
substrate), a complete labeled cell, and SwissBioPics' animal cell (unnecessary
organellar detail and density at the available size). tRNA, DNA, translation,
and regulatory-complex imagery was not repurposed as editing biology.

The selected DBCLS artwork contributes generic backbone, base, bulge, and
stem–loop shapes. Its use does not turn this into a microRNA study or depict a
specific Alu structure. The local view enlarges an existing paired region;
vector annotations identify a conceptual focal A and predicted partner. Cell
compartments contain generic RNA motifs without transport arrows, regulators,
or implied measured localization.

## Figure-level changes

| Main figure | Change and purpose |
|---|---|
| 1, search overview | Replaced all three biological inserts with source-derived RNA, compartment, and enlarged local-pairing artwork. Gave the biological row more vertical space; separated the protocol labels from the evidence-table headings. The RNA window now shows base shapes and a distinguishable focal/partner pair. |
| 2, benchmark/evaluation | Replaced the generic duplex with the RNA stem–loop illustration. Preserved 5′/3′ orientation and placed edited/nonedited/intermediate markers on its backbone. Pair counts, label thresholds, and evaluation logic did not change. |
| 3, predictive scope | Left unchanged: no biological artwork is needed to explain these quantitative comparisons. |
| 4, evidence map | Replaced the target-local mini-panel with an enlarged paired-region illustration and focal-A highlight. Retained the weighting, genomic-interval, and arithmetic symbols because those encode the tested quantities directly. |
| 5, organizing principles | Used the complete RNA motif for whole-duplex processing and a crop for the local representation. Distributed graph markers across the stem and routed the narrower local branch to the illustration's actual boundary. Computations, independent training, and the fixed equal-logit mean are unchanged. |
| 6, held-out synthesis | Replaced the RNA and unmeasured-cellular-context tier inserts. Each remains confined to its own tier, with text unobstructed. The genomic-gap symbol and every quantitative panel were retained. |

## Visual iteration

Initial renders were inspected as complete figures and as enlarged panels.
Reduced images at approximately 950–980 pixels across were checked alongside
the compiled manuscript pages. Subsequent corrections included:

- Enlarging the overview's cell and local assets; improving caption padding.
- Removing an RNA-enlargement leader that could resemble RNA leaving the cell.
- Moving the cytoplasmic RNA wholly inside the cell boundary.
- Keeping the focal-A and partner labels distinct from the window brackets.
- Spreading graph markers across the duplex rather than clustering them near
  one target region.
- Adjusting architecture connectors to terminate at the actual illustration.

## Reproducibility and checks

- Original SVGs are pinned by SHA-256 and preserved unmodified.
- `scripts/paper_licensed_art.py` inlines source styles, adapts palettes/crops,
  namespaces clip-path identifiers, and embeds original vector paths. There is
  no bitmap proxy, raster tracing, external resource, or generated-image input.
- Standalone figure SVGs contain attribution; the SI includes source credits,
  licenses, modifications, and semantic limits. Main Methods retains the
  AI-assisted-code disclosure and identifies the licensed illustration sources.
- Figure labels remain editable SVG text in Nimbus Sans, matching the PNAS
  template's sans-serif figure/caption typography. LaTeX places SVG masters and
  uses the corresponding vector PDF derivatives.
- All 133 tests passed, including seven new licensed-art integrity tests;
  whole-repository Ruff checks passed. `make paper-check` verified references,
  PNAS field limits, claims, source hashes, vector-only placements, compilation,
  and main-paper page bounds. Both PDFs passed `qpdf --check`.
- The numerical ledger, generated quantitative macros, model and experiment
  artifacts, and every quantitative source table are unchanged. Figure 3 and
  all eight SI SVGs remain byte-identical. PDF derivatives were rebuilt.
- Existing template-level microtype/box warnings remain; no undefined citation,
  unresolved reference, or new figure clipping was found.

## Remaining policy boundary

These are licensed third-party illustrations, not GPT Image outputs. Their
adaptation and composition code was AI-assisted. This revision does not certify
that the completed figures are exempt from PNAS's broader policy on AI tools
for graphics. Author/editorial confirmation and complete tool disclosure remain
necessary; historical generated artwork is still excluded. The scientific and
author-metadata limitations recorded in report 29 were not re-reviewed or
changed by this artwork-only pass.
