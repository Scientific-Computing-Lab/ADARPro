# Main-figure redesign workflow and provenance

## Scope

The five main figures were rebuilt as a single visual argument rather than as
independent performance summaries. Quantitative marks remain programmatic and
trace to the retained source tables. GPT Image 1.5 contributes only isolated,
transparent conceptual assets; it does not generate measurements, plots,
chemical structures, protein structures, or experimental observations.

## Visual reference review

The reference pass inspected the full figures, not only article abstracts.
Six main figures from four recent PNAS papers were compared side by side:

- Bian et al., “Sexual stage-specific A-to-I mRNA editing is mediated by
  tRNA-editing enzymes in fungi,” PNAS (2024),
  [doi:10.1073/pnas.2319235121](https://doi.org/10.1073/pnas.2319235121).
- He et al., “Genome-wide single-cell and single-molecule footprinting of
  transcription factors with deaminase,” PNAS (2024),
  [doi:10.1073/pnas.2423270121](https://doi.org/10.1073/pnas.2423270121).
- Mottram et al., “Pseudouridine prevalence in Kaposi's sarcoma-associated
  herpesvirus transcriptome reveals an essential mechanism for viral
  replication,” PNAS (2025),
  [doi:10.1073/pnas.2508523122](https://doi.org/10.1073/pnas.2508523122).
- Shin et al., “A systems-level atlas of carbon-response transcriptional
  states in *Escherichia coli*,” PNAS (2026),
  [doi:10.1073/pnas.2531884123](https://doi.org/10.1073/pnas.2531884123).

Three recent open-access Nature Communications Figure 1 compositions were
also inspected:

- Verstraten et al., “Defining expansions and perturbations to the RNA
  polymerase III transcriptome and epitranscriptome by modified direct RNA
  nanopore sequencing” (2026),
  [doi:10.1038/s41467-025-68230-1](https://doi.org/10.1038/s41467-025-68230-1).
- “Quantitative mapping of pseudouridines in bacterial RNA” (2026),
  [doi:10.1038/s41467-026-70073-3](https://doi.org/10.1038/s41467-026-70073-3).
- “Pan-organ poly(A) atlas reveals a post-transcriptional regulatory layer
  independent of RNA abundance” (2026),
  [doi:10.1038/s41467-026-71703-6](https://doi.org/10.1038/s41467-026-71703-6).

The reusable design lessons were: let quantitative panels dominate; use a
small biological overview to orient rather than decorate; create one strong
reading path per figure; keep palettes restrained and consistent; use
typographic hierarchy instead of heavy containers; align repeated panels to a
strict grid; and put the limiting interpretation next to the evidence that
constrains it.

## PaperBanana workflow

The local PaperBanana corpus was searched for science-application diagrams.
The closest visual exemplars were the JanusDNA, joint single-cell dynamics,
Orochi biomedical imaging, Prot2Text, small-molecule NMR, and virtual-zebrafish
figures. Their strongest transferable properties were multiscale biological
orientation, a clear left-to-right causal reading path, integration of real
data displays with a restrained schematic, and visual encoding that remains
legible without large prose boxes.

The workflow was applied as follows:

1. **Retriever:** select PNAS, Nature Communications, and PaperBanana examples
   closest to RNA mapping, multiscale biology, and computational inference.
2. **Planner:** assign one biological question to each figure and reserve
   illustration only for context that cannot be communicated by a plot.
3. **Stylist:** standardize typography, thin rules, color semantics, panel
   spacing, organ/lineage motifs, and reduction-size hierarchy.
4. **Critic:** reject clipped anatomy, overdecorated compositions, text-like
   image artifacts, plots implied by illustration, and molecular detail that
   could be mistaken for evidence.

The PaperBanana provider wrapper does not natively accept the supplied Azure
image deployment. Its reference-driven planning and critique stages were used
locally, while the supplied Azure GPT Image 1.5 endpoint generated the raster
components directly.

## Generated asset families and curation

Three candidates were generated for each of five families: RNA duplex,
cellular context, human tissues, marine lineages, and latent/unmeasured
context. The prompts and hashes of all candidates are recorded in
`paper/figure_assets/generation_manifest.json`. Selection decisions and hashes
for the five retained assets are recorded in
`paper/figure_assets/selection_manifest.json`.

The retained components are intentionally label-free. RNA identity, evidence
status, explanatory meaning, and all directional relationships are supplied by
the vector compositor. This prevents generated styling from silently becoming
a scientific claim.

## Figure-specific composition

- **Figure 1:** the curated duplex and nuclear-context assets anchor three
  information scales; vector rules, labels, and convergent arrows express the
  inferential hierarchy and the held-out evaluation timeline.
- **Figure 2:** tissue and marine-lineage strips orient the scope of the
  within-context and transfer matrices. They encode dataset identity only.
- **Figure 3:** a subdued latent-context vignette marks the transition from
  measured inputs to unresolved state beside the complete evidence map; the
  burden plots remain fully vector and data-derived.
- **Figure 4:** the duplex asset enters a vector-rendered two-view architecture.
  Graph nodes, target-aligned positions, logits, arrows, and fixed weights are
  drawn programmatically.
- **Figure 5:** the latent-context asset supports a three-level synthesis of
  captured, partially resolved, and missing information beside the untouched
  held-out quantitative panels.

## Nonfabrication and verification rules

- Generated assets have no numerical axes, values, effect sizes, or sample
  counts.
- Generated proteins are generic silhouettes and cannot be read as solved
  molecular structures.
- Organ and lineage illustrations are contextual identifiers, not phenotypic
  observations.
- All plotted values continue to originate from `paper/source_tables/` and the
  upstream artifacts recorded there.
- PDF and SVG remain the publication outputs; generated PNGs are embedded
  sub-assets beneath editable vector labels and quantitative marks.
- The figure source manifest records hashes for quantitative tables, selected
  generated assets, their provenance manifests, and final PDFs.
