# Historical image-generation assets — excluded from the submission build

As of the independent audit dated 2026-09-10, no asset in `selected/` or
`candidates/` is used by `make paper`. The earlier images and their vector traces
remain here solely to preserve provenance. Converting a generated image into
SVG does not remove its image-generation provenance.

The active figures are composed by `scripts/render_paper_figures.py` and
`scripts/paper_vector_diagrams.py`, using quantitative source tables,
code-defined diagrams, and [licensed scientific SVG artwork](../illustration_sources/README.md).
The credited third-party originals are separate from this archive.
All placements in the manuscript are SVG;
their PDF derivatives serve the LaTeX build. Labels use Nimbus Sans, matching
the native PNAS template's sans-serif figure/caption typography.

## Removed image-model-derived components

| Main figure | Historical components removed |
|---|---|
| 1: Search-space overview | RNA substrate, cellular-context, target-local illustrations |
| 2: Benchmark/evaluation | Pair-unit duplex, development-pool motif, frozen-specification illustration |
| 3: Predictive scope | Human tissue strip, Combined/pooled motif, nonhuman lineage illustrations |
| 4: Evidence map | Learning, target-local, substrate/context, and integration illustrations |
| 5: Organizing principles | Duplex input and whole-duplex/target-local representation illustrations |
| 6: Held-out synthesis | Captured, partially resolved, and matched-cell-state tier illustrations |

The SI figures never required these generated components.

## Policy boundary

The [PNAS editorial policy](https://www.pnas.org/author-center/editorial-and-journal-policies#artificial-intelligence),
retrieved 2026-09-10, says: “AI tools for creating images or graphics may not be
used unless the software is the subject of the work under consideration.”
The replaced artwork was not the subject of this research.

The replacement drawing code was also AI-assisted. Removal of image-model
assets is **not** certification of compliance with the broader wording.
Authors must obtain an editorial determination or independently produce the
submitted schematic artwork, and complete the tool/model-version disclosure.
Do not submit the historical generated assets as publication figures.
