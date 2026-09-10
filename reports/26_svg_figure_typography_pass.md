# SVG figure and typography pass

## Outcome

All main-text and Supporting Information figures are now placed from `.svg`
sources through `\includesvg`. The SVGs contain paths and live text only: no
`<image>` elements or base64 raster payloads remain. PDF derivatives are created
solely as deterministic `svg.sty` compilation caches.

Figure typography now uses Nimbus Sans, the installed Helvetica-compatible face
used by the PNAS LaTeX class. The rendered PDFs embed Nimbus Sans regular, bold,
and italic variants as required.

## Pipeline changes

- Curated image-generation outputs are traced into editable SVG paths before
  figure composition. Their PNG files remain provenance masters and candidates;
  they are not placed in the manuscript.
- Generated SVG sub-assets replace the temporary raster layout nodes before a
  figure is written.
- Heatmaps and color bars use vector cells instead of rasterized image layers.
- The paper build creates the PDF derivatives expected by `svg.sty` without
  requiring Inkscape or shell escape.
- Artifact verification rejects figure placement through `\includegraphics`,
  embedded raster content in SVG, absent SVG cache files, and figure SVGs that do
  not declare Nimbus Sans.

## Verification

- 13/13 manuscript figure SVGs contain zero `<image>` or `data:image` nodes.
- Main and SI LaTeX sources place figures exclusively with `\includesvg`.
- `pdffonts` confirms Nimbus Sans in every rendered figure PDF.
- The complete main manuscript and SI compile successfully.
- The full test suite passes (126 tests).
