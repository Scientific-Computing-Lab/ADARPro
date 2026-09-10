# Licensed source illustrations used in the paper

These are third-party, attributed scientific illustrations, not image-model
outputs. The unmodified originals are retained under `originals/`. Their licenses
permit adaptation and publication with attribution. They are **not** covered by
the repository's software license.

## Sources and credits

| Original | Creator / publisher | Source and license | Adaptation |
|---|---|---|---|
| `dbcls_pri_mirna.svg` | @hakushijaco; edited by Hiromasa Ono; DBCLS / TogoTV | [pri-miRNA, 2024-04-01](https://doi.org/10.7875/togopic.2024.065), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Generic RNA stem–loop motifs: cropped, recolored, selectively enlarged, with vector site/window annotations. |
| `servier_cell_membrane.svg` | Servier Medical Art | [Bioicons source](https://github.com/duerrsimon/bioicons/blob/d29e766ea7580b8063c4f47b29e872db40a4d979/static/icons/cc-by-3.0/Cell_membrane/Servier/emptycell-3d-2.svg), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | Recolored cell boundary; combined with a nucleus and conceptual RNA motifs. |
| `servier_nucleus.svg` | Servier Medical Art | [Bioicons source](https://github.com/duerrsimon/bioicons/blob/d29e766ea7580b8063c4f47b29e872db40a4d979/static/icons/cc-by-3.0/Intracellular_components/Servier/nucleus-full-3d.svg), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | Recolored nucleus within the cellular-context illustration. |

The DBCLS primary record identifies the designer, editor, original Illustrator
and SVG files, publication date, and CC BY 4.0 license. It was checked directly,
not inferred from the Wikimedia mirror. The Servier SVGs are pinned to a Bioicons
revision and its asset-specific CC BY 3.0 directories; both originated in the
2021 library. Servier's [current terms](https://smart.servier.com/terms-of-use/)
also permit sharing and adaptation of medical images under CC BY 4.0. We retain
the version attached to the downloaded SVGs. The Bioicons website's MIT license
is not the license for these illustrations.

## Scientific meaning

The RNA source was originally drawn to illustrate a primary microRNA. Its
backbone, base shapes, bulges, and stem–loop are adapted here as generic RNA
secondary-structure motifs, **not** as microRNA substrates or specific Alu
conformations. There are no sequence labels, molecular measurements, or claims
about a particular fold. Site colors, the focal-A marker, and the local-window
outline are conceptual annotations, not experimental observations. Cell and
nucleus illustrations identify an unmeasured information scale; they do not
depict measured RNA localization, transport, or regulator engagement.

All adaptations are deterministic in `../../scripts/paper_licensed_art.py` and
`../../scripts/paper_vector_diagrams.py`. Original paths remain vectors; they
are neither rasterized nor traced. Figure labels use the PNAS template's native
sans-serif family. Final SVGs embed attribution and include no external files,
scripts, bitmap payloads, or inherited source labels.

## Figure mapping

- Figure 1A: whole-substrate RNA, target-local enlargement, cell/transcript context.
- Figure 2A: RNA pair-unit illustration and conceptual label markers.
- Figure 4A: target-local evidence-category illustration.
- Figure 5E: whole-duplex and target-local representation cues.
- Figure 6E: recoverable RNA signal and unmeasured cellular-context tiers.

The statistical-weighting, genomic-interval, integration, and evaluation diagrams
remain code-defined: replacing their mathematical or genomic meaning with stock
art would reduce precision. Quantitative panels are unchanged.

## Disclosure and journal policy

No generative-image model was used in this revision. The original artwork has
named source creators; adaptation and composition code was assisted by Codex.
This distinction does not certify that the complete figures are exempt from
PNAS's broader AI-tools policy. Authors must confirm journal eligibility and the
complete disclosure before submission. Historical GPT Image illustrations
remain excluded from the active build.
