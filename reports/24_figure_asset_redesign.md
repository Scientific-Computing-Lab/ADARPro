# Final figure-asset redesign audit

## Scope and decision rule

This pass changed only schematic and conceptual elements in the four affected
main figures. Quantitative marks, source tables, effect sizes, uncertainty
intervals, and panel order were left unchanged. Biological imagery was used
only for nonquantitative concepts; labels, arrows, grouping rules, and model
operations remain vector-rendered by `scripts/render_paper_figures.py`.

## Initial defect inventory

- Figure 1A used a generic ladder-like duplex, a manually drawn target window,
  and a cellular insert whose visual grammar competed with the annotations.
- Figure 1C read as a classroom flowchart: large numbered circles and cards
  carried more visual weight than the leakage-safe partition itself.
- Figure 2A's information strip used generic circles, wavy lines, and arrows
  that decorated the table without adequately identifying the corresponding
  information scale.
- Figure 3E mixed small graph doodles, pill-like logits, and loosely aligned
  connectors; the two branches did not read as strictly parallel computations.
- Figure 4E used tiny programmatic duplex icons and a cellular asset that were
  too weak to support the three-tier synthesis at manuscript scale.

## Asset-generation and curation record

Four new asset families were generated with the deployed Azure GPT Image 1.5
endpoint on native transparent backgrounds. Six candidates were generated for
each RNA-focused family after the first round retained DNA-like silhouettes;
three candidates were generated for each contextual family. The complete
prompts, model settings, file hashes, and candidates are recorded in
`paper/figure_assets/generation_manifest.json`.

| Asset family | Candidates | Retained use | Curation decision |
| --- | ---: | --- | --- |
| `rna_substrate_editorial` | 6 | Figure 1 intrinsic substrate; Figure 3 input; Figure 4 captured tier | Retained the continuous low-profile fold with asymmetric bulges; rejected canonical B-DNA and three-arm silhouettes. |
| `target_pair_context` | 6 | Figure 1 target-local window; Figure 2 target level; Figure 3 local branch | Retained the central amber/teal paired position with open annotation space; rejected twisted-helix and block-token variants. |
| `substrate_organization` | 3 | Figure 2 substrate/context level; Figure 4 partial tier | Retained one continuous transcript forming an imperfect stem; rejected candidates containing a separate DNA-like helix. |
| `matched_cell_state` | 3 | Figure 1 cellular context; Figure 2 context level; Figure 4 unmeasured tier | Retained the RNA-dominant cell/nuclear scene; rejected the candidate containing an explicit DNA helix and the ambiguous icon set. |

Selected assets and their rationales are recorded in
`paper/figure_assets/selection_manifest.json`. Generated text and arrows were
prohibited; all semantic annotation was added later in vector form.

## Vector reconstruction

- **Figure 1A:** replaced the generic duplex and cell insert, embedded the
  curated target-pair asset beneath vector focal/partner leaders and an aligned
  window bracket, balanced the three convergence arrows, and flattened the
  observed-editing endpoint.
- **Figure 1C:** retained the exact train/validation/test counts but replaced
  circular pipeline nodes and boxed cards with a partition-led editorial
  protocol. Each split maps directly to a colored rule and a compact
  develop/freeze/evaluate statement.
- **Figure 2A:** converted the left strip into a row-aligned hierarchy. Pair
  weighting remains a vector concept; target-local, substrate organization,
  and matched cellular context use curated assets; fixed integration remains a
  mathematical vector operation.
- **Figure 3E:** rebuilt the panel on a shared column grid. The pair-first input
  feeds one junction; blue and purple branches terminate on their respective
  representations, proceed through aligned encoders and logit rules, and enter
  the fixed `0.5 z1 + 0.5 z2` operator. Logit pills were removed.
- **Figure 4E:** replaced the tiny generic icons with distinct substrate,
  organization, and cellular-state assets in a dedicated right column. The
  tiers are flat scientific bands with restrained rules and unobstructed text.

## Visual inspection

Every affected figure was inspected at native PNG resolution, by cropped panel
at high zoom, and at 50% scale approximating manuscript reading. A second
render removed the remaining card-like treatment in Figure 1C and added an
explicit cellular cue to Figure 2's substrate/context level. No generated
asset carries a numeric value or depicts an experimental observation.
