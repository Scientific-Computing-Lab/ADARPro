# Pre-submission editorial and policy audit

Date checked: 2026-09-10

## Scientific and editorial status

The manuscript's central claim remains bounded to information recoverable from
the tested measurements, representations, effective sample size, and evaluation
regime. Development OOF comparisons support component-level attribution; the
canonical held-out test supports only the frozen synthesis on unseen pairs from
the same benchmark construction. No repository artifact provides an external
evaluation of the selected predictor in an independently generated human
cohort. The retained RNAAtlas analysis assesses editing-label concordance only.
Submission should not be delayed to reinterpret that analysis as predictor
validation.

## Required author-supplied fields

The main manuscript still contains explicit placeholders for the author list,
affiliations, lead author, author contributions, competing-interest statement,
corresponding author and email, acknowledgments, and funding. The SI repeats the
author and corresponding-author placeholders. The DOI placeholder is expected
before production but should be handled according to the submission system.

## Generative-image policy

The current PNAS *Editorial and Journal Policies* page states: "AI tools for
creating images or graphics may not be used unless the software is the subject
of the work under consideration." The paper studies A-to-I RNA editing rather
than image-generation software. On the current policy wording, disclosure alone
does not make the generated illustration components eligible for submission.
Policy source:
<https://www.pnas.org/author-center/editorial-and-journal-policies#artificial-intelligence>.

The following main-figure sources contain traced SVG components derived from
GPT Image 1.5 and require replacement with author-created or otherwise
policy-compliant artwork before PNAS submission:

- LaTeX Figure 1, `figure0_search_space.svg`: `rna_substrate_editorial`,
  `search_context_scale`, and `target_local_editorial`.
- LaTeX Figure 3, `figure2_sequence_scope.svg`: `human_contexts`,
  `pooled_human_context`, and `marine_lineages`.
- LaTeX Figure 4, `figure2_evidence_map.svg`: `pair_weighting_editorial`,
  `target_local_editorial`, `substrate_context_editorial`, and
  `whole_duplex_graph_view`.
- LaTeX Figure 5, `figure3_organizing_principles.svg`:
  `rna_substrate_editorial`, `whole_duplex_graph_view`, and
  `target_local_editorial`.
- LaTeX Figure 6, `figure4_locked_validation.svg`: `rna_substrate_editorial`,
  `substrate_organization`, and `matched_cell_state`.

The revised LaTeX Figure 2, `figure1_systematic_dissection.svg`, is fully
programmatic vector artwork and contains no generated sub-asset. The SI figures
are quantitative, programmatically rendered graphics and do not use generated
illustration assets. The complete generation prompts, candidate choices, and
hashes remain in `paper/figure_assets/` and
`paper/source_tables/figure_source_manifest.csv` for provenance.

Before submission, replace the listed components and then remove or update the
GPT Image disclosure in the Data, Materials, and Software Availability section
so that it describes the submitted figure set exactly.

The same PNAS policy requires disclosure of generative-AI use during the
research process in Materials and Methods, or in Acknowledgments if no Methods
section is present. The authors should supply an accurate final disclosure of
any writing or research assistance, including tool names and versions. The
current Data Availability sentence covers figure generation only and is not in
the policy's specified disclosure location.
