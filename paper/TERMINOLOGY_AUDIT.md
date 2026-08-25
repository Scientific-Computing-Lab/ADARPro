# Manuscript terminology audit

Audit date: 2026-08-25 UTC

The paper and SI use the following terms consistently.

| Concept | Preferred term | Boundary enforced in the manuscript |
|---|---|---|
| Biological RNA object | substrate or duplex | `pair` is reserved for the benchmark/genomic grouping when the statistical unit matters. |
| Cluster identifier | Alu pair / pair ID | Confidence intervals resample pair IDs and retain every site from a sampled pair. |
| Development estimator | pair-disjoint development OOF | It is an internal same-fold attribution and selection instrument, not a replacement for published AdarEdit performance. |
| Final evaluation | single-use canonical held-out evaluation | The canonical partition was accessed once after the final manifest was frozen. |
| Threshold robustness analysis | post-test descriptive matched-recall analysis | Its threshold was selected on final predictions only to match baseline recall; it is not a frozen confirmatory result. |
| Sampling baseline | expanded row-wise sampling | `site-balanced` is avoided because it can be mistaken for class balancing. |
| Selected graph component | pair-first whole-duplex-only graph predictor | This component contains no local convolutional branch. |
| Selected augmented component | pair-first local-augmented graph predictor | This is a graph model internally fused with an aligned target/helix-partner CNN, not a standalone local CNN. |
| Final integration | fixed 0.5/0.5 equal-logit mean | No ensemble-weight search, stacker, or post-test fitting was used. |
| Local structural context | target/helix-partner | `partner` denotes the dot-bracket-paired nucleotide; abbreviated forms are avoided in paper prose. |
| Negative class | nonedited | This spelling is used in prose; source identifiers retain upstream naming. |
| Tissue name | skeletal muscle | `MuscleSkeletal` remains only in machine-readable identifiers and paths. |
| Uncertainty | pair-clustered bootstrap interval | The pair is the resampling unit; site-wise intervals are not reported. |
| Pooled endpoint | Combined | Combined is one reported benchmark endpoint, not a fifth biological tissue. |
| Pair-level observation | substrate-level editing burden or propensity | Geometry and pair models provide indirect evidence consistent with realization/engagement; neither molecular state was directly measured. |
| Cellular RNA conformation | in vivo RNA structure/conformation | The phrase is not hyphenated in paper prose. |

Machine-readable context names, historical experiment IDs, and authoritative
artifact paths are intentionally unchanged because altering them would break
provenance. Their paper-facing labels are standardized at rendering time.
