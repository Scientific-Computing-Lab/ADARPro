# Manuscript terminology audit

Audit date: 2026-09-10 UTC

The paper and SI use the following terms consistently.

| Concept | Preferred term | Boundary enforced in the manuscript |
|---|---|---|
| Search coverage | major biologically defined information scales accessible to the benchmark | ``Systematic'' refers to the explicit substrate, context/realization, target-local, statistical-dependence, and model-sufficiency axes; it never means every determinant in nature was tested. |
| Secondary AI question | model-sufficiency search | Groups richer supervision, cross-context or multitask learning, richer representations, and capacity or integration; conclusions remain conditional on the tested regime. |
| Biological RNA object | substrate or duplex | `pair` is reserved for the benchmark/genomic grouping when the statistical unit matters. |
| Intrinsic RNA property | intrinsic substrate suitability or intrinsic editability | `Intrinsic competence` is avoided in prose because it can sound like a directly measured phenotype. |
| Cluster identifier | Alu pair / pair ID | Confidence intervals resample pair IDs and retain every site from a sampled pair. |
| Development estimator | pair-disjoint development OOF | It is an internal same-fold attribution and selection instrument, not a replacement for published AdarEdit performance. |
| Final evaluation | frozen canonical held-out evaluation | The canonical partition was evaluated once after the final manifest was frozen; this confirms the composite predictor on unseen benchmark pairs. |
| Threshold robustness analysis | post-test descriptive matched-recall analysis | Its threshold was selected on final predictions only to match baseline recall; it is not a frozen confirmatory result. |
| Sampling baseline | expanded row-wise sampling | `site-balanced` is avoided because it can be mistaken for class balancing. |
| Selected graph component | pair-first whole-duplex-only graph predictor | This component contains no local convolutional branch. |
| Selected augmented component | pair-first local-augmented graph predictor | This is a graph model internally fused with an aligned target/helix-partner CNN, not a standalone local CNN. |
| Final integration | fixed 0.5/0.5 equal-logit mean | No ensemble-weight search, stacker, or post-test fitting was used. |
| Local structural context | target-centered local information organized by predicted pairing | `Partner` denotes the dot-bracket-paired nucleotide. The shifted-window control tests correct centering of the full paired neighborhood; it does not isolate the partner nucleotide alone. |
| Negative class | nonedited | This spelling is used in prose; source identifiers retain upstream naming. |
| Tissue name | skeletal muscle | `MuscleSkeletal` remains only in machine-readable identifiers and paths. |
| Uncertainty | pair-clustered bootstrap interval | The pair is the resampling unit; site-wise intervals are not reported. |
| Within-duplex discrimination | equal-pair mean within-pair AUROC | Includes only duplexes containing both labels; each eligible duplex contributes one concordance value regardless of site count. |
| Matched tissue comparison | same-site tissue variation or matched-site tissue comparison | Tissue association is observed while site identity is fixed; tissue identity is not interpreted as a regulator or mechanism. |
| Pooled endpoint | Combined | Combined is one reported benchmark endpoint, not a fifth biological tissue. |
| Pair-level observation | substrate-level editing burden or propensity | Geometry and pair models provide indirect evidence consistent with realization/engagement; neither molecular state was directly measured. |
| Empirical scope | present recoverability boundary or conditional saturation | Applies only to the tested measurements, representations, effective sample size, and evaluation regime; `universal limit` and `ceiling` are avoided. |
| Unresolved context | residual site-level variation | Matched cellular state is a biologically motivated hypothesis, not the uniquely identified source of residual error. |
| Cellular RNA conformation | in vivo RNA structure/conformation | The phrase is not hyphenated in paper prose. |
| Status criterion | recorded selection criterion; predeclared or prespecified only where documented | Individual protocols may have prospective rules; the full adaptive program was not preregistered. `Promotion criterion` is retained only in historical provenance, not paper prose. |

Machine-readable context names, historical experiment IDs, and authoritative
artifact paths are intentionally unchanged because altering them would break
provenance. Their paper-facing labels are standardized at rendering time.
