# PNAS reconceptualization plan: determinants and limits of A-to-I RNA editing

This plan was completed before revision of the manuscript. It distinguishes the
newer, leakage-controlled ADARPro campaign in this repository from upstream
AdarEdit analyses retained in the adjacent historical repository snapshot. The
latter are admissible as existing evidence, but they must be labeled as
benchmark characterization or scope tests rather than as new ADARPro model
selection evidence.

## 1. PNAS exemplars and the rhetorical model derived from them

Thirteen recent PNAS research articles were read in full through their open
full-text records. They were selected to cover RNA editing and modification,
genome-scale mapping, systems biology, biological machine learning,
generalization, bias-aware evaluation, and empirical limits.

| Exemplar | What the paper actually does | Transferable writing and design lesson |
|---|---|---|
| Bian et al., *Sexual stage-specific A-to-I mRNA editing is mediated by tRNA-editing enzymes in fungi* (2024)[^1] | Moves from isoform discovery through genetics, purified-enzyme assays, and cofactors to a bounded mechanistic conclusion. | Lead with the biological surprise and arrange Results as an evidence ladder. Reserve causal verbs for genetic and biochemical tests; distinguish them from predictive associations. |
| Rosenwasser et al., *ADAR-GPT: A continually fine-tuned language model for predicting A-to-I RNA editing sites* (2026)[^2] | Presents a model-first three-figure story: workflow, ROC, and precision-recall. | This is the nearest topical comparator and a useful countermodel: ADARPro must not repeat its architecture/benchmark framing. The present manuscript should use prediction as an instrument for an information audit. |
| Mottram et al., *Pseudouridine prevalence in Kaposi's sarcoma-associated herpesvirus transcriptome reveals an essential mechanism for viral replication* (2025)[^3] | Progresses from enzyme requirement to transcriptome-wide map, site validation, and focal functional consequence. | A map becomes biologically consequential when global discovery is followed by focused validation and explicit functional limits. Figure order should narrow from landscape to decisive examples. |
| He et al., *Genome-wide single-cell and single-molecule footprinting of transcription factors with deaminase* (2024)[^4] | Introduces a technology through the multiscale biological gap it closes, then demonstrates cell typing, genomic distributions, and cooperativity. | A tool belongs in Figure 1 only as part of the experimental logic; later figures should emphasize the biological quantities the tool reveals. |
| Romano et al., *Perturbation-specific transcriptional mapping for unbiased target elucidation of antibiotics* (2024)[^5] | Builds a reference map, validates it against external compounds, and uses it for prospective mechanism-of-action assignment. | Organize computational evidence as construction, independent validation, application, and boundary. A broad biological bottleneck should frame the Significance Statement, not algorithm novelty. |
| Shang et al., *Predicting the unseen: A diffusion-based debiasing framework for transcriptional response prediction at single-cell resolution* (2025)[^6] | Defines generalization to unseen perturbations, benchmarks a correction, and closes with dataset-dependent limits and biological insights. | State the extrapolation problem before the method, keep benchmarking subordinate to it, and make conditionality of generalization explicit. |
| Yilmaz et al., *Bias-aware training and evaluation of link prediction algorithms in network biology* (2025)[^7] | Shows how edge-uniform evaluation confounds node-level inference, then develops balanced and stratified evaluation and training. | The unit of inference can itself be a scientific result. ADARPro's pair-first finding should be framed as resolving substrate-level dependence, not merely as a sampler tweak. |
| Yang et al., *An interpretable molecular framework for predicting cancer driver missense mutations* (2026)[^8] | Maps 34 molecular features across more than 120,000 variants before introducing a transparent predictor and independent benchmarks. | Put the determinant landscape before the named model. Use coverage, associations, strata, and failure modes to justify the predictor rather than allowing the predictor to define the paper. |
| Visani et al., *T cell receptor specificity landscape revealed through de novo peptide design* (2025)[^9] | Uses a physically motivated neural model to move from affinity prediction to activity, design, and specificity landscapes. | Computational performance is persuasive when each result answers a biological question and is followed by an orthogonal or out-of-domain test. |
| Burghi et al., *Rapid, interpretable data-driven models of neural dynamics using recurrent mechanistic models* (2025)[^10] | Uses measurable biophysical priors to fit neural dynamics while explaining why prediction error cannot uniquely diagnose mechanism, data quality, or missing variables. | Explicitly distinguish predictive adequacy from mechanistic identification. Residual error should be interpreted as a bounded mixture of missing state, measurement error, representation limits, and stochasticity. |
| Wang et al., *Biophysical principles predict fitness of SARS-CoV-2 variants* (2024)[^11] | Maps molecular binding and stability onto fitness, tests held-out variants, and analyzes epistasis and plateau behavior. | A predictive model earns biological meaning through interpretable factorization, held-out stress tests, and saturation analysis. Avoid claiming universality where omitted biological factors remain. |
| de Pins et al., *Rubisco is slow across the tree of life* (2025)[^12] | Systematically screens broad natural diversity, samples promising clades more deeply, reconstructs ancestors, and uses ML to extrapolate the observed ceiling. | A negative ceiling can be the headline discovery. Repeated failure to find exceptional activity after broader and deeper sampling is presented as knowledge, not as an unsuccessful optimization campaign. |
| Shin et al., *A systems-level atlas of carbon-response transcriptional states in Escherichia coli* (2026)[^13] | Builds a 43-condition compendium, identifies modules and response groups, connects them to physiology, and validates them in an independent time course. | An atlas paper should progress from coverage to organizing principles, physiological interpretation, and external validation, ending as a reference framework that generates testable hypotheses. |

### PNAS rhetorical and structural model

1. **Title.** State the biological object and the organizing discovery. Use a
   model name only if the model itself is the discovery. Ceiling or atlas
   papers often make the empirical conclusion the title.
2. **Significance Statement.** In approximately 80--120 words, move through
   problem, unresolved gap, what was systematically compared, what was learned,
   and why the result changes biological understanding or experimental design.
   Avoid architecture names and metric inventories.
3. **Abstract.** Use five moves: biological gap; systematic scope; principal
   positive determinant map; principal boundary/negative evidence; implication.
   The predictor should appear once, as the confirmatory instrument or
   deliverable.
4. **Opening motivation.** Begin with biological selectivity and its
   consequences, then explain why current sequence/structure knowledge is
   incomplete. Introduce prediction only after defining explanation and its
   limits.
5. **Knowledge gap.** Pose a discriminating question: which information scales
   are sufficient, which add independent signal, and where does improvement
   saturate? This is stronger than saying that existing accuracy is imperfect.
6. **Results progression.** Move from a validated baseline and scope to
   successive biological scales, targeted controls, generalization, and a final
   boundary. Each subsection should end with the inference licensed by the
   experiment.
7. **Computation and biology.** Treat models as controlled assays of
   information. Require same-fold comparisons, capacity controls, transfer
   tests, or held-out validation before giving a biological interpretation.
8. **Figure architecture.** Figure 1 should show the biological hierarchy and
   study logic. Later figures should each answer one question. Global maps
   precede focal tests; validation and limitations appear in the same visual
   sequence as the claim they constrain.
9. **Discussion.** Reconstruct the organizing model, separate established
   signal from unresolved state, state why negative tests matter, and end with
   decisive measurements rather than a generic call for more data.
10. **Generality and restraint.** Name the domain of inference: human
    inverted-Alu sites for the controlled ADARPro campaign, with broader scope
    tests treated as supporting evidence. Predictive associations do not prove
    molecular causation; failed representations do not disprove the underlying
    biology.

## 2. Revised scientific thesis

**By treating predictive models as controlled information probes, the study
maps a hierarchy in which duplex sequence and structure encode substantial
intrinsic editability, substrate organization and genomic geometry capture a
second propensity layer, target/helix-partner context resolves local
selectivity, and the remaining ceiling is set chiefly by cellular states that
current bulk measurements do not resolve.**

## 3. Hierarchy of contributions

### Primary scientific contribution

A systematic empirical map of the information that does and does not explain
endogenous A-to-I editing at three nested scales: whole-duplex suitability,
substrate-level realization/propensity, and target-local selection. Controlled
additions, ablations, transfers, and failed integrations delimit the currently
recoverable signal and expose a missing matched cellular-state layer.

### Secondary methodological contribution

Recognition that the RNA duplex is both a biological unit and the effective
statistical unit. Recovering all valid labels and weighting substrates before
sites prevents site-rich duplexes from dominating inference.

### Enabling computational contribution

A target/helix-partner-aligned representation recovers information diluted by
whole-duplex pooling; independently optimized whole-duplex and local-augmented
views combine more reliably than the tested joint architecture.

### Deliverable, not conceptual centerpiece

ADARPro is the frozen equal-logit ensemble that operationalizes these findings
and improves the canonical held-out benchmark. Its performance validates the
information map; the paper is not primarily a report of a new neural network.

## 4. Existing-result-to-question evidence map

| Existing result or experiment | Scientific question answered | Evidence and disposition | Narrative placement |
|---|---|---|---|
| Human GTEx benchmark: four tissues plus pooled Combined; 884 graph-buildable inverted-Alu pairs; 566/139/179 pair-disjoint split | What system and inferential unit are being mapped? | Exact reconstruction; zero pair or cross-tissue leakage | Figure 1 and Methods |
| Supplied AdarEdit checkpoint reproduction | How much is already captured before new information is added? | Human within-context AUROC 0.869--0.933 | Early Results baseline |
| Editing-threshold relaxation | Does the binary model recover a graded editing continuum? | Scores rise monotonically across editing bins; AUROC is stable as the positive cutoff changes | SI robustness; supports graded but incomplete information |
| RNAAtlas independent-cohort concordance | Are GTEx-derived labels reproducible across cohorts and processing pipelines? | Pearson 0.967, Spearman 0.932 across 16,028 sites | Early validation/SI; not a model result |
| Donor-genotype SNP audit | Could germline variation masquerade as editing? | Editing-mimicking variants affect 0.151% of labeled sites and are class-balanced | Early validation/SI |
| GTEx ADAR/ADARB1 tissue-selection audit | Why these tissues, and how heterogeneous is enzyme abundance? | Tissue choices span ADAR-family expression; descriptive only | Methods/SI |
| Triplet sequence--structure baseline | Is a light local structure-string representation sufficient? | Liver AUROC 0.784--0.787 versus 0.909--0.910 for graph models | Figure 2: whole-duplex structure adds substantial information |
| Baseline versus bio-aware GNN across human tissues | Do hand-engineered biochemical features consistently exceed the compact graph? | Bio-aware model is not uniformly better; generic feature enrichment is insufficient | Figure 2 and Discussion boundary |
| Component ablations of the legacy bio-aware model | Which encoded features carry signal? | Neighbor bases, true pairing partners, and pair edges matter most; handcrafted geometry does not help in the single Liver run | Figure 2/SI; descriptive single-seed evidence |
| Substrate structural-stability analysis | Is accuracy confined to rigid hairpins? | 94.6% retain substantial inter-arm pairing in genomic-context folds; discrimination persists in the least-stable quartile but is lower | Figure 2/SI; sequence--structure signal is broad but not uniform |
| Attention probe | Where does the compact graph place predictive weight? | Proximal positions dominate; an attention-only probe generalizes but underperforms the complete graph | SI; associative localization only |
| In-silico mutagenesis | Which local sequence/pairing perturbations most affect predictions? | Strong -1 nucleotide and focal-neighbor pairing effects; counterfactual model behavior, not causal biochemistry | SI; motivates target-aligned tests |
| Human cross-tissue transfer matrix | Which rules generalize between tissues? | Transfer is often strong among human tissues but not uniformly optimal | Figure 2 generalization boundary |
| Legacy joint multi-tissue models and no-Combined control | Does shared supervision recover common rules? | Small/context-dependent benefit; not attributable solely to pooled Combined labels | Figure 2/SI; precursor to stricter OOF transfer test |
| New four-tissue masked model | Does cross-tissue sharing add beyond independently trained pair-first models? | Positive macro ranking/F1 signal below preregistered gates; calibration worsens | Results: partial transfer, not selected |
| Three non-Alu species within-context models | Does sequence--structure information extend beyond human inverted Alu? | Within-species AUROC 0.823--0.865 baseline; bio-aware gains only in octopus | Figure 2 scope |
| Human/species cross-evaluation matrix | Which rules survive lineage and substrate shifts? | Human-to-human and species-to-species transfer exceed many cross-lineage transfers; transfer is asymmetric | Figure 2 generalization boundary |
| Species negative-distance sensitivity | Is nonhuman performance an artifact of negatives near positives? | AUROC remains stable or improves when nearby negatives are removed | SI robustness |
| Seven coding-target scope test | Does an Alu-trained representation rank editing in coding substrates? | Strong results for several genes, but sparse per-gene class counts make this descriptive | SI/scope paragraph; no broad generalization claim |
| OOF baseline and error atlas | Can residual structure be studied without training on the evaluated pair? | Five global pair folds and inner-fold selection provide the attribution instrument | Figure 1/Methods |
| RBP literature and resource survey | Which plausible trans regulators can be tested at benchmark scale? | Prioritized eCLIP, PrismNet, RBPNet, and GTEx abundance panels | Context Results setup |
| RBP occupancy, predicted binding, abundance, interactions, and controls | Does measured RBP context explain residual site behavior? | No reproducible residual improvement; Alu multimapping and unmatched cell/tissue origin constrain inference | Figure 3: measured bulk context boundary |
| Cross-tissue switch analysis | Do binding-by-abundance terms explain tissue-dependent label changes? | Tissue identity nearly separates direction; RBP terms add no transferable mechanism | Figure 3/SI |
| Transcript, repeat-subfamily, divergence, and neighborhood panel | Do broad genomic annotations add beyond the graph? | No stable site-level residual improvement | Figure 3 evidence map |
| Inter-arm gap versus editing burden | Is there a substrate-level propensity distinct from local site selection? | Negative Spearman correlation in all four tissues (-0.33 to -0.48) | Figure 3 positive substrate result |
| Pair-level burden models | Can substrate propensity be predicted beyond graph summaries? | Geometry adds 0.041--0.216 R-squared; absolute R-squared 0.050--0.277 | Figure 3; supported but not mechanistically identified |
| Cross-fitted engagement gate | Does the substrate propensity improve individual-site calls? | AUPRC +0.84 points, F1 interval crosses zero, Brier worsens | Figure 3 boundary |
| Neural engagement auxiliary | Can end-to-end substrate supervision recover the missing state? | Stopped after adverse folds | SI negative result |
| Bulk H3K36me3 overlap and H2A.Z audit | Does cotranscriptional chromatin state add information? | Liver AUPRC-only increment; no stable AUROC/F1/pair benefit; H2A.Z untestable with matched coverage | Figure 3; measured proxy limited, biology not rejected |
| Expanded high/low labels under row-wise sampling | Was useful supervision omitted by the canonical row set? | Independent AUROC and F1 gain | Figure 4 |
| Pair-first sampling with identical label pool | Is the substrate the effective statistical unit? | Further AUROC and F1 gain; replicated across five endpoints with all primary intervals positive | Figure 4; major principle |
| Pair-balance diagnostic | Do site-rich duplexes otherwise dominate optimization? | Confirms unequal row contribution corrected by pair-first sampling | Figure 4/SI |
| Continuous/ordinal/binary multitask model | Do intermediate EditingIndex values add substantial information? | Small ranking/calibration gain, negligible F1, adverse natural-prevalence integration | Figure 3 evidence map/SI; saturation of richer labels |
| Target/contact geometry residual | Is missing information localized near the focal adenosine? | Ranking and calibration improve; F1 does not | Figure 4 evidence ladder |
| Focal-node retention and duplicate-global control | Does whole-duplex pooling dilute the target state? | Focal state improves ranking more than matched duplicate-global capacity; F1 remains limited | Figure 4 evidence ladder |
| Aligned target/helix-partner graph-plus-local model | Does an explicitly aligned local view add independent signal? | Improves AUROC/AUPRC/F1 in muscle and replicates in liver | Figure 4 selected representation |
| 96-nt shifted local control | Is the gain merely extra convolutional capacity? | Large adverse shift; aligned-minus-shifted intervals wholly favorable | Figure 4 decisive control |
| Prediction-level equal-logit integration | Are whole-duplex and aligned-local views complementary? | Positive AUROC/AUPRC/F1 differences in muscle and liver; natural-prevalence gains | Figure 4; enabling integration |
| Relative within-pair rank | Does explicit site rank add beyond pair-first learning? | Helps the weaker original encoder; no value beyond pair-first or shuffled-rank control | SI negative/redundant result |
| Substrate-wide all-A decoder | Can joint decoding exploit within-pair structure better? | Prespecified futility: -6.64 AUROC, -6.73 AUPRC, -5.99 F1 points | Figure 3 boundary/SI |
| Frozen RiNALMo embeddings, 35M and 150M | Do generic RNA foundation representations add transferable signal? | No AUROC/AUPRC gain beyond graph and matched controls; isolated F1 threshold effect fails rule | Figure 3 boundary/SI |
| Inverted-Alu pretraining feasibility | Could task-specific unlabeled pretraining overcome the foundation-model limit? | More than 288,000 candidates identified, but connectivity/foldability/redundancy controls unresolved | Deferred; not evidence of benefit |
| Larger/deeper GNN and proposed Graph Transformer/GATv2 | Does more generic capacity close the gap? | Larger supplied models not consistently better; further expansion superseded by lower-capacity aligned evidence | Figure 3 boundary |
| Preregistered unified end-to-end model | Does joint optimization beat fixed integration? | AUROC/AUPRC worse in every fold; worse Brier; repair/replication gates not opened | Figure 4/SI; negative result defines sample-size/optimization boundary |
| Natural-prevalence development analysis | Do selected views help outside the balanced benchmark? | Ranking, AUPRC, F1, and false-positive counts improve in muscle and liver | SI robustness |
| Frozen one-use canonical comparison | Does the information-derived predictor improve the untouched benchmark? | All five endpoints improve AUROC/AUPRC/F1; mean gains +0.0277/+0.0261/+0.0253; Brier improves throughout | Figure 5 confirmatory validation |
| Frozen-threshold error transitions | Are gains achieved by indiscriminate positive suppression? | False-positive rescue exceeds true-positive damage in all endpoints | Figure 5 |
| Matched-recall rethresholding | What is the descriptive false-positive reduction at equal sensitivity? | 68--158 fewer false positives per endpoint; post-test and explicitly descriptive | Figure 5/SI |
| Donor/cell-resolved state, in-vivo structure, isoforms, and matched perturbations | What information remains missing? | Benchmark-compatible matched measurements unavailable | Final boundary and future-measurement program |

## 5. Title recommendation

**Selected title:** *Mapping the measurable determinants and predictive limits
of endogenous A-to-I RNA editing*

Alternatives:

1. *Systematic dissection reveals the information hierarchy and predictive
   limits of A-to-I RNA editing*
2. *Substrate organization, target context, and the unexplained cellular layer
   of A-to-I RNA editing*
3. *A predictive information map of endogenous A-to-I RNA editing*
4. *What current measurements can—and cannot—explain about A-to-I RNA editing*
5. *Multiscale determinants and the current prediction ceiling of endogenous
   A-to-I RNA editing*

The selected title does not contain ADARPro or a model class. “Measurable” is
retained to avoid implying a complete causal inventory.

## 6. Proposed manuscript outline

1. **Introduction: editing as a multiscale selection problem.** Establish
   duplex formation, substrate availability, and nucleotide selection; define
   prediction as an information assay rather than a synonym for explanation;
   state the missing-information question.
2. **A leakage-safe framework tests information at distinct biological
   scales.** Define the benchmark, units, evaluation regimes, and five classes
   of evidence status.
3. **Sequence and predicted structure capture substantial but non-universal
   editability.** Establish the baseline, show advantage over light local
   encodings, stability-stratified performance, and the tissue/species transfer
   boundary.
4. **Substrate organization exposes a propensity layer not resolved by bulk
   cellular context.** Present gap/burden results first, then engagement,
   RBP/chromatin/transcript annotations, and explain why the negative tests
   identify a measurement-resolution gap.
5. **The RNA duplex is the effective statistical unit.** Separate additional
   labels from pair weighting; end with the replicated pair-first conclusion.
6. **Target/helix-partner alignment recovers local selectivity.** Progress from
   contact geometry to focal state to aligned local encoding and the shifted
   control; distinguish representation evidence from biochemical mechanism.
7. **Additional labels, transfer, foundation models, and end-to-end integration
   approach a plateau.** Use small or adverse results to define the empirical
   boundary rather than narrating an optimization diary.
8. **A frozen predictor validates the map on the untouched benchmark.** Report
   ADARPro once as the operational synthesis; separate confirmatory gains,
   calibration, frozen-threshold transitions, and post-test matched-recall
   description.
9. **Discussion: captured information, unresolved state, and the next decisive
   measurements.** Reconstruct the three-scale map, state the current ceiling,
   bound generality, and specify matched donor/cell measurements needed to test
   the proposed factorization.

## 7. Figure-by-figure redesign specification

All quantitative marks will be generated from retained tables. All lettering,
arrows, legends, and schematics will remain editable SVG/PDF. No generative
image will encode data or molecular structure.

| Figure | Scientific question and panels | Production mode |
|---|---|---|
| Figure 1, “An information hierarchy for endogenous editing” | A, duplex suitability, target selectivity, and cellular realization converging on observed editing; B, controlled scientific questions; C, pair-disjoint discovery/freeze/one-use validation | Vector hierarchy and annotations composed with curated GPT Image 1.5 RNA-duplex and nuclear-context sub-assets |
| Figure 2, “Sequence–structure information is substantial but not universal” | A, within-context human and three-species performance; B, cross-context/cross-lineage transfer matrix; C, transfer groups | Quantitative vector plots with curated human-tissue and marine-lineage context strips; illustrations encode identity only, never values |
| Figure 3, “A substrate-level signal remains unresolved by current context measurements” | A, complete evidence map; B, inter-arm gap versus edited fraction; C, incremental pair-burden R-squared | Quantitative vector map and plots with a curated latent-context vignette to mark information scale |
| Figure 4, “Substrate-aware learning and target alignment recover orthogonal information” | A, label-expansion/pair-weighting decomposition; B, pair-first replication; C, focal-to-aligned representation ladder and shifted control; D, ensemble complementarity; E, compact enabling architecture | Quantitative vector panels and architecture composed around a curated RNA-duplex sub-asset; all model flows and labels remain vector |
| Figure 5, “Frozen validation and the present explanatory boundary” | A, held-out metric deltas; B, absolute F1; C, Brier reduction; D, error transitions; E, final captured-versus-missing information map | Quantitative vector plots plus vector synthesis composed with a curated latent-context sub-asset; matched-recall analysis remains in SI |
| SI figures | Label continuum; RNAAtlas concordance; SNP audit; species negative sensitivity; coding-target scope; attention and mutagenesis; bulk-feature controls; natural prevalence; unified-model negative; ROC/PR and calibration | Existing quantitative assets reindexed and captioned by scientific question |

The PaperBanana reference corpus and Retriever--Planner--Stylist--Critic logic
are used to select visual precedents, plan panel hierarchy, and conduct the
final critique. Its installed provider wrapper does not accept the supplied
Azure deployment directly, so GPT Image 1.5 is called through that deployment
for isolated transparent sub-assets. Three candidates are generated for each
of five visual families, reviewed for biological restraint and reduction-size
legibility, and only selected assets enter the programmatic Matplotlib
composition. Quantitative marks, text, arrows, legends, and panel structure
remain vector and trace to retained source tables. Prompts, candidate hashes,
selection decisions, and output hashes are retained under
`paper/figure_assets/`.

## 8. Claim-editing register

### Strengthen or foreground

- The duplex graph already captures substantial signal across human tissues
  and nonhuman species, but transfer is not uniform.
- The duplex is both a biological substrate and the effective statistical
  sampling unit.
- Inter-arm geometry contributes reproducible pair-level editing-burden
  information beyond graph summaries.
- Target/helix-partner alignment passes a matched-capacity spatial control.
- Multiple plausible information sources fail to add independent signal,
  defining a current empirical ceiling and a measurement agenda.
- The one-use canonical result validates conclusions chosen in development.

### Soften or qualify

- Replace causal “determinants of editing” with “measurable determinants,”
  “predictive information,” or “consistent with” unless perturbational evidence
  exists.
- Treat “substrate realization/engagement” as a latent interpretation of
  burden, never as a directly observed molecular event.
- Restrict RBP and chromatin conclusions to the tested bulk/cell-line proxies.
- Describe coding-target and legacy single-seed analyses as scope tests.
- Describe Combined as a pooled endpoint, never a fifth tissue.
- State that residual error may include missing biology, measurement error,
  finite-sample effects, representation limits, and stochasticity.

### Remove

- Title-level and section-level emphasis on the ADARPro name.
- Optimization-diary language such as “promotion,” “campaign winner,” and
  architecture shopping in the main text.
- Any implication that a failed feature representation disproves the
  corresponding biology.
- Any implication that benchmark AUROC is a fraction of causal variance
  explained.

### Reposition

- Move model architecture after the biological and statistical principles it
  operationalizes.
- Move the full evidence-status inventory to Figure 3/SI; keep the main prose
  question-driven.
- Move post-test matched-recall rethresholding to SI or label it unmistakably
  descriptive.
- Put legacy generalization and robustness evidence before ADARPro-specific
  refinements.

## 9. Verification checklist

- [ ] Every number in the main text, captions, and SI is generated from or
      linked to an existing retained artifact.
- [ ] Legacy/upstream analyses are labeled separately from the newer
      pair-disjoint development campaign and from the one-use canonical test.
- [ ] No development effect is added arithmetically to a canonical score.
- [ ] No test result is used to justify a model choice made after test access.
- [ ] Combined is identified as a pooled endpoint.
- [ ] Pair-clustered intervals retain the pair as the resampling unit.
- [ ] “Explained” is not equated with causal variance explained; AUROC/AUPRC/F1
      are described as predictive discrimination or classification measures.
- [ ] “Engagement” and “realization” remain interpretations, not measurements.
- [ ] Negative results are scoped to their exact measurements, model, and
      resolution.
- [ ] Deferred measurements are never described as negative experiments.
- [ ] Cross-species and coding-target evidence is presented as scope and
      generalization, with sample-size and domain-shift limits.
- [ ] Every figure has a machine-readable source table and an editable vector
      output; raster previews contain no unique information.
- [ ] Generated visual elements, if any, contain no data, molecular structure,
      or experimental observations.
- [ ] Figure captions identify comparator, evaluation regime, uncertainty,
      and descriptive versus confirmatory status.
- [ ] Abstract and Significance Statement contain no unsupported mechanism.
- [ ] Title, Abstract, Results headings, and Discussion do not make ADARPro the
      conceptual center.
- [ ] PNAS word limits, bibliography integrity, references, page bounds, and
      deterministic figure/table generation pass repository checks.
- [ ] Final PDFs are inspected page by page after a clean rebuild.

## Sources

[^1]: Bian Z, Wang Z, Wang D, Xu JR. “Sexual stage-specific A-to-I mRNA editing is mediated by tRNA-editing enzymes in fungi.” *Proceedings of the National Academy of Sciences* (2024). [doi:10.1073/pnas.2319235121](https://doi.org/10.1073/pnas.2319235121); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10962958/).
[^2]: Rosenwasser Z, Cohen-Fultheim R, Levitt M, Levanon EY, Oren G. “ADAR-GPT: A continually fine-tuned language model for predicting A-to-I RNA editing sites.” *Proceedings of the National Academy of Sciences* (2026). [doi:10.1073/pnas.2529073123](https://doi.org/10.1073/pnas.2529073123); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12798952/).
[^3]: Mottram TJ, Harper KL, Vasconcelos EJR, Whitehouse A. “Pseudouridine prevalence in Kaposi's sarcoma-associated herpesvirus transcriptome reveals an essential mechanism for viral replication.” *Proceedings of the National Academy of Sciences* (2025). [doi:10.1073/pnas.2508523122](https://doi.org/10.1073/pnas.2508523122); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12478172/).
[^4]: He R et al. “Genome-wide single-cell and single-molecule footprinting of transcription factors with deaminase.” *Proceedings of the National Academy of Sciences* (2024). [doi:10.1073/pnas.2423270121](https://doi.org/10.1073/pnas.2423270121); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11670102/).
[^5]: Romano KP et al. “Perturbation-specific transcriptional mapping for unbiased target elucidation of antibiotics.” *Proceedings of the National Academy of Sciences* (2024). [doi:10.1073/pnas.2409747121](https://doi.org/10.1073/pnas.2409747121); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11551328/).
[^6]: Shang E, Wei Y, Roeder K. “Predicting the unseen: A diffusion-based debiasing framework for transcriptional response prediction at single-cell resolution.” *Proceedings of the National Academy of Sciences* (2025). [doi:10.1073/pnas.2525268122](https://doi.org/10.1073/pnas.2525268122); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12772209/).
[^7]: Yilmaz S, Yorgancioglu K, Koyuturk M. “Bias-aware training and evaluation of link prediction algorithms in network biology.” *Proceedings of the National Academy of Sciences* (2025). [doi:10.1073/pnas.2416646122](https://doi.org/10.1073/pnas.2416646122); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12184500/).
[^8]: Yang Y, Sun W, Liu Y, Zhang J, Li M. “An interpretable molecular framework for predicting cancer driver missense mutations.” *Proceedings of the National Academy of Sciences* (2026). [doi:10.1073/pnas.2524289123](https://doi.org/10.1073/pnas.2524289123); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12890849/).
[^9]: Visani GM, Pun MN, Minervina AA, Bradley P, Thomas PG, Nourmohammad A. “T cell receptor specificity landscape revealed through de novo peptide design.” *Proceedings of the National Academy of Sciences* (2025). [doi:10.1073/pnas.2504783122](https://doi.org/10.1073/pnas.2504783122); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12557503/).
[^10]: Burghi TB, Ivanova M, Morozova E, Wang H, Marder E, O'Leary T. “Rapid, interpretable data-driven models of neural dynamics using recurrent mechanistic models.” *Proceedings of the National Academy of Sciences* (2025). [doi:10.1073/pnas.2426916122](https://doi.org/10.1073/pnas.2426916122); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12358897/).
[^11]: Wang D, Huot M, Mohanty V, Shakhnovich EI. “Biophysical principles predict fitness of SARS-CoV-2 variants.” *Proceedings of the National Academy of Sciences* (2024). [doi:10.1073/pnas.2314518121](https://doi.org/10.1073/pnas.2314518121); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11161772/).
[^12]: de Pins B et al. “Rubisco is slow across the tree of life.” *Proceedings of the National Academy of Sciences* (2025). [doi:10.1073/pnas.2501433122](https://doi.org/10.1073/pnas.2501433122); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12663927/).
[^13]: Shin J et al. “A systems-level atlas of carbon-response transcriptional states in Escherichia coli.” *Proceedings of the National Academy of Sciences* (2026). [doi:10.1073/pnas.2531884123](https://doi.org/10.1073/pnas.2531884123); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13343008/).
