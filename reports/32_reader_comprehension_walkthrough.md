# Sequential reader-comprehension walkthrough

Starting revision: `cd0bdac67c9d878fa4b31b5219a4c588fa6b1308`

Date: 2026-09-10 UTC

## Reader and procedure

The simulated reader is a biologist in an adjacent field. They know RNA,
gene expression, correlations, and confidence intervals. They have not read
AdarEdit, do not know this benchmark, and are not specialists in graph learning,
classifier evaluation, or the RNA-binding-protein resources. The goal is not
to remove technical terms but to supply their meaning when the argument first
depends on them.

This is an editorial simulation, not a study with human participants and not
an independent blind review. The questions below are constructed predictions
of reading difficulty, not quotations from a real reader. Later explanations
are not credited to an earlier checkpoint. After revision, the audit asks
whether each identified obstacle has an explicit textual remedy; that is not
an empirical comprehension score.

Before editing, the complete main text, captions, Methods, SI, coverage table,
and rendered PDFs were read. All six main figures were inspected at manuscript
scale. The printed figure numbers below are 1--6, not the historical filenames
that begin with `figure0`.

States: **C** = enough information to follow; **P** = partially clear, with a
question that may reasonably remain open; **B** = an avoidable obstacle or
plausible wrong interpretation. A scientific uncertainty is not automatically
a writing defect. Page references below identify the starting PDF.

## Pass 1: main-paper reading log, before editing

| ID / passage | State | What the reader can understand so far | Remaining question or likely wrong interpretation; editorial decision |
|---|---|---|---|
| T / title, p1 | P | The study concerns the sources of predictive information in endogenous editing. | The title cannot explain the biology on its own. Keep the user-selected title and let the opening do that work. |
| A1 / Abstract, problem and gap | C | Editing relates to self-RNA tolerance; neighboring sites and tissues differ; prior predictors do not settle the contribution of each scale. | No need for another motivation rewrite. |
| A2 / Abstract, approach and findings | P | Measurements, representations, and learning were compared; different changes helped differently. | “Frozen two-view” and “endpoints” are unexplained project vocabulary. Use a plain description of the combined predictor and evaluation datasets. |
| S / Significance | P | Biological stakes and the broad comparison are apparent. | “What information distinguishes whole RNA duplexes” could mean recognizing RNA identity rather than editing outcomes. State the distinction in terms of editing. |
| I1 / Introduction, biological setting, p1 | C | ADAR chemistry, self-RNA tolerance, Alu duplex formation, and heterogeneous editing are now introduced. | Strong existing paragraph; preserve. |
| I2 / Introduction, prior predictors, p1 | C | What earlier models did and why another comparative study is justified; AdarEdit's graph is explained. | Preserve the prior-work credit and the distinction from an accuracy-only claim. |
| I3 / Introduction, three scales, p2 | C | A duplex's overall editing and choice of an individual site are different questions; a static RNA description can miss tissue differences. | Useful explanation; do not replace it with a taxonomy list. |
| I4 / Introduction, controlled prediction, p2 | P | Adding an input, reorganizing it, and changing observation weights answer different questions. | What exactly is a “pair-disjoint” comparison and why does it matter? Explain the grouping in the benchmark paragraph, before numerical results. |
| I5 / Introduction, modeling axis, p2 | P | The model search is adaptive and includes unsuccessful routes. | “Richer supervision” and “pretrained representations” have only approximate meaning. Resolve in the model-sufficiency section, not with an introductory tutorial. |
| R1.1 / Search design opening, p2 | P | Evidence regimes and unavailable measurements are distinguished. | Repeats the preceding Introduction before explaining the actual prediction task. Compress this orientation. |
| R1.2 / Benchmark and folds, p2 | B | Pairs are kept together and a final evaluation is reserved. | What is one observation? Which tissues? What counts as edited? Does the output predict editing percentage? What are OOF and clustered uncertainty? The main task must be defined here, not on p9. |
| R2.1 / Baseline AUROC, p2 | B | The reference appears successful. | AUROC is not defined; a reader may take 0.93 as 93% of biological variation explained. Define ranking and separate it from editing level and variance. |
| R2.2 / Transfer, p2 | P | Transfer is stronger among human datasets and heterogeneous elsewhere. | “Transfer” needs a concrete train-in-one/evaluate-in-another definition; name the nonhuman organisms without requiring taxonomic recognition. |
| R2.3 / Baseline scope, p2 | C | High ranking accuracy does not resolve all biology or quantify explained variance. | Preserve this essential boundary; it is not ceremonial caution. |
| R3.1 / Genomic organization, p2 | B | Genomic separation contributes beyond graph summaries. | “Edited-site fraction” might mean EditingIndex at one site. R-squared might be compared directly with the preceding AUROC. Define the pair-level denominator and explain why genomic gap is missing from the folded input. |
| R4.1 / Matched-site tissue observations, p2 | B | The same site has different outcomes in different tissues. | EditingIndex is undefined, and “Alu-pair rankings” could mean model-score rankings. Identify the measured site-level percentage and observed pair-mean ranking. |
| R4.2 / Burden gate, p2 | B | A pair-level result does not automatically improve site prediction. | “Outer-isolated gate,” “expanded row-wise,” “neural burden auxiliary,” and “pair-first” arrive before their explanations. Keep the scientific conclusion beside the geometry result; place the numerical gate details in SI E. |
| F1 / Search overview and caption, p3 | P | The three biological axes, two analytical axes, and evidence classes are visible. | It appears after several empirical Results despite being the conceptual guide. Try placing it on p2. The small candidate names are a lookup map, not required first-pass reading. |
| R4.3 / RBP screens, p2--3 | B | Known regulators motivated a negative predictive screen. | eCLIP and bulk abundance have different sample resolution, which is not yet explained. Define the assays' role and the mismatch to tissue/site outcomes; retain the exploratory-fit limitation. |
| R4.4 / Transcript/chromatin screens, p3--4 | P | The tested contextual additions were not consistently useful. | H3K36me3 appears as an unexplained code. Identify it as a chromatin mark; do not add a chromatin review. |
| F2 / Benchmark/evaluation figure, p4 | P | Sites are nested in pairs; folds and final evaluation have different roles. | Useful diagram, but the caption must help interpret observation, pair, and held-out group. Do not redesign its geometry for a prose task. |
| R5.1 / Statistical weighting, p4 | P | Site-rich substrates receive greater influence, independently of biological importance. | “Canonical row set” is database language. Explain that the original balanced benchmark kept only some eligible observations. |
| R5.2 / Label recovery versus weighting, p4 | C | With the expanded pool held fixed, weighting adds a second gain. | The numbers become interpretable if metrics and label definitions were introduced earlier. Keep the class-conditional sampling distinction. |
| R5.3 / Five endpoints, p4 | C | The combined formulation improves every endpoint; this is a statistical improvement. | Preserve the bounded conclusion, not a universal biological unit claim. |
| R5.4 / Pair-size diagnostic, p4 | P | The gain is not uniform over substrate sizes. | “Log loss” appears for one secondary sentence. A metric guide and SI S8 are sufficient; no new main-text explanation of this side diagnostic. |
| R6.1 / Pooling and focal node, p4 | B | The reference already contains the target and pairing information. | “Pooling,” “focal node,” and “duplicate-global” are undefined. Explain whole-duplex summarization and retaining the target's own representation. |
| R6.2 / Local representation, p4 | B | An aligned local view improves the model. | Dot-bracket, helix partner, dilation, and embedding obscure the biological operation. Describe the neighborhood and predicted partner first; keep exact architecture in Methods/SI. |
| R6.3 / Shifted control, p4--5 | C | Moving the local window with capacity fixed tests correct centering, not the partner alone; liver replication and muscle-only control are distinguished. | Preserve the control and bounds. Define “natural prevalence” as undownsampled eligible observations, not an external cohort. |
| F3 / Scope/transfer figure, p5 | P | Rows are training datasets and columns are evaluation datasets; the human block is clear. | The two Latin genus names do not identify the organisms to a nonspecialist. Supply common names in the caption. Both metrics are performance scores, not editing percentages. |
| R6.4 / Structural strata, p5 | P | Gains are not confined to paired targets or one structural stratum. | “Stem extent” means length of a contiguous predicted stem; define compactly or use “stem lengths.” No causal claim about structural features. |
| R7.1 / Combining views, p5 | B | Averaging predictions helps relative to the whole-duplex component. | What is a logit? Why is a second whole-duplex model retained? Define log-odds and the independently trained components; do not imply that the local branch omits the global graph. |
| R7.2 / Fixed combination, p5 | P | Weights are fixed and independent fits are combined. | “Stacker” is unnecessary jargon. Explain it as another fitted combination model. |
| R7.3 / Within-duplex diagnostic, p5--6 | P | The combined model helps in two contexts; local-only evidence differs by tissue. | State that edited and nonedited sites are compared inside the same duplex, with equal duplex weight. Preserve the local-only replication limit. |
| F4 / Evidence map and geometry panels, p6 | P | Positive and negative branches are visible alongside pair-level quantitative evidence. | Its table is intentionally a detailed lookup. Define burden and distinguish the correlation panel from the incremental-prediction panel in the caption. Keep deferred versus unsupported separate. |
| R8.1 / Model-sufficiency classes, p6--7 | P | Extra learning sophistication could address errors without new measurements. | The four classes need concrete translations: intermediate levels, joint tissue training, pretrained RNA features, larger/connected models. No need to teach every architecture. |
| R8.2 / Model outcomes, p7 | B | Generic extensions were limited or adverse; targeted changes helped. | A catalogue of RiNALMo, all-A, hierarchical, Bio-aware, and GATv2 names interrupts the point. Explain representative functions; retain exact variant/status distinctions in SI. |
| F5 / Targeted changes and architecture, p7 | P | Label expansion, weighting, local centering, and fixed combination are separable comparisons. | The caption should identify logit scores and point units. Quantitative geometry and existing vector artwork need no redesign. |
| R9.1 / Final held-out result, p8 | P | The selected combination improves every reported metric on unseen pairs. | Are the earlier OOF scores and these scores from the same fits? Explain the separate evaluation regimes before this point. Spell out what the main metrics measure. |
| R9.2 / Scope of final test, p8 | C | Combined overlaps tissues and the final test does not independently confirm every component attribution. | Keep. |
| R9.3 / Error transitions, p8 | P | More baseline false positives are corrected than true positives are damaged. | The two percentages have different denominators. Say so in Fig. 6's caption; do not silently read the bars as fractions of all sites. |
| D1 / Discussion synthesis, p8 | C | Added genomic information, better use of existing local information, and weighting are different sources of gain. | Keep the three distinctions; avoid suggesting a full variance decomposition. |
| D2 / Relation to biology, p8 | C | Prior geometry/sequence/structure knowledge is credited, and incremental claims are bounded. | Preserve. |
| D3 / Context interpretation, p8 | P | Observed tissue variation and failed coarse proxies motivate matched measurements. | “A measurement hypothesis, not an observed latent state” answers an internal editorial concern before stating the biological point. Lead with unresolved tissue variation. |
| D4 / Modeling limit and residual error, p8 | C | Tested complexity did not close the gap, but several sources of residual error remain possible. | Preserve the finite-data/model/noise alternatives once at this scope. |
| D5 / Limitations, p8 | P | The benchmark, structures, screens, adaptive search, and external validity delimit the conclusions. | Dense but appropriate here. New reader aids should make it intelligible without adding more caveats. |
| D6 / Future measurement, p8 | C | The next test is joint measurement at matching biological resolution, not an assumed sufficient assay set. | Preserve the concrete testing criterion. |
| M1 / Search Methods, p8 | C | Candidate prioritization and outcome classifications are now explicit; adaptive work is not preregistered confirmation. | Keep. A reader need not memorize all class labels. |
| M2 / Benchmark Methods, p9 | B retrospectively | The actual high/low task finally becomes explicit. | This information arrived too late. Introduce it in the opening Results and retain the precise Methods definition. |
| M3 / Development protocol, p9 | P | Direct neural comparisons and early multistage screens differ in isolation. | Technical detail belongs here. Main text should state its consequence in plainer language. |
| F6 / Final test and boundary, p9 | P | Performance improves and the three tiers distinguish measured signal from unresolved context. | Explain error-transition denominators and that the unmeasured tier is not an assignment of all remaining error. |
| M4 / Model equation, p9--10 | P | Two logit scores are averaged before a sigmoid. | Define the component probabilities and logistic function. Preserve the equation. |
| M5 / Statistics, p10 | P | Whole-pair resampling and frozen choices prevent two common misreadings of precision. | Add a navigable metric/units guide in SI A, not a full tutorial here. |
| M6 / Preparation disclosure, p10 | C | Figure assets and AI assistance are disclosed, with pending author confirmation. | Human-action item, not a comprehension defect to edit away. |
| DA / Data availability, p10 | C | The reader knows where code, sources, and omitted-data instructions are located. | Preserve. |
| META / Authors, funding, declarations | B for submission | The document visibly remains an author-review draft. | Do not invent missing metadata. Continue to flag human confirmation. |
| REF / Bibliography, p10 | C | Primary biology and prior prediction work can be located. | No new references are needed for the planned comprehension fixes. |

## Pass 1: SI and cross-reference checks

| Passage | State | What the SI resolves / what still needs action |
|---|---|---|
| SI A: benchmark and coordinates | P | Correct labels and pair identity, but EditingIndex versus classifier probability versus pair burden still needs explicit separation. |
| SI A: OOF and uncertainty | C | Details explain the pooled-score limitation and grouped bootstrap; do not duplicate all of them in Results. |
| SI A: metric paragraph, Fig. S1, Table S1 | P | Many scores are listed without a reading guide; the graph and table become usable once their meanings and directions are supplied. |
| SI B: transfer | C | Heterogeneous dataset construction and RNAAtlas label concordance are distinguished from external predictor validation. |
| SI C: prior work and coverage | C | Prior methods are not caricatured; statuses are adaptive and task-specific. |
| SI D: RBP/context, Fig. S2 | P | Assay/sample distinctions are present but too late for the main story. Transfer a compact explanation to Results. |
| SI E: burden, Tables S2--S3 | P | Burden-gate outcome is documented. Define its denominator from `prepare_pair_table`; retain the main paragraph's exact numerical effects here. |
| SI F: supervision, Tables S4--S5 | C | Equal class draws, uniform pairs within class, and fixed label/epoch comparisons are explicit. |
| SI G: local controls, Table S6 | C | The local representation and muscle-only shifted control can be audited. Main text can use simpler biological wording without losing these details. |
| SI H: shared tissues | P | The formulation is explicit. Brier worsening is called “less calibrated,” although Brier alone does not isolate calibration; align this wording with SI L. |
| SI I: foundation and decoder | C | Frozen versus untested pretraining and completed versus stopped branches are distinguished. |
| SI J: natural prevalence, Fig. S3, Table S7 | P | The undownsampled high/low pool is defined; explicitly separate it from population-wide prevalence including intermediates. |
| SI K: unified model, Fig. S4, Table S8 | P | Prospective gate and stopped later stages are clear. Replace calibration-only wording based on Brier with probability-error wording. |
| SI L: final test, Figs. S5--S6, Table S9 | C | Frozen evaluation, descriptive matched recall, and probability-error/calibration distinction are explicit. |
| SI M / Table S10 | C | Provides the detailed branch-by-branch evidence behind the main synthesis. Preserve the navigable ledger and all dispositions. |
| SI N: tissue context, Fig. S7 | C | Same-site comparison and matching site sets within pairs make the observation interpretable without claiming causality. |
| SI N: within-duplex/strata, Fig. S8 | C | Within-pair ranking and its tissue limit are stated. Metric guide will clarify log loss/Brier and point units. |
| SI O: future measurements, credits, references | C | Hypotheses and licensed conceptual art are correctly distinguished from measured mechanism. |

## Revision decisions recorded before manuscript edits

1. Define the observation, high/low task, four tissues plus pooled Combined,
   OOF grouping, and meaning of the predicted probability before the first
   performance result. Avoid rewriting the already improved biological opening.
2. Separate site editing percentage from high-site fraction within a pair and
   from classifier probability. Check definitions against retained source code.
3. Place the search overview earlier if the actual PDF layout supports it.
   Explain figures through captions rather than reopen artwork design.
4. Keep the failed translation from pair burden to site prediction beside the
   geometry finding; move its gate/auxiliary numerical details into SI E.
5. Explain RBP binding maps, tissue averages, whole-duplex pooling, predicted
   pairing partners, and logit combination at first argumentative use.
6. Translate the model-sufficiency classes into functions; leave the exact
   variant inventory and dispositions in SI. Preserve all reported outcomes.
7. Add an SI metric/units guide; correct two calibration-only descriptions of
   Brier results. Make error-transition denominators explicit in the caption.
8. Reread after editing, with a separate implementation audit for quantitative
   preservation. Do not claim that a simulated second reading is human validation.

## Pass 2 and verification

### Sequential reread after revision

The second reading followed the rendered main paper from page 1 through the
bibliography, including every figure and caption, and then checked the SI
explanations and rendered pages. It tested the obstacles recorded above, not
whether the revised wording sounded different. A final short pass removed an
awkward duplicate interpretation after the focal-node comparison. The initial
log has been retained rather than rewritten to make the revision look inevitable.

| Checkpoint | After revision | What the reader can now infer; remaining dependency |
|---|---|---|
| T | P | The title names the question but deliberately leaves its biological stakes to the opening. The user-selected title is unchanged. |
| A1 | C | Why editing matters and why heterogeneous editing needs explanation are already clear. Preserved. |
| A2 | C | The final result concerns a combination fixed before testing on unseen duplexes. “Frozen two-view” and unexplained “endpoints” no longer interrupt the abstract. |
| S | C | The comparison concerns information needed to predict editing across duplexes, sites, and tissues, not distinguishing RNA identities. |
| I1--I3 | C | ADAR chemistry, self-RNA tolerance, Alu substrates, earlier predictors, and the three distinct biological questions can be followed without reading AdarEdit. These paragraphs were preserved. |
| I4 | P | Adding a measurement, reorganizing an input, and changing observation weights are different tests. The exact fold construction is reasonably pending until the benchmark paragraph; it is now explained before metrics appear. |
| I5 | P | Extra modeling is a separate question. The four class names are orientation, with their concrete meanings supplied in R8 rather than an introductory architecture tutorial. |
| F1, now p2 | P | The overview arrives before detailed Results. Its major axes and outcome classes are readable as an orientation map; individual candidate names still require the later Results/SI. It is not expected to replace those explanations. |
| R1.1--R1.2 | C | One observation is an adenosine in a pair with a tissue-level editing percentage. The high/low labels, excluded intermediate range, four tissues plus pooled Combined, and meaning of the classifier probability precede all performance claims. Keeping each pair together explains both OOF evaluation and whole-pair resampling. |
| R2.1 | C | AUROC is ranking, not percentage of editing or explained biological variance. The other principal metrics have short functional descriptions and an SI reading guide. |
| R2.2--R2.3 | C | Transfer means training in one dataset and evaluating in another. Its heterogeneity is interpretable without claiming evolutionary mechanism or equating AUROC with explained variance. |
| R3.1 | C | Pair burden is the fraction of eligible high/low sites classified as high-editing. Genomic distance is absent from the fixed-separator RNA input, so the incremental geometry comparison has a clear purpose. |
| R4.2, relocated after R3.1 | C | A useful pair-level predictor did not become a reliable site-level correction. That conclusion now immediately follows the geometry result; gate and auxiliary-model details are in SI E. |
| F2 | C | Individual sites travel with their complete pair through development and final evaluation. Earlier task definitions now support the existing diagram and caption. No artwork reset was needed. |
| R4.1 | C | The matched-site result concerns measured EditingIndex differences. The persistent pair rankings are rankings of observed pair means, not model probabilities. |
| R4.3 | C | Cell-line binding maps, sequence-based binding predictions, and tissue-average RBP expression are distinct proxies. The sample-resolution mismatch and Alu read-assignment problem are explicit. The exploratory fitting limitation is stated, with its technical explanation left to Methods. |
| R4.4 | C | H3K36me3 is identified as a chromatin mark. The negative conclusion remains about tested measurements and comparisons. |
| R5.1--R5.3 | C | The original balanced benchmark retained only part of the eligible observations. Recovering those labels and changing their weights are separate comparisons; uniform pair sampling remains conditional on class. |
| R5.4 | P | The small-substrate diagnostic has an understandable direction, but its log-loss definition and detailed quintiles appropriately require SI A/S8. This is secondary evidence, not a new premise of the main argument. |
| F3 | C | Both axes, transfer groups, group means, and human/pooled/nonhuman datasets are identified. The caption supplies acorn worm and sea urchin alongside the genus names. |
| R6.1 | C | Pooling means averaging nucleotide states into a whole-duplex summary. Retaining the target state is compared with duplicating the global state, making the control intelligible. |
| R6.2 | C | The extra input comprises two local sequence neighborhoods centered on the target and predicted partner. The main text explains what information enters; SI G gives the implementation. |
| R6.3 | C | Moving the window with capacity fixed tests correct centering, not the partner alone. Liver repeats the aligned-versus-whole comparison, not the shifted-window control. The secondary undownsampled high/low pool is defined. |
| R6.4 | C | The structural-strata result is described through pairing status and contiguous stem lengths, without implying that either feature alone explains the gain. |
| F4 | P | The quantitative panels now distinguish site-ranking AUROC from pair-level burden prediction and define the burden denominator. The evidence matrix remains a specialist lookup with Table S2, rather than something every reader must memorize. |
| R7.1--R7.2 | C | Both components include the graph; one adds local input. Their independently fitted log-odds scores are averaged with fixed weights, without another fitted combination model. |
| R7.3 | C | Within-duplex AUROC compares edited with nonedited sites in the same duplex, giving each mixed-label duplex equal weight. The local-only liver limit remains visible. |
| R8.1--R8.2 | C | Richer labels, joint tissue training, pretrained RNA features, and larger/connected models answer interpretable questions. RiNALMo and substrate-wide decoding are described by function. Exact variant names, confounding, and untested directions remain in SI. |
| F5 | C | Percentage-point effects and log-odds combination are explicit. The caption identifies development refits so their absolute scores are not mistaken for supplied-checkpoint test results. |
| R9.1--R9.2 | C | The final comparison evaluates the chosen combination on unseen benchmark pairs. It does not independently confirm every adaptive component attribution or constitute an external human cohort. |
| R9.3 / F6 | C | False-positive rescue and true-positive damage use their own baseline-group denominators. Positive Brier reduction means lower probability error. The synthesis's unmeasured tier names candidate measurements, not assigned causes of all error. |
| D1--D2 | C | Additional genomic information, better use of local sequence/structure, and changed statistical weighting are separate sources of improvement. Prior biological findings are credited. Preserved. |
| D3 | C | The paragraph starts with observed tissue differences left unresolved by the tested proxies. Matched measurements are a proposed next test, not an observed latent-state explanation. |
| D4 | C | Generic model extensions did not close the gap, while finite data, model limitations, measurement noise, structural uncertainty, and biological variation remain possible sources. Preserved at this scope. |
| D5 | P | This is the deliberately technical scope paragraph. The reader can follow the benchmark and measurement limits; fully nested estimation remains a Methods/SI dependency. No new catalogue of caveats was added. |
| D6 | C | Future measurements must improve prediction at the same sites across contexts beyond existing inputs. The proposed assay set is not claimed to be sufficient. Preserved. |
| M1--M2 | C | Evidence classes and task definitions are now methodological detail supporting an already intelligible story, not the reader's first chance to discover the prediction task. |
| M3 | P | The specialist can audit fold isolation. An adjacent-field reader can understand its consequence from Results, but reconstructing the multistage dependency still requires careful reading of SI A. |
| M4 | P | Component probabilities and the logistic function are defined. Reproducing the encoder still requires SI G and source code, an appropriate technical dependency. |
| M5 | C | Whole-pair uncertainty, frozen thresholds, and conditioning on fitted predictions are distinguished. SI A explains metric meanings and directions. |
| M6 / DA / REF | C | Disclosure, access paths, and citations remain available. No new references or unsupported reproducibility claims were added. |
| META | B for submission | Author, affiliation, correspondence, contributions, declarations, funding, and disclosure approval still require human completion. This is not concealed by the editing pass. |

The remaining **P** states identify legitimate reading depth or specialist
dependencies, not missing biological motivation. This audit found an explicit
remedy for the avoidable main-narrative obstacles recorded in Pass 1. It does
not establish that every real reader will understand the result or that no
further editorial improvement is possible.

### SI follow-through

| Initial SI checkpoint | Remedy or deliberate retention |
|---|---|
| A: outcomes, metrics, and coordinates | A compact, unnumbered reading guide distinguishes EditingIndex, pair burden, and classifier probability; defines metric direction and percentage points; and warns that the different references are not interchangeable. Existing Table S1--S10 numbering is preserved. |
| A: OOF / B: transfer / C: prior work and coverage | Technical evidence-regime distinctions and prior-work credit were preserved. The main text now supplies the concepts needed before these details. |
| D: RBP/context | A compact explanation of assays, tissue averages, and sample mismatch was transferred to the main Results. Full resources and controls remain in SI. |
| E: burden | The high/low denominator and separate all-available-site mean EditingIndex are explicit. The gate's three numerical effects moved here unchanged; its expanded-row reference remains distinct from the neural auxiliary model's pair-first reference. |
| F: supervision | Class-conditional pair weights, fixed update budget, and label-recovery control were preserved. |
| G: local representation | A source-code check corrected the feature description, as detailed below. Shifted-window scope and replication limits remain unchanged. |
| H / K: sharing and connected integration | Brier-based findings are described as probability error, not calibration alone. Their outcomes and effect sizes were preserved. |
| I: foundation / M: coverage | Completed, limited, superseded, and untested branches remain separate. The main text directs exact variant questions here. |
| J: natural prevalence | “Natural prevalence” explicitly means the undownsampled eligible high/low pool, not population-wide prevalence across all RNA sites. False-positive reduction is bounded to that secondary pool. |
| L: held-out / N: post-hoc | Final versus descriptive evidence remains distinct. The metric guide makes the error and within-duplex panels easier to interpret; no predictions were regenerated. |
| O: measurement and artwork | Candidate measurements and licensed conceptual illustration credits were preserved. No figure asset was changed in this pass. |

### What the reader should now be able to reconstruct

These are audit targets, not statements collected from human readers.

1. **After the first page:** Editing helps cells tolerate their own RNA, but
   different adenosines and tissues have different outcomes. Earlier predictors
   can recognize sites; their overall accuracy does not tell us which extra
   measurements or uses of existing information resolve those differences.
2. **After the benchmark and reference:** This is primarily a high/low site
   classification task, not prediction of each site's exact editing percentage.
   Complete pairs are held together. A good ranking score is not a complete
   biological explanation.
3. **After geometry and context:** Genomic organization helps predict differences
   in burden between substrates, without reliably identifying their individual
   edited sites. The same sites differ across tissues, but tested coarse proxies
   do not supply a stable correction.
4. **After weighting and local representation:** Some improvement comes from
   balancing correlated observations, and some from retaining target-local
   information already present in the RNA description. These are not new
   molecular measurements or a partner-specific causal discovery.
5. **After modeling and final evaluation:** Several generic extensions were
   limited. The selected combination works better on unseen pairs from this
   benchmark; that result has a different evidential role from adaptive OOF
   comparisons and does not establish external transportability.
6. **After the Discussion:** The comparisons guide what to measure next while
   leaving the causes of residual error unresolved. The paper's contribution is
   the separation of information sources and learning effects, not merely a
   small increase in a final score.

### Source-grounded correction found while explaining the local input

Checking what the reader should picture exposed a genuine description error.
`src/adarpro/models/multitask_oof.py::local_window` constructs two lanes with
12 channels in total: five nucleotide categories (A, G, C, T/U, and
unknown/unavailable) plus paired/unpaired state for each lane. The second lane
runs oppositely around the target's predicted partner. An unpaired target has
an unavailable partner lane. Target alignment comes from indexing; the local
input has no separate signed-position or focal/partner-indicator channels.
The graph's existing focal flag is a different input and remains present.

`encode` concatenates central and max-pooled convolutional states with the
whole-duplex embedding. Of the two convolutional layers, the second explicitly
uses dilation 2. The final training/prediction pipeline reuses these input
functions. Main Methods and SI G now describe that implementation instead of
the incorrect feature list. The implementation and trained models were not
modified. The correction is recorded in `paper/EPISTEMIC_CLAIM_AUDIT.md`.

### Preservation, builds, and visual checks

- The 398 definitions in `generated/quantitative_claims.tex` are byte-identical
  to the starting revision. Across main text plus SI, numerical-macro usage is
  unchanged except for one additional use of each existing high/low threshold
  in the earlier task definition. The three burden-gate effect macros moved
  from main text to SI; none was dropped.
- All 34 cited keys and the selected-predictor equation were preserved.
  The reference checker passed its static audit of 38 DOI records. No new
  literature-verification claim is made for this editorial pass.
- Both edited source CSVs retain their row counts and every numerical token.
  Changes are confined to interpretation fields replacing calibration-only
  descriptions with Brier-error descriptions. Generated tables and manifests
  were synchronized. The figure-source manifest changed only for the edited
  coverage-table hash; all figure SVGs remain unchanged.
- No model, experimental, data, prediction, checkpoint, or frozen-provenance
  file changed. No training, inference, new analysis, or new canonical-test
  evaluation was performed. Verification used already retained aggregate
  result tables, not test inputs or prediction-level artifacts.
- The main narrative (Introduction, Results, Discussion, excluding captions,
  headings, and Methods) grew from approximately 2,776 to 2,968 lexical tokens
  under the same counting convention, with LaTeX macros counted as placeholders.
  The increase buys first-use definitions and biological interpretation, not
  additional experimental claims. The abstract is 206 words and Significance
  115 under the repository checker.
- Both PDFs were repeatedly compiled. The main manuscript is 11 pages (10 at
  the starting revision), with the overview moved to page 2 and the bibliography
  kept together; SI remains 16 pages. All main pages were visually inspected,
  including full-width figures and captions, and all SI pages were checked at
  page scale with the new metric guide and changed passages inspected more
  closely. The guide's initial table overflow and an orphaned bibliography
  continuation were corrected. No new clipping or unresolved reference was
  found. Template-origin first-page overfull-box warnings and microtype/caption
  compatibility warnings remain; the visible output and automated main-page
  bounds check were inspected rather than treating a successful build as proof.
- `verify_paper_artifacts.py`, `verify_paper_references.py`, `ruff check
  scripts/render_paper_tables.py`, `git diff --check`, and `qpdf --check` for
  both PDFs passed. Figures remain SVG placements with no embedded raster
  assets; this pass did not alter their artwork or font system.

Final manuscript artifacts (SHA-256):

| File | SHA-256 |
|---|---|
| `paper/main.tex` | `470b54c54c9beb60034cc0194afa2bbcad6cc57aaffbf9deba92d1e8dcb88899` |
| `paper/main.pdf` | `d892dff1474b7a15a001f349cc7c6896033b5784b3ccf0298e94f5ca88352965` |
| `paper/supplementary.tex` | `bf67216acd10ca7f5b39d5c9e20bb1e6b4353f2b0a090808ab8befaf9e4a4585` |
| `paper/supplementary.pdf` | `a655957a1fef3b12676d61b795e8a9189cefaf3f689f887220059e72bada2a9a` |

### Questions editing should not pretend to answer

The causes of unexplained editing variation, partner-specific information,
fully nested replication of early residual screens, and transportability to
an independently generated human cohort remain scientific questions. An actual
adjacent-field reader's unaided retelling would be a valuable next editorial
check; the simulated retellings above are not a substitute. Submission also
still requires author metadata, declarations, funding/acknowledgment details,
and author approval of disclosure and graphics-policy compliance. None was
invented or silently marked complete.
