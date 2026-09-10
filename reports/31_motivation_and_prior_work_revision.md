# Why this study was needed after earlier predictors

Date: 2026-09-10 UTC

Starting revision: `f0ea18df96ea64a287cc4cff16dd40ea01ace5cc`

## Reader problem and scientific answer

The previous opening assumed that a reader already understood the editing
problem and the earlier AdarEdit paper. It named information scales and a
search, but did not sufficiently explain why a successful predictor left a
scientific question worth pursuing.

The revised entry point is concrete: editing helps prevent self-RNA from
triggering antiviral defenses, yet neighboring adenosines differ in editing
and the same genomic site can differ across tissues. Those observations pose
related but distinct questions about an entire RNA duplex, the selection of a
site within it, and its cellular setting. Alu repeats, duplex formation, and
the graph reference are explained before they become shorthand.

**Thesis:** Controlled comparisons beyond an already predictive RNA reference
distinguish information added by genomic organization from signal recovered
by target-centered representation and substrate-aware learning, while showing
which contextual questions the tested measurements leave unresolved.

An overall prediction score cannot make these distinctions. They matter for
deciding which gains can be obtained from existing RNA descriptions and which
biological measurements should be tested next. This is not a claim to have
assigned every residual error to a cause or to have discovered RNA-editing
determinants for the first time.

## Prior-work check

The revision checked the actual scope of prior work before defining the gap.

| Prior work | What it already contributes | Consequence for this manuscript |
|---|---|---|
| Eggington et al. (2011), doi:10.1038/ncomms1324 | Sequence preferences, predictive modeling, and structural interpretation of departures from prediction | Prediction as a biological probe is not itself new. |
| DeepRed, Ouyang et al. (2018), doi:10.1038/s41598-018-24298-y | Sequence-based editing-site identification and biological feature analysis | Earlier predictors are not described as accuracy-only work. |
| EditPredict, Wang et al. (2021), doi:10.1016/j.ygeno.2021.09.016 | Local-window prediction, motif analysis, and scope/validation tests | Local sequence windows and interpretation are not claimed as firsts; its label construction differs from this benchmark. |
| Uzonyi et al. (2021), doi:10.1016/j.molcel.2021.03.024 | Systematic probing of sequence and structural determinants | The novelty is not the first systematic study of editing determinants. |
| ADAR-GPT, Rosenwasser et al. (2026), doi:10.1073/pnas.2529073123 | Fine-tuned sequence prediction, motif interventions, representation comparisons, and controls | Modern AI and negative representation tests predate this study; their scores are not directly comparable across different benchmark definitions. |
| AdarEdit v2, Rosenwasser et al. (2026), doi:10.64898/2026.02.02.703375 | Whole-duplex graph prediction, target features, structural sensitivity, grouped evaluation, and cross-context scope | Explain the reference for new readers and credit the inherited benchmark. The added contribution is the subsequent controlled incremental search. |

Primary-source access:

- Eggington full text: <https://pmc.ncbi.nlm.nih.gov/articles/PMC3113232/>.
- DeepRed full text: <https://pmc.ncbi.nlm.nih.gov/articles/PMC5902551/>.
- EditPredict full text: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8671215/>.
- ADAR-GPT full text: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12798952/>.
- AdarEdit v2 full text:
  <https://www.biorxiv.org/content/10.64898/2026.02.02.703375v2.full>.
- Uzonyi bibliographic metadata and abstract were verified through Crossref
  and Europe PMC. Publisher full-text access was blocked; the citation is
  limited to the verified scope of systematic sequence/structure probing.
  No detailed result was imported from an unread full text.

Three bibliography records were added: DeepRed, EditPredict, and Uzonyi.
ADAR-GPT already had a record but is now cited in the main narrative.

## Implemented changes

- **Significance and Abstract:** begin with the biological role and observed
  selectivity, then explain what overall prediction accuracy cannot resolve,
  the comparative approach, its findings, and the next measurement question.
  The abstract has 202 words and the significance statement 113 under the
  repository's word-count check. Neither inventories performance metrics.
- **Introduction:** define the biological objects and previous achievements
  without assuming familiarity with AdarEdit. Credit prior motif, structural,
  and perturbation work. Explain why duplex burden, local site choice, and
  tissue differences need distinct comparisons, and why adding a measurement,
  reorganizing an existing input, and changing observation weights answer
  different questions.
- **Results:** add the purpose of the decisive comparisons at their point of
  use. Identify the baseline as inherited; acknowledge known Alu geometry
  before testing its increment; motivate RBP screens from known regulation;
  state that the graph already includes target and pairing information before
  testing whether pooling uses it effectively. Keep the result order intact.
- **Overview caption and Discussion:** connect the comparisons to the
  unresolved question. Separate new input, better representation, and
  statistical reweighting. The final predictor validates the selected learning
  and representation changes, not every interpretation of the search.
- **SI section C:** document the relationship to prior work and differences
  between its tasks, so the main paper need not carry a comparator catalogue.
- **Consistency correction in SI section G:** the early target-contact screen
  is now correctly labeled pooled-score exploratory, not fold-isolated. The
  separate focal-node neural comparison remains fold-isolated. This agrees
  with the existing implementation audit and changes no result.

## Preserved boundaries

No model, analysis, selection decision, numerical result, equation, figure
asset, source table, or claim macro was changed. No canonical-test predictions
were accessed or recomputed. Existing citation keys were retained. Development
attribution, early-screen nestedness limits, the muscle-only shifted-window
control, noncausal tissue associations, limited local-only within-duplex
replication, and lack of external-cohort predictor validation remain explicit.

The paper does not establish a universal modeling ceiling or attribute all
unexplained variation to cellular state. Additional matched measurements are
proposed tests, not a known sufficient set. The title and non-branded framing
are unchanged.

## Verification

- Static and online Crossref reference audit passed for all 38 audited DOI
  records; 34 keys are cited across the main text and SI.
- Generated quantitative-macro occurrence counts and numerical literals were
  compared with the starting revision in both manuscript sources; unchanged.
- PNAS length limits, claim traceability, source hashes, vector-only SVG
  placement, compilation, and main-paper page bounds passed the repository
  artifact verifier.
- Both PDFs compiled and passed `qpdf --check`: main 10 pages; SI 16 pages.
- All pages were rendered for layout review; first-page motivation, the
  continuing Introduction, Discussion, and revised SI passages were inspected
  at reading scale. Existing PNAS-template box/microtype warnings remain;
  no unresolved citation or cross-reference was reported.
- Narrative text excluding captions and Methods increased from 2,542 to 2,745
  words. The added space supplies missing biological explanation and prior-work
  context; it does not add experimental branches or results.

Author, affiliation, contribution, competing-interest, correspondence, funding,
and disclosure confirmations remain human-action items already identified by
the submission audit. This revision does not represent them as resolved.
