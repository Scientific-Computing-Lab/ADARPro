# ADARpro final scientific report

Status: complete. The development campaign, ensemble-versus-unified selection,
hash-complete freeze and single canonical-test evaluation are complete.

## Executive summary

ADARpro asked what information or modeling assumption is missing from
AdarEdit, while preserving the published 884-substrate, globally pair-disjoint
human benchmark. The strongest development result was not generic RBP
occupancy, a larger GNN, or a broad collection of biologically named
annotations. It was a combination of two controlled changes:

1. train on every eligible high-confidence edited/non-edited observation from
   training pairs, while sampling pairs first so site-rich duplexes do not
   dominate; and
2. preserve target-centered sequence and helix-partner geometry with a compact
   local encoder rather than relying only on a whole-duplex pooled graph.

The selected successor is a fixed 50:50 logit ensemble of the pair-balanced
compact graph model and the target/helix-partner-aligned local-fusion model. No
ensemble weight was searched. On pair-disjoint development folds, the ensemble
improved over the expanded pair-balanced component by 2.56 AUROC, 3.08 AUPRC
and 1.39 F1 points in Muscle Skeletal, and by 1.82, 2.14 and 1.79 points in
Liver. All corresponding pair-bootstrap intervals were positive. The local
model also decisively beat an identical CNN centered 96 nt away. At natural
prevalence, the fixed ensemble improved F1 by 7.52 points in Muscle and 2.65
points in Liver relative to pair-balanced training.

The separately preregistered unified end-to-end model did not improve on this
ensemble: Muscle AUROC/AUPRC/F1 were 0.9072/0.8922/0.8317 versus
0.9142/0.9040/0.8347, with adverse AUROC and AUPRC pair-bootstrap intervals and
worse Brier calibration. Its conditional repair and replication gates were not
opened. Thus representation-level joint learning did not outperform the fixed
prediction-level integration, and the ensemble remains the final development
winner.

These development results were not substituted for the published benchmark.
After the ensemble was frozen, its one authorized canonical-test evaluation
improved macro AUROC/AUPRC/F1 by 0.0277/0.0261/0.0253 over reproduced published
AdarEdit. Every context improved all three metrics with positive pair-clustered
95% intervals. F1 gains ranged from 1.78 points in Combined to 4.01 points in
Muscle Skeletal. Brier score improved in every context, and false positives
fell at matched baseline recall. This satisfies the project's standard for a
material, reproducible successor while leaving the original RBP hypothesis
unsupported.

## 1. Reproduced foundation

The project is based on upstream AdarEdit revision
`5c8b57d70cdc2eb1c5e168431013275bcc9ed0c5`. The original checkout is preserved
separately. Reproduction verified 884 graph-buildable inverted-Alu substrates,
566/139/179 train/validation/test pairs, and zero pair overlap. The local
environment, canonical input hashes and supplied checkpoints are recorded in
`provenance/`.

The supplied Baseline GAT reproduced the following authoritative held-out
results:

| Context | AUROC | AUPRC | F1 | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Artery Tibial | 0.9238 | 0.9262 | 0.8671 | 0.8345 | 0.9023 | 0.7930 |
| Brain/Cerebellum | 0.9301 | 0.9286 | 0.8689 | 0.8371 | 0.9032 | 0.8021 |
| Liver | 0.9092 | 0.9242 | 0.8541 | 0.8368 | 0.8721 | 0.7788 |
| Muscle Skeletal | 0.8688 | 0.8757 | 0.8139 | 0.7767 | 0.8549 | 0.7152 |
| Combined | 0.9331 | 0.9356 | 0.8674 | 0.8334 | 0.9043 | 0.8026 |

`Combined` is a pooled endpoint, not a fifth tissue, and was never assigned a
fabricated tissue-expression vector. The existing four-tissue no-Combined
joint Baseline is also retained as a secondary published comparator, notably
F1 0.8390/AUROC 0.8982 in Muscle and 0.8647/0.9128 in Liver.

## 2. Leakage-control and statistical design

The canonical test JSONL and metadata files were inventoried by SHA-256 and
placed behind a stateful quarantine. During discovery, every learned model,
normalizer, feature selector and threshold excluded the outer pair fold. OOF
predictions are an internal instrument for estimating incremental value under
identical folds; their lower absolute scores do not redefine the AdarEdit
benchmark.

Uncertainty is computed by resampling whole Alu pairs. Candidate selection used
direction across folds, pair-bootstrap intervals, calibration, false-positive
rescue and true-positive damage. The natural-prevalence analyses preserve the
published pair assignment and include every eligible high/low observation on
development pairs; they are secondary robustness analyses, not a replacement
task.

Before the final test, one global model family is used for all five contexts.
Its components are trained on published training pairs, epochs and thresholds
are selected only on published validation, and every checkpoint is hashed into
a final manifest. The manifest is bound to the code revision and receives one
single-use authorization.

## 3. Research tree and falsification results

### Generic RBP context

ENCODE eCLIP, PrismNet and RBPNet binding features, bulk GTEx abundance and
binding-by-abundance interactions did not add credible within-tissue residual
signal. Experimental peak absence was treated as uncertain because inverted
Alu reads are vulnerable to multimapping. Predictive binding evidence avoided
that specific absence problem but still failed the matched and randomized
controls. In strict cross-tissue switches, tissue identity alone nearly
saturated switch direction because the four GTEx summaries have strong tissue
ordering; adding current RBP features did not establish a transferable RBP
mechanism.

Conclusion: this operationalization of the original RBP hypothesis is not
supported. It was terminated rather than made larger or deeper. The result
does not show that RBPs are biologically irrelevant; it shows that cell-line
binding resources combined with bulk tissue abundance do not resolve this
benchmark.

### Genomic architecture, transcript state and chromatin

A broad site-level IRAlu/transcript panel overfit and failed to add stable
residual value. In contrast, genomic pair geometry predicted a pair's edited-
site fraction in every individual tissue. The actual gap between Alu arms—the
information removed when the model joins the arms with a fixed separator—had
Spearman correlations from -0.33 to -0.48 with edited fraction. Geometry plus
graph summaries reached pair-level R² 0.050 in Muscle and 0.239–0.277 in the
other tissues, and beat graph-inferable summaries under pair bootstrap.

That finding supports a substrate-realization layer biologically, but two
direct integrations were too weak for promotion. An outer-isolated engagement
gate improved AUPRC by 0.85 points but not F1 robustly and worsened Brier. A
neural context/auxiliary-engagement model was stopped after three folds because
F1 decreased by 0.65 points. Exact-tissue H3K36me3 features were also small and
inconsistent: a Liver AUPRC gain did not extend robustly to AUROC/F1 or the
pair-engagement endpoint. H2A.Z lacked an adequately matched resource for a
fair powered test.

Conclusion: substrate realization is measurable, but the currently available
static geometry and bulk chromatin proxies are not sufficient to improve the
leading site predictor materially.

### Richer supervision

Pair-balanced training on all high/low observations from training pairs was
the strongest general correction. Relative to the same-fold internal AdarEdit
reference, AUROC/AUPRC/F1 changes were:

| Context | ΔAUROC | ΔAUPRC | ΔF1 |
|---|---:|---:|---:|
| Artery | +0.0122 | +0.0137 | +0.0069 |
| Brain | +0.0117 | +0.0164 | +0.0111 |
| Liver | +0.0179 | +0.0172 | +0.0164 |
| Muscle Skeletal | +0.0429 | +0.0473 | +0.0426 |
| Combined | +0.0147 | +0.0170 | +0.0143 |

All five F1 and ranking intervals excluded zero. The gain therefore reflects
both more valid supervision and a correction to effective sample weighting:
each pair is sampled uniformly before sampling one of its sites.

Continuous/ordinal/binary multi-task learning added a small ranking and
calibration signal, but its standalone F1 gain over pair-balanced binary
training was negligible. Adding it to an ensemble reduced natural-prevalence
F1. It was excluded from the primary successor.

### Substrate-wide and shared models

A direct all-adenosine decoder lost 6.64 AUROC, 6.73 AUPRC and 5.99 F1 points
over two complete folds and was stopped for futility. A cross-fitted relative
rank feature helped the weaker original encoder but added no value beyond the
pair-balanced model or a within-pair shuffled-rank control. Thus the current
data support pair-aware sampling, not the tested high-capacity joint decoder.

A four-tissue masked multi-label model produced genuine transfer signal over
independently trained pair-balanced models: macro changes were +0.0087 AUROC,
+0.0116 AUPRC and +0.0075 F1. It helped data-limited Muscle most, but missed
both precommitted +0.01 macro promotion gates and worsened mean Brier. It is a
secondary result, not part of the frozen successor.

### Foundation and structure representations

Frozen RiNALMo embeddings did not improve AUROC/AUPRC beyond the selected graph
representation and matched controls. Scaling from the 35M to 150M model did
not alter the conclusion. A task-specific pretraining corpus is feasible—over
288,000 benchmark-excluded, opposite-orientation Alu pairs lie within a GENCODE
transcript span—but isoform connectivity, foldability and redundancy controls
must be completed before this becomes a valid training resource. It is a
future experiment, not part of the present benchmark claim.

RNAfold ensemble summaries were likewise small and unstable. Truly matched
in-vivo structural state and donor-level editing outcomes are not available at
adequate scale in the current data. These remain high-value data-acquisition
directions rather than untested claims.

## 4. Selected successor

The pair component keeps AdarEdit's compact whole-duplex GAT but trains from
the expanded high/low pool with equal class sampling and uniform pair-first
sampling. The local component adds a deterministic 65-position representation
centered on the target and its dot-bracket-aligned partner. A dilated CNN
preserves target-centered base, pairing and relative-position patterns before
fusion with the graph representation. The two independently trained
probabilities are combined by an equal-weight logit average.

This model is deliberately smaller in scientific scope than the original RBP
hypothesis. It makes no claim to measure the cell environment. It tests the
more defensible result supported by the data: current AdarEdit underuses valid
training labels and under-resolves the local editing center when hundreds of
nucleotide embeddings are collapsed to a global mean.

The capacity control is strong. The same local network centered 96 nt away
scored 0.7122 AUROC/0.6943 AUPRC/0.7028 F1 in Muscle, versus
0.9045/0.8936/0.8337 when aligned to the target. The aligned-minus-shifted
pair-bootstrap differences were +0.1926 AUROC, +0.1993 AUPRC and +0.1311 F1,
with every interval favorable. The gain is therefore not generic CNN capacity.

| Development comparison | ΔAUROC | ΔAUPRC | ΔF1 |
|---|---:|---:|---:|
| Muscle: local − pair | +0.0161 | +0.0204 | +0.0131 |
| Muscle: fixed ensemble − pair | +0.0256 | +0.0308 | +0.0139 |
| Liver: local − pair | +0.0085 | +0.0087 | +0.0088 |
| Liver: fixed ensemble − pair | +0.0182 | +0.0214 | +0.0179 |

All listed intervals are pair-clustered and exclude zero. No development delta
is arithmetically added to a published test score.

### Frozen validation artifacts

All ten selected component fits completed the prespecified 1,000 epochs. The
fixed equal-logit ensemble was then assembled independently for each context,
using only published validation labels for its threshold:

| Context | Threshold | Validation AUROC | Validation AUPRC | Validation F1 |
|---|---:|---:|---:|---:|
| Artery | 0.475 | 0.9384 | 0.9208 | 0.8711 |
| Brain | 0.425 | 0.9455 | 0.9339 | 0.8836 |
| Liver | 0.550 | 0.9407 | 0.9268 | 0.8591 |
| Muscle Skeletal | 0.225 | 0.9312 | 0.9340 | 0.8693 |
| Combined | 0.350 | 0.9477 | 0.9412 | 0.8842 |

These values are selection-set diagnostics, not final performance estimates.
The integrity audit verified complete epoch histories, selected epochs and
thresholds, checkpoint hashes/configuration, exact equal-logit arithmetic,
saved metrics, and the sealed-test flag. The repository test suite passes 120
tests and Ruff reports no violations.

## 5. False-positive behavior and natural prevalence

On canonical Muscle development rows, the fixed ensemble corrected 302 of 764
pair-model false positives and created 54 new false positives. Relative to the
internal original-model OOF reference, it corrected 541 of 942 false positives
and created 115, while damaging 206 true positives and recovering 208 false
negatives.

At natural prevalence, the ensemble reached AUROC/AUPRC/F1
0.9165/0.4540/0.3621 in Muscle and 0.9290/0.7687/0.6684 in Liver. Versus the
pair-balanced component, the changes were +0.0265/+0.0880/+0.0752 and
+0.0196/+0.0558/+0.0265, respectively. In Muscle this removed 4,196 false
positives at a cost of 103 true positives. These estimates come from grouped
development OOF evaluation with fold-specific validation thresholds.

## 6. Biological interpretation

AdarEdit's strongest remaining, measurable limitation in these data is not a
single missing RBP. The evidence supports two scales:

- the genomic separation and arrangement of the Alu arms carry substrate-level
  information about whether an editable duplex is realized;
- conditional on the represented substrate, target/partner-aligned local
  patterns carry site-level information that global graph pooling dilutes.

Only the second currently produces a material, reproducible predictive gain
when integrated with the stronger supervision regime. The first is credible
mechanistic evidence but not a successful model component. Consequently the
full factorization
`P(substrate engaged) × P(site edited | engagement)` remains a useful
scientific model, but the available engagement proxies are too noisy for the
predictive successor.

## 7. Locked canonical test

The final manifest froze the immutable equal-logit ensemble at revision
`0f50fac33bec8bad325e6745d83ceaacebe42fc1` and has SHA-256
`dd04f9786d52f57653da31b12dc777d10aa4247dfccd2b2c59ddff64ad395a1c`.
The authorization was consumed once; the quarantine now records
`final_consumed` and `evaluations_consumed = 1`.

| Context | Published F1 | ADARpro F1 | ΔF1 | Published AUROC | ADARpro AUROC | ΔAUROC |
|---|---:|---:|---:|---:|---:|---:|
| Artery | 0.8671 | 0.8870 | +0.0199 | 0.9238 | 0.9477 | +0.0239 |
| Brain | 0.8689 | 0.8902 | +0.0213 | 0.9301 | 0.9508 | +0.0206 |
| Liver | 0.8541 | 0.8812 | +0.0271 | 0.9092 | 0.9414 | +0.0322 |
| Muscle Skeletal | 0.8139 | 0.8541 | +0.0401 | 0.8688 | 0.9144 | +0.0456 |
| Combined | 0.8674 | 0.8852 | +0.0178 | 0.9331 | 0.9491 | +0.0160 |

AUPRC gains are +0.0224, +0.0217, +0.0258, +0.0450 and +0.0157 in the same
order. All AUROC, AUPRC and F1 pair-bootstrap intervals exclude zero. At the
frozen thresholds, false positives decreased by 160, 109, 159, 42 and 31,
respectively. At exactly matched published-baseline recall, the candidate
avoids 126, 119, 158, 68 and 148 false positives; every matched-recall
pair-bootstrap interval excludes zero. Detailed metrics, error transitions,
calibration and figures are in report 07 and `figures/final/`.

## 8. Answers to the project questions

1. **What was reproduced?** The exact upstream split, supplied per-context
   checkpoints and aggregate held-out metrics.
2. **Where does AdarEdit fail?** Development residuals include underused
   training signal, local target information diluted by global pooling, and a
   weaker substrate-realization component associated with Alu geometry.
3. **Are false positives enriched for current RBP interference features?** No
   robust enrichment that survives the required controls.
4. **Which RBPs matter?** None can be declared predictive from the tested
   eCLIP/PrismNet/RBPNet × bulk-expression measurements.
5. **Does binding alone add information?** Not credibly in the present data.
6. **Does tissue abundance add information?** Not beyond controls in the
   tested bulk formulation.
7. **Does binding × abundance explain switches?** No; tissue ordering dominates
   the limited matched-switch task.
8. **Does ADARpro improve prediction beyond AdarEdit?** Yes. On the untouched
   canonical test, macro AUROC/AUPRC/F1 improve by 0.0277/0.0261/0.0253, and
   every context has positive clustered intervals for all three metrics.
9. **How much does it reduce false positives?** At frozen thresholds it rescues
   32.0–50.6% of baseline false positives, with 1.7–5.9% TP damage. At matched
   baseline recall it avoids 68–158 false positives per endpoint.
10. **Is the gain pair-robust?** Yes. Development selection was pair-grouped,
    and final AUROC/AUPRC/F1 intervals are positive in all five contexts under
    whole-pair bootstrap resampling.
11. **Does it survive controls?** Yes for pair sampling and target alignment;
    no for the terminated RBP, broad context, chromatin, joint-decoder and
    generic foundation-model branches.
12. **Does it survive the canonical test?** Yes, in all five endpoints across
    ranking, F1 and Brier score, under one consumed authorization.
13. **What mechanism is supported?** Target-centered information must remain
    explicitly represented; Alu pair geometry also marks substrate realization
    but has not yet improved the leading predictor.
14. **What remains uncertain?** Donor/cell-state regulation, in-vivo structure,
    isoform-specific ADAR activity and direct perturbational RBP effects.
15. **What should come next?** Validate on a genuinely independent cohort and
    acquire matched donor-level editing/molecular-state data. A second priority
    is leakage-controlled task-specific IRAlu pretraining after transcript,
    isoform and sequence-redundancy filtering.

## 9. Reproducibility map

- Baseline reproduction: `reports/00_adaredit_reproduction.md`
- Test quarantine and protocol: `ADARPRO_PROTOCOL.md`,
  `provenance/test_quarantine.json`
- Ranked research tree: `experiments/performance_frontier_opportunities.csv`
- RBP evidence and negative result: reports 02–05
- Biology/AI integration: `reports/06_integrated_models.md`
- Genomic architecture and chromatin: reports 10–11
- Donor, in-vivo structure and foundation-model audits: reports 13–15
- Richer supervision and target-local experiments:
  `reports/16_richer_supervision_and_substrate_context.md`
- Task-specific pretraining feasibility: report 17
- Deferred unified end-to-end objective and test-boundary rule: report 18
- Unified preregistration and development result: reports 19–20
- Locked canonical-test result: report 07,
  `results/adarpro/final_locked_test/`, `figures/final/`
- One-row-per-run registry: `experiments/experiments.csv`

Every numerical figure is regenerated from saved tables. Final-test plots were
generated only after the quarantine recorded the consumed evaluation.
