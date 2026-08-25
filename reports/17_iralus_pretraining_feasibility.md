# 17 — Task-specific IRAlu pretraining feasibility

Date: 2026-08-22. This is an outcome-independent resource audit, not a
predictive result. No canonical test label or prediction was accessed.

## Rationale

Frozen generic RiNALMo representations did not add robust ranking information
to the leading graph ensemble. That negative result does not answer the
different question of whether self-supervision on the exact inverted-Alu
substrate class could learn useful duplex priors. Before paying the cost of
folding or pretraining, the available external sequence pool was quantified.

The GRCh38 UCSC RepeatMasker table contains 1,282,821 Alus. Every RepeatMasker
record overlapping either arm of any of the 905 canonical AdarEdit pairs was
conservatively excluded (1,805 records), without reading benchmark labels.
Among the remaining 1,281,016 Alus, exact counts of nonoverlapping,
opposite-orientation pairs were:

| Maximum genomic gap | Candidate pairs |
|---:|---:|
| 1 kb | 501,152 |
| 2 kb | 991,945 |
| 5 kb | 2,406,873 |
| 10 kb | 4,700,792 |

These are upper bounds, not ready training examples. Many pairs will not occur
within a compatible expressed transcript, will be redundant, or will fail to
form the intended duplex. Chromosome-level and sequence-similarity exclusion
will also be needed to prevent homologous leakage into development pairs.

As a second gate, requiring the entire ≤1 kb pair span to lie inside at least
one GENCODE v46 transcript retained 288,056 pairs (57.5% of the sequence-only
pool). This deliberately permissive proxy establishes scale; it does not prove
that both Alus coexist in a mature isoform, are expressed in the benchmark
tissues, or form a duplex.

## Decision

Raw sequence availability and coarse transcript containment are not the
bottlenecks, so the biological hypothesis is not falsified. The available pool
is nevertheless not a valid pretraining corpus: isoform co-occurrence,
foldability and sequence-homology leakage have not been resolved. Building and
auditing that corpus is a separate data-construction project, not a justified
late architecture sweep on the current benchmark.

The completed target-local ladder has meanwhile produced a controlled,
independently replicated successor using only validated inputs. Task-specific
pretraining is therefore deferred beyond the frozen ADARpro benchmark rather
than allowed to delay or contaminate the single final evaluation. It should be
reopened only after transcript-compatible pairing and homology-group exclusion
are implemented, followed by a compute-matched development-only pilot. Generic
foundation-model fusion stays terminated.

Machine-readable result:
`data/derived/pretraining/iralus_feasibility.json`.
