# 03 — RBP feature audit

## Experimental occupancy coverage

The outcome-blind audit uses 25 replicate-combined GRCh38 eCLIP tracks across
14 proteins and 46,133 unique development sites. Features were fixed at the
target nucleotide, its predicted base-paired partner, both Alu arms, the whole
duplex, and ±5/10/25/50/100 nt before any residual outcome analysis.

| RBP | Tracks | Focal overlap | Within 25 nt | Any duplex peak |
|---|---:|---:|---:|---:|
| ILF3 | 2 | 211 (0.457%) | 370 (0.802%) | 1,979 (4.290%) |
| ADAR legacy control | 1 | 105 (0.228%) | 181 (0.392%) | 759 (1.645%) |
| DROSHA | 2 | 39 (0.085%) | 50 (0.108%) | 142 (0.308%) |
| HNRNPC | 2 | 27 (0.059%) | 50 (0.108%) | 330 (0.715%) |
| TIA1 negative control | 2 | 23 (0.050%) | 40 (0.087%) | 341 (0.739%) |
| IGF2BP1 negative control | 2 | 15 (0.033%) | 32 (0.069%) | 269 (0.583%) |
| MATR3 | 2 | 3 (0.007%) | 5 (0.011%) | 85 (0.184%) |

FXR1, HNRNPM, PUS1, RBFOX2, TROVE2, and XRCC6 have no confident peak anywhere
on these duplexes despite available assays; 14 of 25 individual tracks have no
duplex peak. ILF3 is the clearest experimental candidate, but even ILF3 covers
only 4.3% of substrates at duplex scale.

## Why zero peaks are not negative binding calls

The official ENCODE eCLIP SOP v2.2 first maps reads to a human RepBase index
and removes them, then maps the remaining reads to the genome with STAR
`--outFilterMultimapNmax 1`. This design directly censors repeat-derived and
multi-mapping evidence. The SOP and checksum are preserved under `provenance/`.

An independent GRCh38 100-mer Umap audit shows that 96.1% of focal bases have
a score of 1.0 and 96.3% of all 100-mer starts overlapping a focal site are
unique on average. Thus many individual loci are genomically distinguishable
at 100 nt, but unique genomic mappability cannot recover reads already removed
for matching RepBase. Positive eCLIP peaks remain useful; absence is uncertain.

## Tissue abundance

The current GTEx v8 API supplied 54-tissue median TPM for all 19 panel proteins,
ADAR, and ADARB1 using exact GENCODE v26 identifiers. API values independently
match AdarEdit's frozen ADAR/ADARB1 table after its two-decimal rounding. The
four mappings are Artery Tibial, Brain Cerebellum, Liver, and Muscle Skeletal.
Combined receives no fabricated expression vector.

ILF3 median TPM is 69.6, 104.9, 20.5, and 44.8 across those contexts;
corresponding ILF2 values are 94.6, 78.4, 31.2, and 69.9. The model-ready layer
retains TPM, log1p TPM, cross-GTEx percentile, RBP/ADAR ratios, generic binding,
and prespecified binding × log1p(TPM) interactions.

## Shifted-location falsification control

Each experimental feature now has a matched negative-control measurement at
the same chromosome and strand exactly 1,000,000 nt downstream. The offset was
fixed before any outcome analysis and is applied to the focal nucleotide,
paired partner, and both duplex arms. A genuine substrate-local occupancy
effect must outperform this same-dimensional genomic peak-density control.

## Consequence for the modeling funnel

Experimental peaks will be tested as sparse positive evidence, not as reliable
binary absence. Sequence-based RBPNet profiles are therefore the required
complementary binding-potential layer. The coverage audit itself did not read
labels or baseline residuals; enrichment and prediction analyses remain
separate.

PrismNet provides a denser cell-aware intermediate layer. Across its K562 and
HepG2 predictions, ILF3 covers 736 focal sites (1.60%), 1,937 sites within 25
nt (4.20%), and 7,718 duplexes (16.73%). Other duplex-level coverage ranges
from 1.27% for the RBFOX2 negative control to 21.49% for HNRNPM. Because these
are thresholded model calls rather than experimental peaks, they are analyzed
as predicted binding and compared directly with RBPNet and negative controls.

Machine-readable outputs are under
`results/adarpro/rbp_feature_audit/`, `data/derived/rbp/`, and
`data/derived/features/`.
