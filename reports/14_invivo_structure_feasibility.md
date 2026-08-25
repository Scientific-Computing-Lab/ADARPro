# 14 — In-vivo RNA-structure feasibility

## Public resource audit

An ENCODE API audit on 2026-08-21 found six released human icSHAPE
experiments: NAI-N3 and DMSO controls in GM12878, HepG2 and K562. None is an
exact match for tibial artery, cerebellum, liver tissue or skeletal muscle.
The released GRCh38 products include 12 enrichment TSVs (roughly 13–43 MB
each) plus multi-gigabyte BAMs and FASTQs. Enrichment rows are gene-centric
nucleotide arrays against GENCODE v24, not a simple genome-coordinate signal
track that can be overlapped with intronic IRAlu sites without transcript and
isoform reconstruction.

As a coverage upper bound, HepG2 file `ENCFF927ZZD` contains 1,152 Ensembl
genes, 1,101 of which match GENCODE v46 gene records. Only 61/5,967 Muscle
canonical development sites fall anywhere inside those genes, spanning 8/585
pairs (1.0% of sites and 1.4% of pairs). This deliberately overstates usable
coverage: a site being inside a measured gene does not mean that its intronic
coordinate appears in the processed transcript array. Repeat multi-mapping and
cell-line-to-tissue mismatch create additional uncertainty.

## Relation to the completed ensemble pilot

The native-gap ±200 nt RNAfold ensemble pilot is a different measurement. It
has broad pair coverage but lacks protein- and cell-conditioned structure, and
its best thermodynamic block produced only median internal ΔAUROC +.0031 and
ΔF1 +.0039 with confidence intervals crossing zero. The sparse ENCODE audit
does not rescue that branch, and the ensemble result does not falsify a future
well-matched DMS/icSHAPE experiment.

## Decision

Do not scale the current ENCODE cell-line icSHAPE route or download the raw
alignments. Its effective pair count is too small for the grouped benchmark,
and absence in repetitive Alu sequence would be hard to interpret. Re-open
direct in-vivo structure only for an exact or defensibly matched tissue/sample
resource with genome-resolvable signal and materially broader pair coverage.
Frozen RNA foundation representations and target-centered structural-contact
features remain separate, executable tests.

Audited endpoints:

- `https://www.encodeproject.org/search/?type=Experiment&assay_title=icSHAPE&format=json&limit=all`
- `https://www.encodeproject.org/search/?type=File&assay_title=icSHAPE&format=json&limit=all`
