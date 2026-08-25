# 02 — RBP literature and resource survey

Status: active survey through 2026-08-21. The evidence-ranked panel was fixed
before RBP feature performance was evaluated; see
`context/rbp/candidate_panel.csv`.

Five documented PubMed searches covered DZF proteins, the seeded non-DZF
panel, all recent 2024–2026 RBP regulators, competitive/shielding mechanisms,
and chromatin-linked future layers. Search strings, result counts, and a
machine-readable evidence table are preserved in
`context/rbp/literature_searches.csv` and `literature_evidence.csv`.

## Mechanistic anchor

Freund et al. (Cell Reports 2020, DOI 10.1016/j.celrep.2020.107656) used ADAR1/
ADAR2 BioID, ENCODE knockdown RNA-seq, eCLIP, co-immunoprecipitation, and
overexpression. The four human DZF proteins ILF2, ILF3, STRBP, and ZFR interact
with ADARs. ILF3 is a broad negative regulator, requires its dsRNA-binding
domains to suppress editing, and binds near regulated sites. STRBP also
suppressed editing under overexpression, whereas ZFR perturbation evidence is
consistent with positive regulation, particularly in neuronal/ADAR2 settings.

Park et al. (bioRxiv 2026, DOI 10.64898/2026.01.14.699349) provides unusually
direct support for the primary Alu hypothesis: ILF2/3 binds Alu elements in
chromatin-associated RNA, acute degradation increases Alu A-to-I editing, and
the authors propose direct shielding from ADAR1.

Yi et al. (Nature Communications 2026, DOI
10.1038/s41467-026-71207-3) shows that EZH2 competes with ILF2 for ADAR1 binding
and bidirectionally changes ADAR1 substrate selectivity. This supports a
competitive/cooperative interaction layer but also warns against forcing every
RBP into a unidirectional suppressor score.

The systematic screen also recovers Quinones-Valdez et al. (Communications
Biology 2019, DOI 10.1038/s42003-018-0271-8), which perturbed more than 200 RBPs
in K562/HepG2 and identified NF45/90 (ILF2/3), DROSHA, and Ro60/TROVE2 through
ADAR1 abundance, interaction, or Alu binding. Its strong cell-type dependence
is a central reason to keep binding potential separate from tissue abundance.

## Resource and predictor assessment

| Resource | Evidence/input | Cell specificity | Scale/local use | Main limitation for ADARpro |
|---|---|---|---|---|
| ENCODE eCLIP | Experimental crosslink peaks | K562/HepG2 | Batch-downloadable GRCh38 BED; open ENCODE data | Repeat/multi-mapping makes absence in inverted Alu uncertain; cell lines do not match GTEx tissues |
| PrismNet (2021; DOI 10.1038/s41422-021-00476-y) | Sequence + matched in-vivo RNA structure + CLIP | Explicit; seven profiled cell types, 168 RBPs | Models/downloads advertised; candidate-only feasibility audit pending | Requires a compatible in-vivo structurome; predicted AdarEdit MFE structure is not equivalent |
| RBPNet (2023; DOI 10.1186/s13059-023-03015-7) | Sequence-to-CLIP signal at base resolution | One trained CLIP context/model | Local batch inference; 104 HepG2 models are available in a checksum-verified local cache | Learns protocol/cell-line signal and sequence biases; models do not cover ILF2/STRBP/ZFR or ADAR |
| DeepCLIP (2020; DOI 10.1093/nar/gkaa530) | Sequence-only binding/profile model | Not inherently cell aware | Stand-alone and web tool | Model coverage and maintenance must be audited; cannot express abundance or in-vivo structure |
| RBPmap (2014/2022; DOI 10.1093/nar/gku406) | Experimentally defined motif/PSSM with motif-environment scoring | Region background, not abundance | Stand-alone/web; scalable motif screen | Motif presence is binding potential, not occupancy; many dsRNA interactions are poorly represented by short motifs |
| RBP2GO 2.0 (2026; DOI 10.1093/nar/gkaf1258) | RBP identity, function, disease and sequence features across 13 species | Annotation rather than occupancy | Download/web resource | Useful for priors/localization, not site-level binding evidence |
| iDeepB (2025; DOI 10.1093/nar/gkaf748) | Sequence + cell-line gene expression, base-resolution profiles | Explicit expression-aware training | Current candidate for later stage | Frozen pretrained model coverage/local release could not be established |
| PaRPI (2025; DOI 10.1038/s42003-025-08807-0) | Cross-protocol/cross-batch RBP prediction | Designed for transfer | Candidate for later stage | Must establish usable pretrained human models and licensing before inclusion |

The first feature pass uses replicated GRCh38 ENCODE peaks because it is
transparent, inexpensive, and falsifiable. Predictive evidence is explicitly a
complement, not a replacement, when eCLIP absence is unreliable in repeats.

The current PrismNet download page was then verified directly. It supplies
hg38 scored prediction BEDs for 171 RBPs in each of K562 and HepG2. Both files
were frozen and checksummed; 13 prespecified panel proteins are present in both.
The repository license file is MIT, although its README also says
"non-commercial purposes"; that inconsistency is recorded and would require
clarification before commercial redistribution.

## Prespecified panels

The core mechanistic panel is ADAR, ILF3, ILF2, STRBP, ZFR, XRCC6, TROVE2, and
DROSHA. The extended panel adds MATR3, AUH, PUS1, FXR1, PABPC1, HNRNPC, and
HNRNPM. RBFOX2, TIA1, IGF2BP1, and SLBP are matched-coverage negative controls.
Biological-prior scores are ordinal 0–2 evidence components and are kept
separate from predictive performance.

ENCODE currently returns 24 released, replicate-combined GRCh38 peak tracks
across ILF3, TROVE2, XRCC6, DROSHA, MATR3, HNRNPC, HNRNPM, RBFOX2, TIA1,
IGF2BP1, FXR1, PUS1, and SLBP. The only ADAR track is archived and is retained
solely as a labeled legacy positive control, bringing the audited total to 25.
AUH has eCLIP experiments but no released replicate-combined GRCh38 peak file.
No experiment was returned under ILF2, STRBP, ZFR, PABPC1, or EZH2. Absence
from ENCODE is not biological absence.

The frozen RBPNet release (Zenodo 10185223, DOI
10.5281/zenodo.10185223; 2023; CC-BY-4.0 models, MIT code) contains 104 HepG2
models. Exact model coverage for the prespecified panel is ILF3, XRCC6, TROVE2,
DROSHA, MATR3, HNRNPC, HNRNPM, RBFOX2, TIA1, and IGF2BP1. Its archive MD5
matches Zenodo (`e88fb57483ba9a3d81b842cc5aa140fb`). These models predict a
relative CLIP signal profile along an input sequence, not an absolute binding
probability; ADARpro preserves that distinction in feature names and reports.
