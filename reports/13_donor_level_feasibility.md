# 13 — Donor/sample-level feasibility

## Available outcome resolution

The downloaded AdarEdit source tables are tissue-level per-site aggregates.
They provide one `EditingIndex` per site and tissue, not donor identifiers,
donor-level edited/read counts, library covariates, or sample-level splicing.
The reconstructed development table therefore has 275,569 site–tissue rows
but zero site–donor outcomes. Public GTEx sample-level expression cannot repair
that mismatch: it would attach many donor covariate vectors to one pooled
editing label and create pseudo-replication rather than new supervision.

GTEx v8 individual-level sequence data are governed through dbGaP study
`phs000424.v9.p2`; the official study page directs users to Authorized Access.
The current workspace has no authorized donor BAM/CRAM data or corresponding
sample phenotype manifest. The aggregate tables' source and limitations are
documented in `data/raw/editing_levels/README.md`.

## Leakage-safe design if access becomes available

A valid donor study would have to reconstruct edited and total read counts for
each site × donor, apply genotype/SNP and mapping-quality filters, and retain
coverage rather than reducing every observation immediately to a percentage.
Model evaluation would need crossed biological boundaries:

1. the canonical global pair split remains the substrate boundary;
2. donor-held-out folds assess transfer to unseen people;
3. all transformations of expression, splicing, cell-composition and IFN state
   are fitted inside the relevant training fold;
4. repeated tissues from one donor remain in one donor fold;
5. binomial/beta-binomial likelihoods account for read depth and
   overdispersion.

Holding out only rows would be invalid because both the same pair and the same
donor could be memorized. Tissue-averaged labels also cannot support claims
about donor state.

## Decision

Defer donor/sample-level modeling under the current data authority. This is a
data-resolution constraint, not a negative result about the hypothesis. The
branch re-enters only after authorized sample-level editing counts can be
constructed and the crossed pair × donor split is frozen. Available compute is
better spent on raw high/low cross-tissue sharing, target-centered structure,
chromatin replication and frozen foundation representations, all of which are
testable without inventing donor outcomes.

The same identifiability problem applies to ADAR regulatory state. Public
tissue-median gene expression can supply total `ADAR`/`ADARB1` abundance, but
cannot resolve ADAR1 p110 versus interferon-inducible p150 activity,
localization, donor-specific interferon state or enzyme occupancy at a site.
With only four tissue-average environments, any isoform or IFN vector is also
nearly inseparable from tissue identity; the matched-switch analysis already
showed that tissue identity saturates that comparison. This branch is therefore
deferred under current outcome resolution rather than approximated with
gene-level TPM. Re-entry requires matched sample-level editing counts and
isoform/signature measurements under the crossed pair × donor design above.

Sources audited 2026-08-21:

- GTEx dbGaP study: `https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000424.v9.p2`
- GTEx Portal protected-data entry: `https://gtexportal.org/home/protectedDataPage`
