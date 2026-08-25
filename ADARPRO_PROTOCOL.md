# ADARpro research protocol

## Evaluation boundary

The canonical AdarEdit `test` pairs are not development data. ADARpro uses
only the original `train` and `valid` pair union for grouped folds, feature
selection, normalization, hyperparameter selection, thresholds, biological
candidate selection, and model comparison.

There are only two permitted test-access modes:

1. `published_reproduction`: supplied upstream checkpoints may be evaluated
   and only aggregate metrics may be persisted. No per-site test output is
   written by the ADARpro reproduction command.
2. `final_locked_test`: after a manifest of code, data, features,
   hyperparameters, thresholds, and candidate RBPs is frozen, one final
   evaluation may consume the single-use authorization. Any later test-error
   work is descriptive and cannot reopen model development.

All ADARpro development loaders call the quarantine guard. The guard rejects
paths or split names containing the canonical test partition and records
authorized access in an append-only hash-chained audit log.

## Development design

The published supplied-checkpoint held-out benchmark is the authoritative
AdarEdit performance reference and the final target to beat. Grouped OOF scores
are not a replacement benchmark: because each outer model uses only part of
the development data, their absolute values are expected to be lower. They are
used only to compare incremental signal under identical folds during research.

- Grouping unit: canonical `pair_id`.
- Biological individual tissues: Artery Tibial, Brain/Cerebellum, Liver, and
  Muscle Skeletal.
- `Combined` is a pooled GTEx endpoint and is never assigned a fabricated
  tissue expression vector or treated as a fifth tissue in matched-site
  analyses.
- Published/final reference: the supplied per-context Baseline GAT evaluated
  under the original canonical held-out protocol.
- Development reference: the same Baseline GAT architecture cross-fitted on
  train+validation pairs solely for leakage-free within-fold comparisons.
- Residual feature tests compare the baseline out-of-fold logit with the same
  logit plus new context features in identical grouped folds. Neural successor
  tests preserve those folds, the inner-fold checkpoint/threshold rule and the
  canonical task while varying one prespecified source of information or
  modeling assumption at a time.
- All learned preprocessing and data-driven RBP selection is nested inside
  training folds.
- Statistical unit for uncertainty: Alu pair, using paired clustered
  bootstrap differences.

## Prespecified feature scales and controls

RBP occupancy summaries use focal, 5, 10, 25, 50, and 100 nucleotide windows
plus duplex-wide burden. These windows are biological prespecifications; any
subset selection happens in nested development CV. Mandatory controls include
permuted RBP identities, shuffled tissue expression, shifted genomic binding,
negative-control RBPs, matched-dimensional random features, tissue ID alone,
binding alone, and abundance alone.

## Freeze criteria

Before final-test authorization, the frozen manifest must name the exact git
revision, environment, development folds, input hashes, selected biological
variables and feature/context schema, fitted-development artifacts, model
architecture and hyperparameters, decision thresholds, and analysis scripts.
The final result is reported regardless of direction or statistical
significance.
