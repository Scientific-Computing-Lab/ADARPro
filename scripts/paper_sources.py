#!/usr/bin/env python3
"""Development result files the paper scripts read: PNAS's run -> the run that re-produces it on our models (D48).

The paper scripts (`build_paper_claim_ledger.py`, `render_paper_figures.py`, `render_paper_tables.py`) name PNAS's
result paths at their read sites and pass them through this module, which returns the clone file that replaces each
one and applies the row filter that makes the new file hold the same rows as PNAS's.

The map is COMPUTE's source map (working folder `step11b/claim_sources.csv`, written by `step11b/claim_sources.py`;
steps 6, 11b and 11c, all FINAL), plus `SIBLINGS`: files of the same PNAS run directories that the scripts also read
and that COMPUTE's new directories hold under the same name.

Row filters. Most new files keep PNAS's role names and columns, so the scripts' own role selection stays as it is and
no filter is needed. Two need one:
  * PNAS's `binary_pair_balanced/<endpoint>/pair_bootstrap_ci.csv` has no role columns (it holds only graph minus
    reference); the step-11b ladder holds every candidate against every other, so its graph-minus-reference rows are
    selected.
  * the step-11b natural-prevalence `metrics.csv` adds the gap stage (flagship, "Selected predictor") to PNAS's four
    models; the rows are restricted to those five (WRITING shows the selected predictor in Fig. S3 and Table S7,
    D48), so any further role COMPUTE adds there does not reach the paper unannounced.

Test-set files (results/adarpro/final_locked_test*, figures/final*) and PNAS side branches that were never re-run are
not in the map: the scripts read them directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = "experiments/runs"
ENDPOINTS = ("Artery", "Brain", "Combined", "Liver", "MuscleSkeletal")
PNAS_NATURAL_MODELS = ("development_reference", "pair_balanced", "local", "pair_local_ensemble")
NATURAL_MODELS = PNAS_NATURAL_MODELS + ("gap_stage",)  # PNAS's four and the selected predictor

# PNAS path -> (new path, row filter {column: value or tuple of allowed values}); paths relative to the clone root.
SOURCES: dict[str, tuple[str, dict[str, str | tuple[str, ...]]]] = {}
for _endpoint in ENDPOINTS:
    SOURCES[f"{RUNS}/richer_supervision/binary_pair_balanced/{_endpoint}/pair_bootstrap_ci.csv"] = (
        f"{RUNS}/step11b/ladder/{_endpoint}/pair_bootstrap_ci.csv",
        {"candidate": "pair_balanced", "reference": "development_reference"},
    )
for _endpoint in ("Liver", "MuscleSkeletal"):
    SOURCES[f"{RUNS}/richer_supervision/natural_prevalence/pair_local_comparison/{_endpoint}/metrics.csv"] = (
        f"{RUNS}/step11b/natural_prevalence/comparison/{_endpoint}/metrics.csv",
        {"model": NATURAL_MODELS},
    )
    SOURCES[f"{RUNS}/richer_supervision/pair_local_ensemble_comparison/{_endpoint}/pair_bootstrap_ci.csv"] = (
        f"{RUNS}/step6/richer_supervision/pair_local_ensemble_comparison/{_endpoint}/pair_bootstrap_ci.csv",
        {},
    )
for _pnas, _new in (
    # step 11b: PNAS's graph experiments retrained in our campaign, and the flagship's post-hoc analyses
    ("richer_supervision/pair_balance_comparison/metrics.csv", "step11b/richer_supervision/pair_balance_comparison/metrics.csv"),
    ("richer_supervision/pair_balance_comparison/pair_bootstrap_ci.csv",
     "step11b/richer_supervision/pair_balance_comparison/pair_bootstrap_ci.csv"),
    ("richer_supervision/pooling_comparison/metrics.csv", "step11b/richer_supervision/pooling_comparison/metrics.csv"),
    ("richer_supervision/pooling_comparison/pair_bootstrap_ci.csv",
     "step11b/richer_supervision/pooling_comparison/pair_bootstrap_ci.csv"),
    ("richer_supervision/pair_balanced_multitask_comparison/pair_bootstrap_ci.csv",
     "step11b/richer_supervision/pair_balanced_multitask_comparison/pair_bootstrap_ci.csv"),
    ("post_analysis/within_duplex/within_duplex_contrasts.csv", "step11b/post_analysis/within_duplex/within_duplex_contrasts.csv"),
    ("post_analysis/within_duplex/within_duplex_summary.csv", "step11b/post_analysis/within_duplex/within_duplex_summary.csv"),
    ("target_local_interpretation/MuscleSkeletal/geometry_strata.csv",
     "step11b/target_local_interpretation/MuscleSkeletal/geometry_strata.csv"),
    # step 6: the CNN fix and graph + CNN alone
    ("richer_supervision/local_alignment_control/MuscleSkeletal/metrics.csv",
     "step6/richer_supervision/local_alignment_control/MuscleSkeletal/metrics.csv"),
    ("richer_supervision/local_alignment_control/MuscleSkeletal/pair_bootstrap_ci.csv",
     "step6/richer_supervision/local_alignment_control/MuscleSkeletal/pair_bootstrap_ci.csv"),
    ("unified/comparison_primary/MuscleSkeletal/pair_bootstrap_ci.csv",
     "step6/unified/comparison_primary/MuscleSkeletal/pair_bootstrap_ci.csv"),
    ("unified/comparison_primary/MuscleSkeletal/metrics.csv", "step6/unified/comparison_primary/MuscleSkeletal/metrics.csv"),
    # step 11c: the three graph analyses R4 found; the parameter count of graph + CNN alone (R5a)
    ("richer_supervision/pair_balance_diagnostic/raw_site_count_quintiles.csv",
     "step11c/richer_supervision/pair_balance_diagnostic/raw_site_count_quintiles.csv"),
    ("richer_supervision/pooling_controls_final/pair_bootstrap_ci.csv",
     "step11c/richer_supervision/pooling_controls_final/pair_bootstrap_ci.csv"),
    ("richer_supervision/natural_prevalence/fixed_ensemble_multitask_comparison/pair_bootstrap_ci.csv",
     "step11c/richer_supervision/natural_prevalence/fixed_ensemble_multitask_comparison/pair_bootstrap_ci.csv"),
):
    SOURCES[f"{RUNS}/{_pnas}"] = (f"{RUNS}/{_new}", {})
# PNAS took the ensemble's parameter count from a report's prose; step 11c counts it from the checkpoints (JSON).
SOURCES["reports/20_unified_end_to_end_result.md"] = (f"{RUNS}/step11c/parameter_counts/MuscleSkeletal_graph_cnn_alone.json", {})

# Files of the same PNAS run directories that the scripts read, held by COMPUTE's new directories under the same name.
SIBLINGS: dict[str, str] = {
    f"{RUNS}/richer_supervision/pair_local_ensemble_comparison/{_endpoint}/metrics.csv":
        f"{RUNS}/step6/richer_supervision/pair_local_ensemble_comparison/{_endpoint}/metrics.csv"
    for _endpoint in ("Liver", "MuscleSkeletal")
}
SIBLINGS.update({
    f"{RUNS}/richer_supervision/pair_local_ensemble_comparison/MuscleSkeletal/summary.json":
        f"{RUNS}/step6/richer_supervision/pair_local_ensemble_comparison/MuscleSkeletal/summary.json",
    f"{RUNS}/richer_supervision/local_alignment_control/MuscleSkeletal/summary.json":
        f"{RUNS}/step6/richer_supervision/local_alignment_control/MuscleSkeletal/summary.json",
    # the unified model's own training run, retrained in step 6 beside its comparison_primary evaluation
    f"{RUNS}/unified/unified_target_local_interaction_aux_v1/MuscleSkeletal/summary.json":
        f"{RUNS}/step6/unified/unified_target_local_interaction_aux_v1/MuscleSkeletal/summary.json",
    f"{RUNS}/unified/unified_target_local_interaction_aux_v1/MuscleSkeletal/outer_0/config.json":
        f"{RUNS}/step6/unified/unified_target_local_interaction_aux_v1/MuscleSkeletal/outer_0/config.json",
})
for _pnas, _new in SIBLINGS.items():
    SOURCES[_pnas] = (_new, {})


def _key(pnas: str | Path) -> str:
    path = Path(pnas)
    if path.is_absolute():
        path = path.relative_to(ROOT)
    key = path.as_posix()
    if key not in SOURCES:
        raise KeyError(f"{key} is not a mapped PNAS source (scripts/paper_sources.py)")
    return key


def path(pnas: str | Path) -> Path:
    """The clone file (absolute) that replaces PNAS's result path `pnas` (absolute, or relative to the clone root)."""
    return ROOT / SOURCES[_key(pnas)][0]


def relpath(pnas: str | Path) -> str:
    """As `path`, relative to the clone root; only for files that need no row filter."""
    new, row_filter = SOURCES[_key(pnas)]
    if row_filter:
        raise ValueError(f"{new} needs the row filter {row_filter}: read it with paper_sources.read_csv")
    return new


def read_csv(pnas: str | Path, **kwargs) -> pd.DataFrame:
    """The replacement of PNAS's CSV `pnas`, restricted to the rows that correspond to PNAS's file."""
    new, row_filter = SOURCES[_key(pnas)]
    frame = pd.read_csv(ROOT / new, **kwargs)
    for column, allowed in row_filter.items():
        allowed = allowed if isinstance(allowed, tuple) else (allowed,)
        frame = frame[frame[column].isin(allowed)]
    if frame.empty:
        raise RuntimeError(f"{new}: no row matches {row_filter}")
    return frame.reset_index(drop=True)


def read_json(pnas: str | Path) -> dict:
    """The replacement of PNAS's JSON file `pnas`."""
    return json.loads(path(pnas).read_text(encoding="utf-8"))
