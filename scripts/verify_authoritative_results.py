#!/usr/bin/env python3
"""Verify immutable ADARPro scientific identities and authoritative results."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/adarpro/final_locked_test"
BASELINE = "published_baseline_reproduced"
CANDIDATE = "adarpro_pair_local_equal_logit_ensemble"
EXPECTED = {
    "Artery": (0.923808, 0.947727, 0.926172, 0.948541, 0.867078, 0.886950),
    "Brain": (0.930142, 0.950771, 0.928624, 0.950353, 0.868908, 0.890197),
    "Liver": (0.909221, 0.941430, 0.924180, 0.950003, 0.854101, 0.881246),
    "MuscleSkeletal": (
        0.868750,
        0.914365,
        0.875660,
        0.920658,
        0.813939,
        0.854088,
    ),
    "Combined": (0.933100, 0.949104, 0.935624, 0.951327, 0.867397, 0.885215),
}
EXPECTED_MANIFEST_HASH = (
    "dd04f9786d52f57653da31b12dc777d10aa4247dfccd2b2c59ddff64ad395a1c"
)
EXPECTED_METRICS_HASH = (
    "ddd5352c48e84bb121e21d89e5d1777a3657caa06f28fa94ef1e8b3c58a74a0e"
)
# Evaluation 3 (the integration's selected predictor), pinned like the first.
FINAL_V3 = ROOT / "results/adarpro/final_locked_test_v3"
V3_CANDIDATE = "adarpro_gap_stage_graph_cnn_alone_5seed"
EXPECTED_V3_MANIFEST_HASH = (
    "568c8b9b397c75680be362f7912df0337ed535cd97f9b19d59a8525249ef628f"
)
EXPECTED_V3_METRICS_HASH = (
    "cb4d84c072380fd1c1c9c87c46d23def2a9d4eaf7cfde624d00f4706b658bfa0"
)
# Three development-only diagnostics were appended in revision 9fb11dc.
# Keep their identities and evidence regime explicit rather than relaxing the
# original frozen-campaign count to an open-ended minimum.
POST_HOC_DEVELOPMENT_RUNS = {
    "post_analysis_within_duplex_discrimination_v1": (
        "post_hoc_development_oof_within_pair_discrimination"
    ),
    "post_analysis_matched_site_tissue_variation_v1": (
        "post_hoc_matched_site_tissue_variation"
    ),
    "post_analysis_cross_tissue_pair_stability_v1": (
        "post_hoc_matched_site_pair_propensity_stability"
    ),
}
# The 2026-11 integration (branch Eli/updates-and-improvements) appended 273 runs,
# including the evaluation recorded in provenance/test_quarantine_v3.json. Keep
# the count exact and name every row with canonical-test access, as above.
INTEGRATION_RUN_COUNT = 273
CANONICAL_TEST_ACCESS_RUNS = {
    "final_locked_test_adarpro_pair_local_equal_logit_ensemble": (
        "single_consumed_canonical_test_evaluation"
    ),
    "final_figures_adarpro_pair_local_equal_logit_ensemble": (
        "post_lock_descriptive_figures"
    ),
    "final_locked_test_v3_adarpro_gap_stage_graph_cnn_alone_5seed": (
        "single_consumed_canonical_test_evaluation"
    ),
    "final_figures_adarpro_gap_stage_graph_cnn_alone_5seed": (
        "post_lock_descriptive_figures"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 5e-7) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} != {expected}")


def verify_metrics() -> None:
    path = FINAL / "metrics.csv"
    if sha256(path) != EXPECTED_METRICS_HASH:
        raise AssertionError("locked metrics hash mismatch")
    frame = pd.read_csv(path).set_index(["context", "model"])
    for context, expected in EXPECTED.items():
        baseline = frame.loc[(context, BASELINE)]
        candidate = frame.loc[(context, CANDIDATE)]
        values = (
            baseline.auroc,
            candidate.auroc,
            baseline.auprc,
            candidate.auprc,
            baseline.f1,
            candidate.f1,
        )
        for actual, target in zip(values, expected, strict=True):
            close(float(actual), target)
        if not all(candidate[key] > baseline[key] for key in ("auroc", "auprc", "f1")):
            raise AssertionError(f"{context}: non-positive final improvement")
        if not candidate.brier < baseline.brier:
            raise AssertionError(f"{context}: Brier score did not improve")
    deltas = []
    for context in EXPECTED:
        baseline = frame.loc[(context, BASELINE)]
        candidate = frame.loc[(context, CANDIDATE)]
        deltas.append([candidate[m] - baseline[m] for m in ("auroc", "auprc", "f1")])
    macro = pd.DataFrame(deltas).mean().tolist()
    for actual, target in zip(macro, (0.0277, 0.0261, 0.0253), strict=True):
        close(actual, target, tolerance=5e-5)


def verify_intervals_and_false_positives() -> None:
    expected_fp = {
        "Artery": (460, 300),
        "Brain": (425, 316),
        "Liver": (399, 240),
        "MuscleSkeletal": (188, 146),
        "Combined": (459, 428),
    }
    metrics = pd.read_csv(FINAL / "metrics.csv").set_index(["context", "model"])
    for context, (baseline_fp, candidate_fp) in expected_fp.items():
        if int(metrics.loc[(context, BASELINE), "fp"]) != baseline_fp:
            raise AssertionError(f"{context}: baseline FP mismatch")
        if int(metrics.loc[(context, CANDIDATE), "fp"]) != candidate_fp:
            raise AssertionError(f"{context}: candidate FP mismatch")
        ci = pd.read_csv(FINAL / f"{context}_pair_bootstrap_ci.csv")
        for metric in ("auroc", "auprc", "f1"):
            row = ci.loc[ci.metric == metric].iloc[0]
            if float(row.lower) <= 0:
                raise AssertionError(f"{context}: {metric} CI is not positive")
        matched_ci = pd.read_csv(
            ROOT
            / "figures/final/source_tables/final_matched_recall_pair_bootstrap_ci.csv"
        )
        matched = matched_ci.loc[
            (matched_ci.context == context)
            & (matched_ci.metric == "false_positives")
        ].iloc[0]
        if float(matched.upper) >= 0:
            raise AssertionError(f"{context}: matched-recall FP CI is not negative")


def verify_freeze() -> None:
    manifest_path = ROOT / "provenance/final_pipeline_manifest.json"
    if sha256(manifest_path) != EXPECTED_MANIFEST_HASH:
        raise AssertionError("final pipeline manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["candidate_specification"]["weights"] != [0.5, 0.5]:
        raise AssertionError("ensemble weights changed")
    for context in manifest["contexts"].values():
        if context["weights"] != [0.5, 0.5]:
            raise AssertionError("context ensemble weights changed")
        for component in context["components"]:
            path = ROOT / component["checkpoint"]
            if sha256(path) != component["sha256"]:
                raise AssertionError(f"checkpoint mismatch: {path}")
    quarantine = json.loads((ROOT / "provenance/test_quarantine.json").read_text())
    final = quarantine["final_locked_test"]
    if quarantine["state"] != "final_consumed" or final["evaluations_consumed"] != 1:
        raise AssertionError("canonical-test single-use state mismatch")
    if final["freeze_manifest"]["sha256"] != EXPECTED_MANIFEST_HASH:
        raise AssertionError("quarantine manifest identity mismatch")
    unified = json.loads(
        (ROOT / "provenance/unified_development_decision.json").read_text()
    )
    if unified["canonical_test_access"]:
        raise AssertionError("unified development unexpectedly accessed test")
    decision = unified["decision"]
    if decision["unified_family_selected"] or decision["final_development_winner"] != (
        "pair_balanced_plus_target_local_equal_logit"
    ):
        raise AssertionError("unified development decision changed")


def verify_registry() -> None:
    path = ROOT / "experiments/experiments.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["run_id"] for row in rows]
    expected_count = 128 + len(POST_HOC_DEVELOPMENT_RUNS) + INTEGRATION_RUN_COUNT
    if len(ids) != expected_count or len(ids) != len(set(ids)):
        raise AssertionError(
            f"experiment registry must contain {expected_count} unique runs"
        )
    by_id = {row["run_id"]: row for row in rows}
    for run_id, experiment_type in POST_HOC_DEVELOPMENT_RUNS.items():
        row = by_id.get(run_id)
        if (
            row is None
            or row["experiment_type"] != experiment_type
            or row["canonical_test_access"].lower() != "false"
        ):
            raise AssertionError(f"post-hoc development identity mismatch: {run_id}")
    test_access = {
        row["run_id"]: row["experiment_type"]
        for row in rows
        if row["canonical_test_access"].lower() == "true"
    }
    if test_access != CANONICAL_TEST_ACCESS_RUNS:
        raise AssertionError("canonical-test access differs from the named runs")
    final_ids = {
        row["run_id"]
        for row in rows
        if row["experiment_type"] == "single_consumed_canonical_test_evaluation"
    }
    named = {
        run_id
        for run_id, kind in CANONICAL_TEST_ACCESS_RUNS.items()
        if kind == "single_consumed_canonical_test_evaluation"
    }
    if final_ids != named:
        raise AssertionError(
            "registry must contain exactly the two named canonical-test evaluations"
        )


def verify_evaluation3() -> None:
    """Evaluation 3: identities recorded at the session, and the selected predictor beats AdarEdit everywhere."""
    manifest = ROOT / "provenance/final_pipeline_manifest_v3.json"
    if sha256(manifest) != EXPECTED_V3_MANIFEST_HASH:
        raise AssertionError("evaluation 3 manifest hash mismatch")
    if sha256(FINAL_V3 / "metrics.csv") != EXPECTED_V3_METRICS_HASH:
        raise AssertionError("evaluation 3 metrics hash mismatch")
    summary = json.loads((FINAL_V3 / "summary.json").read_text())
    if (
        summary["metrics_sha256"] != EXPECTED_V3_METRICS_HASH
        or summary["manifest_sha256"] != EXPECTED_V3_MANIFEST_HASH
        or summary["evaluation_number"] != 3
    ):
        raise AssertionError("evaluation 3 summary identity mismatch")
    quarantine = json.loads((ROOT / "provenance/test_quarantine_v3.json").read_text())
    final = quarantine["final_locked_test"]
    if (
        quarantine["state"] != "final_consumed"
        or quarantine["evaluation_number"] != 3
        or final["evaluations_consumed"] != 1
        or final["freeze_manifest"]["sha256"] != EXPECTED_V3_MANIFEST_HASH
    ):
        raise AssertionError("evaluation 3 quarantine state mismatch")
    metrics = pd.read_csv(FINAL_V3 / "metrics.csv").set_index(["model", "context"])
    for context in EXPECTED:
        base, cand = metrics.loc[(BASELINE, context)], metrics.loc[(V3_CANDIDATE, context)]
        for metric in ("auroc", "auprc", "f1"):
            if not cand[metric] > base[metric]:
                raise AssertionError(f"evaluation 3 {context}: {metric} does not exceed AdarEdit")


def main() -> None:
    verify_metrics()
    verify_intervals_and_false_positives()
    verify_freeze()
    verify_evaluation3()
    verify_registry()
    print(
        "PASS: authoritative results (first evaluation and evaluation 3), freeze, quarantine, "
        "and registry verified"
    )


if __name__ == "__main__":
    main()
