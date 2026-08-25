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
    if len(ids) != 128 or len(ids) != len(set(ids)):
        raise AssertionError("experiment registry must contain 128 unique runs")
    final_rows = [
        row
        for row in rows
        if row["experiment_type"] == "single_consumed_canonical_test_evaluation"
    ]
    if len(final_rows) != 1 or final_rows[0]["canonical_test_access"].lower() != "true":
        raise AssertionError("registry must contain exactly one canonical-test evaluation")


def main() -> None:
    verify_metrics()
    verify_intervals_and_false_positives()
    verify_freeze()
    verify_registry()
    print("PASS: authoritative results, freeze, quarantine, and registry verified")


if __name__ == "__main__":
    main()
