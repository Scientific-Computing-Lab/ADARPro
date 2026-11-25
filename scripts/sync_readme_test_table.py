#!/usr/bin/env python3
"""Write the README's "Locked canonical result" section from evaluation 3's aggregate files.

Every number comes from a file, and every sentence is asserted against the files before it is written:
  - the table: results/adarpro/final_locked_test_v3/metrics.csv (AdarEdit reproduced vs the selected predictor);
  - mean gains: the same file, checked equal to the paper's ledger values (paper/numerical_claim_ledger.csv);
  - "positive interval for all three differences": figures/final_v3/source_tables/final_metric_difference_ci.csv;
  - matched recall: figures/final_v3/source_tables/final_matched_recall.csv and
    final_matched_recall_pair_bootstrap_ci.csv.
`--check` exits 1 if the README differs from what would be written.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
V3 = ROOT / "results/adarpro/final_locked_test_v3/metrics.csv"
SOURCE = ROOT / "figures/final_v3/source_tables"
LEDGER = ROOT / "paper/numerical_claim_ledger.csv"
BASE, CAND = "published_baseline_reproduced", "adarpro_gap_stage_graph_cnn_alone_5seed"
CONTEXTS = [("Artery", "Artery"), ("Brain", "Brain"), ("Liver", "Liver"),
            ("MuscleSkeletal", "Skeletal muscle"), ("Combined", "Combined")]
METRICS = ["auroc", "auprc", "f1"]


def section() -> str:
    m = pd.read_csv(V3).set_index(["model", "context"])
    rows, diffs = [], {k: [] for k in METRICS}
    for ctx, label in CONTEXTS:
        b, c = m.loc[(BASE, ctx)], m.loc[(CAND, ctx)]
        cells = [b.auroc, c.auroc, b.auprc, c.auprc, b.f1, c.f1]
        rows.append(f"| {label} | " + " | ".join(f"{x:.4f}" for x in cells) + " |")
        for k in METRICS:
            diffs[k].append(c[k] - b[k])
    mean = {k: sum(v) / len(v) for k, v in diffs.items()}

    ledger = pd.read_csv(LEDGER).set_index("claim_id")
    names = {"auroc": "locked_macro_delta_auroc", "auprc": "locked_macro_delta_auprc", "f1": "locked_macro_delta_f1"}
    for k, cid in names.items():
        if abs(float(ledger.loc[cid, "exact_value"]) - mean[k]) > 1e-12:
            raise AssertionError(f"mean {k} gain differs from the paper's ledger ({cid})")
        if abs(float(ledger.loc[cid + "_points", "exact_value"]) - 100 * mean[k]) > 1e-9:
            raise AssertionError(f"mean {k} gain differs from the paper's ledger ({cid}_points)")
    points = {k: ledger.loc[names[k] + "_points", "display_value"] for k in METRICS}

    ci = pd.read_csv(SOURCE / "final_metric_difference_ci.csv")
    ci = ci[ci.metric.isin(METRICS)]
    if len(ci) != len(METRICS) * len(CONTEXTS):
        raise AssertionError("difference intervals: unexpected row count")
    all_positive = bool((ci.lower > 0).all())

    mr = pd.read_csv(SOURCE / "final_matched_recall.csv").set_index("context")
    mr_ci = pd.read_csv(SOURCE / "final_matched_recall_pair_bootstrap_ci.csv")
    mr_ci = mr_ci[mr_ci.metric == "false_positives"].set_index("context")
    avoided = [int(mr.loc[ctx, "false_positives_avoided"]) for ctx, _ in CONTEXTS]
    for ctx, _ in CONTEXTS:
        if int(mr.loc[ctx, "baseline_fp"] - mr.loc[ctx, "candidate_fp"]) != int(mr.loc[ctx, "false_positives_avoided"]):
            raise AssertionError(f"{ctx}: matched-recall counts do not add up")
    fp_ci_negative = bool((mr_ci.loc[[c for c, _ in CONTEXTS], "upper"] < 0).all())

    lines = [
        "## Locked canonical result",
        "",
        "The complete selected predictor was frozen before the canonical held-out",
        "evaluation, and the reported results of the selected predictor and the",
        "equal-weight average were produced in a single session. The table below",
        "compares the reproduced supplied-checkpoint AdarEdit baseline with the frozen",
        "selected predictor on the original published task and split.",
        "",
        "| Context | AdarEdit AUROC | Selected predictor AUROC | AdarEdit AUPRC | Selected predictor AUPRC | AdarEdit F1 | Selected predictor F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        (
            "Mean gains across the five reported benchmark endpoints were "
            f"**+{points['auroc']} AUROC, +{points['auprc']} AUPRC, and +{points['f1']} F1 percentage points**."
        ),
    ]
    if all_positive:
        lines.append("Every endpoint had a positive pair-clustered 95% interval for all three differences.")
    lines.append(
        "In a post-test descriptive analysis at matched reproduced-baseline recall, the selected predictor avoided "
        + ", ".join(str(a) for a in avoided[:-1]) + f", and {avoided[-1]} false positives, respectively"
        + (", with clustered intervals excluding zero throughout." if fp_ci_negative else ".")
    )
    lines += [
        "",
        "These held-out results are the authoritative performance claim. The lower",
        "absolute grouped out-of-fold scores in development are an internal research",
        "instrument used to compare candidates under identical folds; they do",
        "not replace or redefine the published benchmark.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = README.read_text(encoding="utf-8")
    match = re.search(r"^## Locked canonical result\n.*?(?=^## )", text, flags=re.DOTALL | re.MULTILINE)
    if not match:
        raise AssertionError("README: 'Locked canonical result' section not found")
    new = text[: match.start()] + section() + text[match.end():]
    if args.check:
        sys.exit(0 if new == text else 1)
    README.write_text(new, encoding="utf-8")
    print("README locked-result section written from evaluation 3")


if __name__ == "__main__":
    main()
