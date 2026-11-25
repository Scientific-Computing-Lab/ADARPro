#!/usr/bin/env python3
"""Spread of the spacing (gap-stage) controls across the five tracked training runs (development only).

The printed controls (SI Table S4) come from one training run, 20260821, compared by
`python -m adarpro.evaluation.richer_supervision` (experiments/runs/step11b/gap_controls/<endpoint>/). The same module,
with the same settings (2,000 pair resamples, bootstrap seed 20260821), was run on the four further step-9 runs
(experiments/runs/step11b/gap_controls_runs/seed<run>/<endpoint>/). This script reads the five pair_bootstrap_ci.csv
files per endpoint and writes, for the selected second stage minus each control:

  run_spread.csv          one row per endpoint x reference x metric x run (median and 95% pair-clustered interval)
  run_spread_summary.csv  one row per endpoint x reference x metric: runs whose interval excludes zero in the
                          favourable direction, mean / min / max of the run medians, the printed run's median and rank

Differences are candidate minus reference on the 0-1 scale (0.01 = one percentage point). Favourable: AUROC, AUPRC and
F1 above zero; Brier below zero. Development out-of-fold predictions only; no canonical-test file is read.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["20260821", "20265821", "20270821", "20275821", "20280821"]
PRINTED = "20260821"
ENDPOINTS = ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]
REFERENCES = ["no_gap_stage", "shuffled_gap_stage", "equal_average"]
METRICS = ["auroc", "auprc", "f1", "brier"]


def source(run: str, endpoint: str) -> Path:
    if run == PRINTED:
        return ROOT / "experiments/runs/step11b/gap_controls" / endpoint / "pair_bootstrap_ci.csv"
    return ROOT / "experiments/runs/step11b/gap_controls_runs" / f"seed{run}" / endpoint / "pair_bootstrap_ci.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/runs/step11b/gap_controls_runs")
    args = parser.parse_args()
    rows = []
    for run in RUNS:
        for endpoint in ENDPOINTS:
            ci = pd.read_csv(source(run, endpoint))
            for reference in REFERENCES:
                for metric in METRICS:
                    hit = ci[(ci.candidate == "gap_stage") & (ci.reference == reference) & (ci.metric == metric)]
                    if len(hit) != 1:
                        raise AssertionError(f"{source(run, endpoint)}: {len(hit)} rows for gap_stage vs {reference} {metric}")
                    r = hit.iloc[0]
                    favourable = (r.upper < 0) if metric == "brier" else (r.lower > 0)
                    rows.append(dict(endpoint=endpoint, reference=reference, metric=metric, run=run,
                                     printed_run=run == PRINTED, lower=r.lower, median=r["median"], upper=r.upper,
                                     excludes_zero_favourable=bool(favourable)))
    long = pd.DataFrame(rows)
    # the printed run must reproduce the numbers SI Table S4 prints (same file the table reads)
    if long[long.printed_run].shape[0] != len(ENDPOINTS) * len(REFERENCES) * len(METRICS):
        raise AssertionError("printed run incomplete")
    summary = []
    for (endpoint, reference, metric), g in long.groupby(["endpoint", "reference", "metric"], sort=False):
        sign = -1 if metric == "brier" else 1
        printed = g[g.printed_run].iloc[0]
        summary.append(dict(
            endpoint=endpoint, reference=reference, metric=metric, runs=len(g),
            runs_excluding_zero_favourable=int(g.excludes_zero_favourable.sum()),
            mean_median=g["median"].mean(), min_median=g["median"].min(), max_median=g["median"].max(),
            printed_run_median=printed["median"],
            printed_run_rank_most_favourable=int((sign * g["median"] > sign * printed["median"]).sum()) + 1,
        ))
    summary = pd.DataFrame(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.output_dir / "run_spread.csv", index=False)
    summary.to_csv(args.output_dir / "run_spread_summary.csv", index=False)
    show = summary[summary.reference != "equal_average"].copy()
    for c in ("mean_median", "min_median", "max_median", "printed_run_median"):
        show[c] = (100 * show[c]).round(2)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
