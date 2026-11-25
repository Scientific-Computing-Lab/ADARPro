"""Task 3c: how often do the knife-edge flagship - equal-average verdicts flip across bootstrap seeds?
Brier and F1 only (fast numpy), 2000 pair resamples per seed, 40 seeds (1..40), my resampler, every endpoint.
Outputs: s03c_borderline_seeds.csv, s03c_borderline_seeds.txt
"""
from __future__ import annotations

import sys

sys.path.insert(0, "<this folder>")
import numpy as np
import pandas as pd

from common import CONTEXTS, OUT, load

REPS, SEEDS = 2000, range(1, 41)
rows = []
for i, ctx in enumerate(CONTEXTS):
    t = load(ctx)
    y = t["label"].to_numpy()
    pf, pe = t["probability"].to_numpy(), t["equal_average_probability"].to_numpy()
    cf, ce = t["prediction"].to_numpy(), t["equal_average_prediction"].to_numpy()
    sqf, sqe = (pf - y) ** 2, (pe - y) ** 2
    pv = t["pair_id"].to_numpy()
    pairs = pd.unique(pv)
    pos = {q: j for j, q in enumerate(pairs)}
    pidx = np.array([pos[q] for q in pv])
    # per-pair sufficient statistics: rows, squared-error sums, TP/FP/FN counts
    def agg(v):
        return np.bincount(pidx, weights=v, minlength=len(pairs))
    n = agg(np.ones(len(y)))
    sf, se = agg(sqf), agg(sqe)
    tpf, fpf, fnf = agg((y == 1) & (cf == 1)), agg((y == 0) & (cf == 1)), agg((y == 1) & (cf == 0))
    tpe, fpe, fne = agg((y == 1) & (ce == 1)), agg((y == 0) & (ce == 1)), agg((y == 1) & (ce == 0))
    for seed in SEEDS:
        rng = np.random.default_rng(seed * 100 + i)
        W = np.zeros((REPS, len(pairs)))
        k = rng.integers(0, len(pairs), size=(REPS, len(pairs)))
        np.add.at(W, (np.repeat(np.arange(REPS), len(pairs)), k.ravel()), 1)
        N = W @ n
        dbrier = (W @ sf - W @ se) / N
        f1f = 2 * (W @ tpf) / (2 * (W @ tpf) + W @ fpf + W @ fnf)
        f1e = 2 * (W @ tpe) / (2 * (W @ tpe) + W @ fpe + W @ fne)
        for metric, d in (("brier", dbrier), ("f1", f1f - f1e)):
            lo, hi = np.percentile(d, [2.5, 97.5])
            good = (hi < 0) if metric == "brier" else (lo > 0)
            bad = (lo > 0) if metric == "brier" else (hi < 0)
            rows.append({"context": ctx, "metric": metric, "seed": seed, "lower": lo, "upper": hi,
                         "verdict": "better" if good else ("worse" if bad else "cannot be told apart")})
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/s03c_borderline_seeds.csv", index=False)
summary = (R.groupby(["context", "metric"])
           .agg(seeds=("seed", "size"), better=("verdict", lambda s: int((s == "better").sum())),
                cannot=("verdict", lambda s: int((s == "cannot be told apart").sum())),
                lower_min=("lower", "min"), lower_max=("lower", "max"), upper_min=("upper", "min"), upper_max=("upper", "max"))
           .reset_index())
txt = "flagship - equal average, 40 bootstrap seeds x 2000 pair resamples (sufficient-statistic implementation)\n"
txt += summary.round(5).to_string(index=False) + "\n"
open(f"{OUT}/s03c_borderline_seeds.txt", "w").write(txt)
print(txt)
