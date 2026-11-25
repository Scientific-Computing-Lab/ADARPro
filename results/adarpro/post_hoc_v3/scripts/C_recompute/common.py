"""Shared helpers for the independent recompute audit (audit_step14/C_recompute).

Written from scratch with numpy / pandas / scikit-learn only. Nothing from adarpro is imported.
Definitions were READ (not imported) from these adarpro source lines, and are restated here:
  - logit clip: float64 machine epsilon, log(p/(1-p))       src/adarpro/models/gap_stage.py:58-64
  - 5-seed / 10-network average: mean of clipped logits, sigmoid  src/adarpro/models/fixed_checkpoint_ensemble.py:40-53
  - frozen stage: sigmoid(intercept + w . (x - mean)/scale)  src/adarpro/models/final_gap_stage.py:84-91
  - prediction rule: probability >= threshold                src/adarpro/evaluation/final_locked_gap_stage.py:391
  - metrics: sklearn average_precision_score, brier_score_loss  src/adarpro/models/residual.py:234-259
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn import metrics as skm

CLONE = "<repository>"
OUT = "<this folder>"
CONTEXTS = ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]
V3 = f"{CLONE}/results/adarpro/final_locked_test_v3"

# model key -> (probability column, prediction column, threshold column)
MODELS = {
    "flagship": ("probability", "prediction", "threshold"),
    "equal_average": ("equal_average_probability", "equal_average_prediction", "equal_average_threshold"),
    "pnas": ("pnas_probability", "pnas_prediction", "pnas_threshold"),
    "adaredit": ("baseline_probability", "baseline_prediction", "baseline_threshold"),
    "adaredit_cpu": ("baseline_cpu_probability", "baseline_cpu_prediction", "baseline_threshold"),
}
# session metrics.csv model names
SESSION_NAMES = {
    "flagship": "adarpro_gap_stage_graph_cnn_alone_5seed",
    "equal_average": "equal_average",
    "pnas": "pnas_selected_predictor",
    "adaredit": "published_baseline_reproduced",
}
EPS = np.finfo(np.float64).eps


def load(context: str) -> pd.DataFrame:
    return pd.read_parquet(f"{V3}/{context}_predictions.parquet")


def my_logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def my_sigmoid(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable, different code path from 1/(1+exp(-z))


def my_metrics(y, p, yhat) -> dict:
    """Nine metrics computed by hand where practical (confusion counts, F1, MCC, Brier, AUROC via ranks);
    AUPRC = step-wise average precision (sklearn's convention), checked against a hand implementation."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "auroc": auroc_rank(y, p),
        "auprc": ap_stepwise(y, p),
        "f1": f1, "precision": prec, "recall": rec, "specificity": spec,
        "balanced_accuracy": 0.5 * (rec + spec), "mcc": mcc,
        "brier": float(np.mean((p - y) ** 2)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def auroc_rank(y, p) -> float:
    """Mann-Whitney AUROC with mid-ranks for ties."""
    from scipy.stats import rankdata
    y = np.asarray(y, dtype=int)
    r = rankdata(p, method="average")
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ap_stepwise(y, p) -> float:
    """Average precision = sum over distinct thresholds (descending) of (R_k - R_{k-1}) * P_k (no interpolation)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    # last index of each run of equal scores
    distinct = np.where(np.diff(ps))[0]
    ends = np.r_[distinct, len(ps) - 1]
    ctp = np.cumsum(ys)[ends]
    cfp = (ends + 1) - ctp
    precision = ctp / (ctp + cfp)
    recall = ctp / ys.sum()
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def ap_trapezoid(y, p) -> float:
    precision, recall, _ = skm.precision_recall_curve(y, p)
    return float(skm.auc(recall, precision))


def fast_metrics(y, p, yhat) -> dict:
    """The four bootstrap metrics, computed with sklearn (a second, independent implementation from my_metrics)."""
    return {
        "auroc": skm.roc_auc_score(y, p),
        "auprc": skm.average_precision_score(y, p),
        "f1": skm.f1_score(y, yhat, zero_division=0),
        "brier": skm.brier_score_loss(y, p),
    }
