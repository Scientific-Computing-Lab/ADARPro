"""Claim (i) "the gap stage helps on the test mainly through ranking": descriptive decomposition, no refitting.

For flagship vs equal average (same 10 networks), per endpoint:
  1. Brier = calibration loss + refinement loss, refinement = Brier after the best monotone (isotonic, PAV)
     recalibration of the score on these same rows; calibration loss = Brier - refinement. Refinement depends on
     the ranking only, so a Brier gain that sits in refinement is a ranking gain.
  2. F1 at the frozen threshold vs the best F1 any threshold could reach on these rows (an upper bound).
  3. Gap-term ablation of the FROZEN stage: score_nogap = w_g z_g + w_c z_c (z_gap fixed at 0, i.e. the training
     mean). No weight is refitted. AUROC/AUPRC of that score separates "re-weighting graph vs CNN" from "the gap".
  4. Same ablation with weights equalised to the two-logit mean -> equals the equal average's ranking (sanity).
Outputs: s06_ranking_vs_calibration.csv, s06_ranking_vs_calibration.txt
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "<this folder>")
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn import metrics as skm

from common import CLONE, CONTEXTS, OUT, load, my_logit


def refinement(y, p):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
    return float(np.mean((iso.predict(p) - y) ** 2))


def best_f1(y, p):
    prec, rec, _ = skm.precision_recall_curve(y, p)
    f = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-300), 0)
    return float(f.max())


rows = []
for ctx in CONTEXTS:
    t = load(ctx)
    y = t["label"].to_numpy()
    st = json.load(open(f"{CLONE}/models/final_v2/gap_stage/{ctx}/frozen_gap_stage.json"))
    mu, sc, w = (np.asarray(st[k]) for k in ("scaler_mean", "scaler_scale", "weights"))
    zg = (my_logit(t["graph_probability"]) - mu[0]) / sc[0]
    zc = (my_logit(t["cnn_alone_probability"]) - mu[1]) / sc[1]
    zgap = (t["gap"].to_numpy() - mu[2]) / sc[2]
    nogap = w[0] * zg + w[1] * zc
    full = nogap + w[2] * zgap
    eq_rank = my_logit(t["graph_probability"]) + my_logit(t["cnn_alone_probability"])
    r = {"context": ctx}
    for name, p in (("flagship", t["probability"].to_numpy()), ("equal_average", t["equal_average_probability"].to_numpy())):
        b = float(np.mean((p - y) ** 2))
        ref = refinement(y, p)
        r[f"{name}_brier"] = b
        r[f"{name}_refinement"] = ref
        r[f"{name}_calibration_loss"] = b - ref
        r[f"{name}_auroc"] = skm.roc_auc_score(y, p)
        r[f"{name}_auprc"] = skm.average_precision_score(y, p)
        thr_col = "threshold" if name == "flagship" else "equal_average_threshold"
        r[f"{name}_f1_frozen"] = skm.f1_score(y, (p >= t[thr_col].to_numpy()).astype(int))
        r[f"{name}_f1_best_any_threshold"] = best_f1(y, p)
    r["d_brier"] = r["flagship_brier"] - r["equal_average_brier"]
    r["d_refinement"] = r["flagship_refinement"] - r["equal_average_refinement"]
    r["d_calibration_loss"] = r["flagship_calibration_loss"] - r["equal_average_calibration_loss"]
    r["d_f1_frozen"] = r["flagship_f1_frozen"] - r["equal_average_f1_frozen"]
    r["d_f1_best"] = r["flagship_f1_best_any_threshold"] - r["equal_average_f1_best_any_threshold"]
    r["auroc_full_linear_score"] = skm.roc_auc_score(y, full)
    r["auroc_stage_without_gap"] = skm.roc_auc_score(y, nogap)
    r["auroc_equal_rank_check"] = skm.roc_auc_score(y, eq_rank)
    r["auprc_stage_without_gap"] = skm.average_precision_score(y, nogap)
    r["d_auroc_total"] = r["flagship_auroc"] - r["equal_average_auroc"]
    r["d_auroc_from_reweighting"] = r["auroc_stage_without_gap"] - r["equal_average_auroc"]
    r["d_auroc_from_gap_term"] = r["flagship_auroc"] - r["auroc_stage_without_gap"]
    r["d_auprc_total"] = r["flagship_auprc"] - r["equal_average_auprc"]
    r["d_auprc_from_gap_term"] = r["flagship_auprc"] - r["auprc_stage_without_gap"]
    r["graph_to_cnn_weight_ratio_on_logits"] = (w[0] / sc[0]) / (w[1] / sc[1])
    rows.append(r)

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/s06_ranking_vs_calibration.csv", index=False)
pd.set_option("display.width", 250)
cols1 = ["context", "d_brier", "d_refinement", "d_calibration_loss", "d_f1_frozen", "d_f1_best"]
cols2 = ["context", "d_auroc_total", "d_auroc_from_reweighting", "d_auroc_from_gap_term", "d_auprc_total",
         "d_auprc_from_gap_term", "graph_to_cnn_weight_ratio_on_logits"]
log = ["== flagship - equal average: Brier split into refinement (ranking) and calibration; F1 frozen vs best",
       D[cols1].round(5).to_string(index=False),
       f"mean: d_brier {D.d_brier.mean():.5f} = d_refinement {D.d_refinement.mean():.5f} + d_calibration {D.d_calibration_loss.mean():.5f}",
       "", "== AUROC gain over equal average: graph/CNN re-weighting vs the gap term (frozen weights, no refit)",
       D[cols2].round(5).to_string(index=False),
       f"check: max|AUROC(full linear score) - AUROC(flagship)| = {np.abs(D.auroc_full_linear_score - D.flagship_auroc).max():.2e}; "
       f"max|AUROC(g+c logit sum) - AUROC(equal avg)| = {np.abs(D.auroc_equal_rank_check - D.equal_average_auroc).max():.2e}",
       f"mean d_auroc total {D.d_auroc_total.mean():.5f}: re-weighting {D.d_auroc_from_reweighting.mean():.5f}, gap term {D.d_auroc_from_gap_term.mean():.5f}"]
open(f"{OUT}/s06_ranking_vs_calibration.txt", "w").write("\n".join(log) + "\n")
print("\n".join(log))
