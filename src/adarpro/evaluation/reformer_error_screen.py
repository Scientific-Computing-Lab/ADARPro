"""Reformer RBP error screen on a predictor's development out-of-fold predictions (PNAS integration R8, D57).

Question: does sequence-predicted RBP binding (Reformer, 225 eCLIP-trained models) separate a predictor's confident
errors from correctly classified sites at the same editing level?

Design (fixed before any look, ``pnas_integration/r8_reformer/PLAN.md``):

- Rows: the predictor's development out-of-fold predictions for the four individual tissues, pooled (Combined is
  excluded: it re-uses the same sites).
- Features (39): for the 16 dsRNA-binder models, ``max25`` (strongest predicted binding within +/-25 nt of the edited
  A) and ``focal`` (binding at the A); ``total_binding`` (mean ``max25`` over all 225 models); PC1-PC5 of all 225
  models' ``max25``; and a placebo (PC1 permuted across rows).
- Panels: confident misses (FN, probability < 0.10) against confident correct edited calls (TP, > 0.90); confident
  false alarms (FP, > 0.90) against confident correct unedited calls (TN, < 0.10). Each feature is residualized on a
  cubic in the editing level plus the target's arm (first or second), and in a second variant also on composition
  (whole-duplex GC, local GC, Alu divergence, paired fraction). The effect is Cohen's d of the residuals, error minus
  correct, with a 95% interval from resampling whole duplexes (B = 1000) and BH-FDR across the features.
- Also: edited against unedited (does not involve the predictor), and a sweep of the false-alarm confidence cut.

The statistics follow the earlier error screen this design comes from line for line; the random streams (one for
the placebo, one for every bootstrap, both seeded) are consumed in the same order, so the same inputs give the same
numbers.

Inputs are outcome-independent site features on canonical train+valid rows only; no test file is read.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.quarantine import sha256_file

TISSUES = ["Artery", "Brain", "Liver", "MuscleSkeletal"]
CONF_LO, CONF_HI = 0.10, 0.90
SWEEP_CUTS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BOOTSTRAPS = 1000
SEED = 0
SPACER = "N" * 10
COMPOSITION = [
    "intrinsic__whole_gc",
    "intrinsic_site__local_gc_51",
    "repeat__divergence_mean",
    "intrinsic__paired_fraction",
]
PANEL = ["ILF3", "STAU2", "DGCR8", "DROSHA", "SND1", "TROVE2", "DHX30", "DDX21", "DDX24"]
SMALL_EFFECT = 0.2


# ---------------------------------------------------------------- statistics


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y)
    r = np.full_like(y, np.nan, dtype=float)
    if ok.sum() < X.shape[1] + 2:
        return r
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    r[ok] = y[ok] - X[ok] @ beta
    return r


def cohens_d(resid: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = resid[a], resid[b]
    ra, rb = ra[np.isfinite(ra)], rb[np.isfinite(rb)]
    if len(ra) < 3 or len(rb) < 3:
        return np.nan
    na, nb = len(ra), len(rb)
    sp = np.sqrt(((na - 1) * ra.var(ddof=1) + (nb - 1) * rb.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0
    return float((ra.mean() - rb.mean()) / sp)


def cluster_boot(resid, is_err, pid, rng: np.random.Generator, B: int = BOOTSTRAPS):
    """Pair-clustered bootstrap 95% interval and two-sided p for Cohen's d(error, correct) on fixed residuals."""
    uniq = np.unique(pid)
    groups = {p: np.where(pid == p)[0] for p in uniq}
    base = cohens_d(resid, is_err, ~is_err)
    vals = []
    for _ in range(B):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[p] for p in samp])
        d = cohens_d(resid[idx], is_err[idx], ~is_err[idx])
        if np.isfinite(d):
            vals.append(d)
    if len(vals) < 20:
        return base, np.nan, np.nan, np.nan
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return base, float(lo), float(hi), float(max(p, 1 / B))


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = m - rank
        val = min(prev, p[i] * m / k)
        q[i] = prev = val
    return q


def design(editing, arm_first, comp, use_editing: bool, use_comp: bool) -> np.ndarray:
    cols = [np.ones(len(editing))]
    if use_editing:
        e = (editing - np.nanmean(editing)) / (np.nanstd(editing) + 1e-9)
        cols += [e, e**2, e**3]
    cols.append(np.asarray(arm_first, dtype=float))
    if use_comp and comp is not None:
        for j in range(comp.shape[1]):
            c = comp[:, j].astype(float)
            mu = np.nanmean(c)
            c = np.where(np.isfinite(c), c, mu)
            c = (c - mu) / (np.nanstd(c) + 1e-9)
            cols.append(c)
    return np.column_stack(cols)


def run_panel(fmat, vis, err_mask, cor_mask, editing, arm_first, comp, pid, rng, use_editing, use_comp):
    keep = err_mask | cor_mask
    is_err = err_mask[keep]
    X = design(editing[keep], arm_first[keep], comp[keep] if comp is not None else None, use_editing, use_comp)
    pidk = pid[keep]
    rows = []
    for col in fmat.columns:
        y = fmat[col].to_numpy(float)[keep]
        resid = residualize(y, X)
        d, lo, hi, p = cluster_boot(resid, is_err, pidk, rng)
        rows.append(dict(feature=col, visibility=vis[col], d=d, ci_lo=lo, ci_hi=hi, boot_p=p,
                         n_err=int(is_err.sum()), n_cor=int((~is_err).sum())))
    df = pd.DataFrame(rows)
    df["fdr_q"] = bh_fdr(df["boot_p"].to_numpy())
    return df.sort_values("d", key=lambda s: s.abs(), ascending=False)


def err4(label: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    label = np.asarray(label).astype(int)
    prediction = np.asarray(prediction).astype(int)
    return np.select(
        [(label == 1) & (prediction == 1), (label == 0) & (prediction == 0),
         (label == 0) & (prediction == 1), (label == 1) & (prediction == 0)],
        ["TP", "TN", "FP", "FN"], default="??",
    )


def feature_columns(features: pd.DataFrame) -> list[str]:
    panel_cols = [c for c in features.columns if c.split("_")[0] in PANEL]
    return panel_cols + ["total_binding"] + [f"PC{k}" for k in range(1, 6)]


def confident_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    prob = frame["probability"].to_numpy(float)
    e4 = frame["err4"].to_numpy(str)
    return {
        "FN": (e4 == "FN") & (prob < CONF_LO),
        "TP": (e4 == "TP") & (prob > CONF_HI),
        "FP": (e4 == "FP") & (prob > CONF_HI),
        "TN": (e4 == "TN") & (prob < CONF_LO),
    }


def screen(frame: pd.DataFrame, feat_cols: list[str], seed: int = SEED) -> dict[str, pd.DataFrame]:
    """The screen. ``frame`` needs pair_id, label, editing_index, arm_first, err4, probability, COMPOSITION, feat_cols.

    Random streams: the placebo permutation draws from its own generator (seed), and every bootstrap draws from one
    shared generator (seed), in the order miss panel (editing only, then + composition), false-alarm panel (same),
    edited panel, sweep.
    """
    placebo_rng = np.random.default_rng(seed)
    boot_rng = np.random.default_rng(seed)
    fmat = frame[feat_cols].copy()
    fmat["placebo_permPC1"] = placebo_rng.permutation(frame["PC1"].to_numpy())
    vis = {c: "invisible" for c in feat_cols}
    vis["placebo_permPC1"] = "placebo"

    editing = frame["editing_index"].to_numpy(float)
    arm_first = frame["arm_first"].to_numpy(bool)
    pid = frame["pair_id"].to_numpy()
    prob = frame["probability"].to_numpy(float)
    e4 = frame["err4"].to_numpy(str)
    comp = frame[COMPOSITION].to_numpy(float)
    conf = confident_masks(frame)

    out: dict[str, pd.DataFrame] = {}
    for panel, (em, cm) in [("fn", (conf["FN"], conf["TP"])), ("fp", (conf["FP"], conf["TN"]))]:
        a = run_panel(fmat, vis, em, cm, editing, arm_first, comp, pid, boot_rng, use_editing=True, use_comp=False)
        b = run_panel(fmat, vis, em, cm, editing, arm_first, comp, pid, boot_rng, use_editing=True, use_comp=True)
        merged = a[["feature", "visibility", "d", "ci_lo", "ci_hi", "fdr_q"]].rename(
            columns={"d": "d_editOnly", "ci_lo": "lo_editOnly", "ci_hi": "hi_editOnly", "fdr_q": "q_editOnly"}).merge(
            b[["feature", "d", "ci_lo", "ci_hi", "fdr_q"]].rename(
                columns={"d": "d_compCtrl", "ci_lo": "lo_compCtrl", "ci_hi": "hi_compCtrl", "fdr_q": "q_compCtrl"}),
            on="feature")
        out[panel] = merged.sort_values("d_compCtrl", key=lambda s: s.abs(), ascending=False)

    label = frame["label"].to_numpy()
    out["edited"] = run_panel(fmat, vis, label == 1, label == 0, editing, arm_first, comp, pid, boot_rng,
                              use_editing=False, use_comp=True)

    sweep = []
    tnm = conf["TN"]
    for thr in SWEEP_CUTS:
        fpm = (e4 == "FP") & (prob > thr)
        keep = fpm | tnm
        if fpm.sum() < 20:
            continue
        X = design(editing[keep], arm_first[keep], comp[keep], True, True)
        ie = fpm[keep]
        pk = pid[keep]
        row = {"fp_prob_cut": thr, "n_fp": int(fpm.sum())}
        for col in ["total_binding", "PC1"]:
            r = residualize(fmat[col].to_numpy(float)[keep], X)
            dd, lo, hi, _ = cluster_boot(r, ie, pk, boot_rng)
            row[f"d_{col}"] = dd
            row[f"lo_{col}"] = lo
            row[f"hi_{col}"] = hi
        sweep.append(row)
    out["sweep"] = pd.DataFrame(sweep)
    return out


def read_panel(df: pd.DataFrame) -> dict:
    """The fixed reading rule (PLAN.md Part A section 4) applied to one panel's composition-controlled rows."""
    real = df[df["visibility"] != "placebo"]
    sig = real[(real["q_compCtrl"] < 0.05) & (np.sign(real["lo_compCtrl"]) == np.sign(real["hi_compCtrl"]))]
    agg = real[real["feature"].isin(["total_binding", "PC1"])]
    max_abs = float(real["d_compCtrl"].abs().max())
    signal = bool(((sig["d_compCtrl"].abs() >= SMALL_EFFECT)).any())
    agg_inside = bool(((agg["lo_compCtrl"] > -SMALL_EFFECT) & (agg["hi_compCtrl"] < SMALL_EFFECT)).all())
    none_reach = bool((real["d_compCtrl"].abs() < SMALL_EFFECT).all())
    reading = "signal" if signal else ("no usable signal" if (none_reach and agg_inside) else "not shown either way")
    reach = int(((real["lo_compCtrl"] <= -SMALL_EFFECT) | (real["hi_compCtrl"] >= SMALL_EFFECT)).sum())
    return {
        "features": int(len(real)),
        "significant": int(len(sig)),
        "largest_abs_d": max_abs,
        "largest_feature": str(real.loc[real["d_compCtrl"].abs().idxmax(), "feature"]),
        "intervals_reaching_small_effect": reach,
        "aggregate_intervals_inside_small_effect": agg_inside,
        "placebo_d": float(df.loc[df["visibility"] == "placebo", "d_compCtrl"].iloc[0]),
        "reading": reading,
    }


# ---------------------------------------------------------------- inputs


def arm_first_from_sequence(sequence: pd.Series, local_position: pd.Series) -> np.ndarray:
    """True when the target lies in the duplex's first arm (before the 10-N spacer)."""
    if not (sequence.str.count(SPACER) == 1).all():
        raise RuntimeError("expected exactly one 10-N spacer in every sequence")
    spacer = sequence.str.find(SPACER).to_numpy()
    return local_position.to_numpy() < spacer


def import_features(source: Path, output: Path, provenance: dict) -> dict:
    """Copy the outcome-independent Reformer site-feature table into the repository, with a provenance summary."""
    table = pd.read_parquet(source)
    if "crop_uid" not in table or not table["crop_uid"].is_unique:
        raise RuntimeError("source table must have one row per site key (crop_uid)")
    feats = feature_columns(table)
    missing = [c for c in feats if c not in table]
    if missing or len(feats) != 38:
        raise RuntimeError(f"unexpected feature columns (missing {missing}, n={len(feats)})")
    if not np.isfinite(table[feats].to_numpy(float)).all():
        raise RuntimeError("non-finite feature values")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    summary = {
        "canonical_test_access": False,
        "scope": "outcome-independent Reformer site features on canonical train+valid sites of the four tissues",
        "rows": int(len(table)),
        "key": "crop_uid = pair_id:local_position (equals pair_local_key in site_master)",
        "feature_columns": feats,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(output),
        **provenance,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def import_loadings(source: Path, output: Path, features_summary: Path) -> dict:
    """Copy the PCA loadings behind the PC1-PC5 features (models x PC1..PC5), with checks and a provenance summary.

    The source must be the loadings file the feature table's summary records (same sha256), with orthonormal columns.
    """
    load = pd.read_csv(source, index_col=0)
    if list(load.columns) != [f"PC{k}" for k in range(1, 6)] or not load.index.is_unique:
        raise RuntimeError("expected one row per binding model and columns PC1..PC5")
    gram = load.to_numpy(float).T @ load.to_numpy(float)
    orth_err = float(np.abs(gram - np.eye(5)).max())
    if orth_err > 1e-6:
        raise RuntimeError(f"loadings are not orthonormal (max error {orth_err})")
    recorded = json.loads(features_summary.read_text())["files"]["pca_loadings"]["sha256"]
    source_sha = sha256_file(source)
    if source_sha != recorded:
        raise RuntimeError("loadings file differs from the one recorded in the feature summary")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    summary = {
        "canonical_test_access": False,
        "scope": "PCA loadings of the 225 Reformer models' z-scored max25 over the feature table's 39,190 sites",
        "models": int(len(load)),
        "columns": list(load.columns),
        "orientation": "each PC oriented so its loadings sum positive",
        "orthonormal_max_error": orth_err,
        "source": str(source),
        "source_sha256": source_sha,
        "output_sha256": sha256_file(output),
        "matches_feature_summary_record": True,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def describe_pcs(loadings: pd.DataFrame, features: pd.DataFrame, pcs: tuple[str, ...] = ("PC2", "PC3"),
                 top: int = 6) -> dict:
    """What the summary axes weigh: variance explained, and for ``pcs`` the share of squared loading by cell-type
    label and on the dsRNA-binder panel models, against an even spread, with the heaviest models on each side.

    Each model's max25 was z-scored before the PCA, so the total variance equals the number of models whose loadings
    are not all zero (a constant model has zero loadings).
    """
    load = loadings.astype(float)
    n_models = int((load.abs().sum(axis=1) > 0).sum())
    var_explained = {k: float(features[k].var(ddof=0) / n_models) for k in load.columns}
    cell = pd.Series([m.split("_", 1)[1] for m in load.index], index=load.index)
    panel = pd.Series([m.split("_")[0] in PANEL for m in load.index], index=load.index)
    out = {
        "models": int(len(load)),
        "models_by_cell_label": {c: int((cell == c).sum()) for c in sorted(cell.unique())},
        "models_hepg2_or_k562": int(cell.isin(["HepG2", "K562"]).sum()),
        "dsRNA_binder_models": int(panel.sum()),
        "variance_explained": var_explained,
    }
    for pc in pcs:
        w = load[pc]
        sq = w ** 2
        out[pc] = {
            "weight_share_dsRNA_binder_models": float(sq[panel].sum()),
            "even_share_dsRNA_binder_models": float(panel.mean()),
            "weight_share_by_cell_label": {c: float(sq[cell == c].sum()) for c in sorted(cell.unique())},
            "even_share_by_cell_label": {c: float((cell == c).mean()) for c in sorted(cell.unique())},
            "heaviest_positive": [[m, float(w[m])] for m in w.sort_values(ascending=False).index[:top]],
            "heaviest_negative": [[m, float(w[m])] for m in w.sort_values().index[:top]],
        }
    return out


def build_frame(predictions_root: Path, model: str, features_path: Path, site_master_path: Path,
                genomic_context_path: Path, endpoints: list[str] = TISSUES) -> tuple[pd.DataFrame, dict]:
    """Join a predictor's development OOF rows to the Reformer features and covariates, with checks."""
    preds = []
    for ep in endpoints:
        p = pd.read_parquet(predictions_root / ep / "oof_predictions.parquet")
        p = p[p["model"] == model]
        if p.empty:
            raise RuntimeError(f"no rows for model {model} in {ep}")
        if set(p["context"]) != {ep}:
            raise RuntimeError(f"{ep}: rows from another context")
        preds.append(p)
    pred = pd.concat(preds, ignore_index=True)
    if not pred["example_id"].is_unique:
        raise RuntimeError("duplicate example_id in predictions")
    implied = (pred["probability"].to_numpy(float) >= pred["threshold"].to_numpy(float)).astype(int)
    pred_mismatch = int((implied != pred["prediction"].to_numpy(int)).sum())
    if pred_mismatch:
        raise RuntimeError(f"prediction != (probability >= threshold) on {pred_mismatch} rows")

    sm = pd.read_parquet(site_master_path, columns=["example_id", "pair_local_key", "pair_id", "label",
                                                     "editing_index", "sequence", "local_position", "source_split"])
    sm = sm.set_index("example_id").loc[pred["example_id"]].reset_index()
    if not (sm["source_split"].isin(["train", "valid"])).all():
        raise RuntimeError("a prediction row is not a canonical train/valid row")
    for c in ["pair_local_key", "pair_id", "label"]:
        if not (sm[c].to_numpy() == pred[c].to_numpy()).all():
            raise RuntimeError(f"{c} disagrees between predictions and site_master")

    feats = pd.read_parquet(features_path)
    feat_cols = feature_columns(feats)
    feats = feats.rename(columns={"crop_uid": "pair_local_key"})
    gc = pd.read_parquet(genomic_context_path, columns=["example_id"] + COMPOSITION)

    frame = pred[["example_id", "context", "pair_id", "pair_local_key", "label", "probability", "threshold",
                  "prediction"]].copy()
    frame["editing_index"] = sm["editing_index"].to_numpy(float)
    frame["arm_first"] = arm_first_from_sequence(sm["sequence"], sm["local_position"])
    frame["err4"] = err4(frame["label"], frame["prediction"])
    n_before = len(frame)
    frame = frame.merge(feats[["pair_local_key"] + feat_cols], on="pair_local_key", how="left", validate="many_to_one")
    frame = frame.merge(gc, on="example_id", how="left", validate="one_to_one")
    if len(frame) != n_before:
        raise RuntimeError("join changed the row count")
    uncovered = int(frame[feat_cols].isna().any(axis=1).sum())
    if uncovered:
        raise RuntimeError(f"{uncovered} rows have no Reformer features")
    comp_na = int(frame[COMPOSITION].isna().any(axis=1).sum())

    conf = confident_masks(frame)
    qc = {
        "model": model,
        "endpoints": endpoints,
        "rows": int(len(frame)),
        "rows_per_endpoint": frame["context"].value_counts().reindex(endpoints).astype(int).to_dict(),
        "pairs": int(frame["pair_id"].nunique()),
        "unique_sites": int(frame["pair_local_key"].nunique()),
        "prediction_threshold_mismatches": pred_mismatch,
        "rows_without_reformer_features": uncovered,
        "rows_with_missing_composition": comp_na,
        "target_in_first_arm": int(frame["arm_first"].sum()),
        "err4_counts": pd.Series(frame["err4"]).value_counts().reindex(["TP", "TN", "FP", "FN"]).astype(int).to_dict(),
        "confident_counts": {k: int(v.sum()) for k, v in conf.items()},
        "confident_pairs": {k: int(frame.loc[v, "pair_id"].nunique()) for k, v in conf.items()},
        "feature_columns": feat_cols,
        "inputs": {
            "predictions_root": str(predictions_root),
            "features": str(features_path),
            "features_sha256": sha256_file(features_path),
            "site_master": str(site_master_path),
            "genomic_context": str(genomic_context_path),
        },
    }
    return frame, qc


def write_screen(out: dict[str, pd.DataFrame], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    out["fn"].to_csv(output_dir / "fn_effects.csv", index=False)
    out["fp"].to_csv(output_dir / "fp_effects.csv", index=False)
    out["edited"].to_csv(output_dir / "edited_effects.csv", index=False)
    out["sweep"].to_csv(output_dir / "threshold_sweep.csv", index=False)
    reading = {"fn": read_panel(out["fn"]), "fp": read_panel(out["fp"]),
               "sweep_max_abs_d": float(out["sweep"][["d_total_binding", "d_PC1"]].abs().max().max())}
    (output_dir / "reading.json").write_text(json.dumps(reading, indent=2) + "\n")
    return reading


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import-features", help="copy the Reformer site-feature table into the repository")
    imp.add_argument("--source", type=Path, required=True)
    imp.add_argument("--output", type=Path, default=Path("data/derived/features/reformer_context.parquet"))
    imp.add_argument("--provenance-json", type=Path, required=True, help="producer scripts, inputs and run record")

    il = sub.add_parser("import-loadings", help="copy the PCA loadings behind the PC features into the repository")
    il.add_argument("--source", type=Path, required=True)
    il.add_argument("--output", type=Path, default=Path("data/derived/features/reformer_context_pca_loadings.csv"))
    il.add_argument("--features-summary", type=Path,
                    default=Path("data/derived/features/reformer_context.summary.json"))

    dp = sub.add_parser("describe-pcs", help="what the PC features weigh (variance explained, weight shares)")
    dp.add_argument("--loadings", type=Path, default=Path("data/derived/features/reformer_context_pca_loadings.csv"))
    dp.add_argument("--features", type=Path, default=Path("data/derived/features/reformer_context.parquet"))
    dp.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run", help="build the frame (and, unless --build-only, run the screen)")
    run.add_argument("--predictions-root", type=Path, required=True, help="folder with <endpoint>/oof_predictions.parquet")
    run.add_argument("--model", required=True, help="value of the 'model' column to screen, e.g. gap")
    run.add_argument("--features", type=Path, default=Path("data/derived/features/reformer_context.parquet"))
    run.add_argument("--site-master", type=Path, default=Path("data/derived/site_master.parquet"))
    run.add_argument("--genomic-context", type=Path, default=Path("data/derived/features/genomic_context.parquet"))
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--seed", type=int, default=SEED)
    run.add_argument("--build-only", action="store_true", help="write the joined-frame checks and counts only")
    args = parser.parse_args()

    if args.command == "import-features":
        summary = import_features(args.source, args.output, json.loads(args.provenance_json.read_text()))
        print(json.dumps({k: summary[k] for k in ["rows", "source_sha256", "output_sha256"]}, indent=2))
        return
    if args.command == "import-loadings":
        summary = import_loadings(args.source, args.output, args.features_summary)
        print(json.dumps({k: summary[k] for k in ["models", "orthonormal_max_error", "output_sha256"]}, indent=2))
        return
    if args.command == "describe-pcs":
        loadings = pd.read_csv(args.loadings, index_col=0)
        features = pd.read_parquet(args.features, columns=list(loadings.columns))
        out = describe_pcs(loadings, features)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return

    frame, qc = build_frame(args.predictions_root, args.model, args.features, args.site_master, args.genomic_context)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "build_qc.json").write_text(json.dumps(qc, indent=2) + "\n")
    print(json.dumps({k: qc[k] for k in ["rows", "rows_per_endpoint", "err4_counts", "confident_counts",
                                         "confident_pairs"]}, indent=2))
    if args.build_only:
        return
    out = screen(frame, qc["feature_columns"], seed=args.seed)
    reading = write_screen(out, args.output_dir)
    print(json.dumps(reading, indent=2))


if __name__ == "__main__":
    main()
