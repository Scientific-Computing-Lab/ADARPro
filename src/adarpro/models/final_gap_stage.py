"""Frozen final gap stage on top of the final graph and CNN-alone models (PNAS integration step 11, D44-D46).

PNAS's gate scheme mapped onto the final splits: each final network (every seed) scores every canonical
development row; each network is averaged over its seeds in log-odds (D45); one logistic stage on the two averaged
log-odds and the pair's log1p genomic gap is fitted on the published TRAINING split's canonical rows (in-sample
scores, as the gate scheme fits on training folds), and its penalty and F1 threshold are chosen on the published
VALIDATION split (the inner fold's role). The fitted stage is frozen as plain numbers (scaler means and scales,
weights, intercept, threshold) in ``frozen_gap_stage.json``; ``apply_frozen_stage`` applies it without refitting.
Development rows only; no test row is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adarpro.models.baseline_oof import (
    build_graphs,
    choose_threshold,
    torch_load,
    upstream_modules,
)
from adarpro.models.fixed_checkpoint_ensemble import equal_logit_average_many
from adarpro.models.gap_stage import (
    ARMS,
    CNN_MODE,
    GRAPH_MODE,
    average_probabilities,
    logit,
    pair_gap_table,
    score_all_rows,
)
from adarpro.models.multitask_oof import (
    canonical_window_examples,
    check_checkpoint_mode,
    run_tag_suffix,
)
from adarpro.models.residual import CS, best_f1_threshold, metrics
from adarpro.quarantine import sha256_file
from adarpro.registry import register

COLUMNS = ARMS["gap"]  # graph_logit, cnn_logit, gap
SELECTION = "published validation canonical rows"


def fit_final_stage(table: pd.DataFrame, seed: int) -> tuple[object, dict]:
    """``fit_gate``'s loop with the split mapped: fit on source_split == train, choose C and threshold on valid."""
    train = table[table["source_split"] == "train"]
    valid = table[table["source_split"] == "valid"]
    if train.empty or valid.empty or set(table["source_split"]) - {"train", "valid"}:
        raise RuntimeError("final stage needs exactly the published train and validation canonical rows")
    if set(train["pair_id"]) & set(valid["pair_id"]):
        raise RuntimeError("pair leakage between the final stage's training and validation rows")
    best = None
    for c_value in CS:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, penalty="l2", solver="liblinear",
                                                                   max_iter=5000, random_state=seed))
        model.fit(train[COLUMNS], train["label"])
        probability = model.predict_proba(valid[COLUMNS])[:, 1]
        threshold, f1 = best_f1_threshold(valid["label"].to_numpy(), probability)
        candidate = (f1, -c_value, model, threshold)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    f1, negative_c, model, threshold = best
    return model, {"C": -negative_c, "threshold": float(threshold), "validation_f1": float(f1),
                   "train_rows": len(train), "validation_rows": len(valid)}


def freeze(model, selection: dict) -> dict:
    scaler, logistic = model[0], model[-1]
    return {"columns": list(COLUMNS), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "weights": logistic.coef_[0].tolist(), "intercept": float(logistic.intercept_[0]), **selection}


def apply_frozen_stage(frozen: dict, graph_probability, cnn_probability, gap) -> np.ndarray:
    """The frozen stage's probability: logistic(intercept + w . (x - mean) / scale), x = (graph log-odds, CNN
    log-odds, log1p gap); both probabilities are clipped exactly as the stage was fitted (gap_stage.logit)."""
    if frozen["columns"] != list(COLUMNS):
        raise ValueError(f"unexpected frozen columns {frozen['columns']}")
    x = np.column_stack([logit(graph_probability), logit(cnn_probability), np.asarray(gap, dtype=np.float64)])
    z = (x - np.asarray(frozen["scaler_mean"])) / np.asarray(frozen["scaler_scale"])
    return 1 / (1 + np.exp(-(z @ np.asarray(frozen["weights"]) + frozen["intercept"])))


def load_final(path: Path, mode: str, context: str, device: torch.device) -> dict:
    saved = torch_load(path / "best.pth", device)
    check_checkpoint_mode(saved, mode, path / "best.pth")
    config = saved.get("config") or {}
    if config.get("context") != context or config.get("selection_split") != SELECTION:
        raise RuntimeError(f"{path}: not a final model of {context} selected on {SELECTION}")
    return saved


def run(
    repository: Path,
    site_master_path: Path,
    context: str,
    graph_dirs: list[Path],
    cnn_dirs: list[Path],
    output_dir: Path,
    batch_size: int,
    seed: int,
    device: torch.device,
    run_tag: str | None = None,
) -> dict:
    run_tag_suffix(run_tag)  # refuse a bad tag before any work
    for published in ("models/frozen", "models/frozen_ensemble"):
        if output_dir.resolve().is_relative_to((repository / published).resolve()):
            raise RuntimeError(f"refusing to write into the published {published}; pass --output-dir")
    if not graph_dirs or len(graph_dirs) != len(cnn_dirs):
        raise ValueError("give the same, non-zero number of graph and CNN-alone final models")
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("final gap stage received a sealed split")
    canonical = master[master["context"] == context].copy().reset_index(drop=True)
    gaps = pair_gap_table(canonical)
    lanes = {torch_load(d / "best.pth", device)["config"].get("partner_lane") for d in cnn_dirs}
    if len(lanes) != 1:
        raise RuntimeError(f"CNN-alone finals disagree on the partner lane: {lanes}")
    graphs = build_graphs(canonical, upstream_modules(repository))
    windows = canonical_window_examples(canonical, partner_lane=lanes.pop())
    labels = canonical["label"].to_numpy()
    valid = (canonical["source_split"] == "valid").to_numpy()
    per_network, hashes, reproduction = {"graph": [], "cnn": []}, {}, 0.0
    for name, mode, dirs, examples in (("graph", GRAPH_MODE, graph_dirs, graphs), ("cnn", CNN_MODE, cnn_dirs, windows)):
        for directory in dirs:
            load_final(directory, mode, context, device)
            hashes[str(directory)] = sha256_file(directory / "best.pth")
            probability, predicted = score_all_rows(directory / "best.pth", mode, examples, batch_size, device)
            if not np.array_equal(predicted, labels):
                raise RuntimeError(f"{directory}: prediction order mismatch")
            saved = pd.read_parquet(directory / "validation_predictions.parquet").set_index("example_id")
            mine = pd.Series(probability[valid], index=canonical.loc[valid, "example_id"])
            reproduction = max(reproduction, float(np.max(np.abs(saved.loc[mine.index, "probability"].to_numpy()
                                                                  - mine.to_numpy()))))
            per_network[name].append(probability)
    if reproduction > 1e-5:
        raise RuntimeError(f"final models do not reproduce their saved validation predictions: {reproduction}")
    graph_p, cnn_p = average_probabilities(per_network["graph"]), average_probabilities(per_network["cnn"])
    table = canonical[["example_id", "context", "pair_id", "pair_local_key", "label", "source_split"]].copy()
    table["graph_logit"], table["cnn_logit"] = logit(graph_p), logit(cnn_p)
    table = table.merge(gaps[["pair_id", "gap"]], on="pair_id", how="left", validate="many_to_one")
    if table["gap"].isna().any() or not table["example_id"].equals(canonical["example_id"]):
        raise RuntimeError("gap lookup failed or row order lost")
    model, selection = fit_final_stage(table, seed)
    frozen = freeze(model, selection)
    stage_p = apply_frozen_stage(frozen, graph_p, cnn_p, table["gap"].to_numpy())
    if not np.allclose(stage_p, model.predict_proba(table[COLUMNS])[:, 1], rtol=0, atol=1e-12):
        raise RuntimeError("frozen stage does not reproduce the fitted model")
    ensemble_p = equal_logit_average_many(per_network["graph"] + per_network["cnn"])
    ensemble_threshold, _ = choose_threshold(labels[valid], ensemble_p[valid])
    destination = output_dir / context
    destination.mkdir(parents=True, exist_ok=True)
    frozen.update({"context": context, "seed": seed, "final_models": hashes,
                   "averaging": "each network averaged over its seeds in log-odds, then one stage (D45)",
                   "gap": "log1p(start2 - end1), adarpro.features.genomic_context.pair_intrinsic_features",
                   "fit_split": "published train canonical rows (in-sample final-model scores; gate scheme)",
                   "selection_split": SELECTION, "canonical_test_access": False})
    text = json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    (destination / "frozen_gap_stage.json").write_text(text)
    rows = canonical.loc[valid, ["example_id", "context", "pair_id", "pair_local_key", "label"]].copy()
    summary_metrics = {}
    for name, probability, threshold in (("gap_stage", stage_p, frozen["threshold"]),
                                         ("equal_average", ensemble_p, ensemble_threshold)):
        rows[f"{name}_probability"] = probability[valid]
        rows[f"{name}_prediction"] = (probability[valid] >= threshold).astype(int)
        summary_metrics[name] = metrics(rows["label"], rows[f"{name}_probability"], rows[f"{name}_prediction"])
    rows.to_parquet(destination / "validation_predictions.parquet", index=False)
    summary = {
        "scope": "frozen final gap stage; validation metrics are descriptive (that split chose its settings)",
        "context": context, "seeds": len(graph_dirs), "frozen_stage_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "selection": selection, "equal_average_threshold": float(ensemble_threshold),
        "validation_metrics": summary_metrics, "maximum_validation_reproduction_difference": reproduction,
        "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    register(
        repository,
        {
            "run_id": f"final_gap_stage_{context}_seeds{len(graph_dirs)}" + run_tag_suffix(run_tag),
            "experiment_type": "frozen_gap_stage_train_validation_only",
            "context": context,
            "input_hashes": {"site_master": sha256_file(site_master_path), "final_models": hashes},
            "fold_definition": "published train split fits; published validation split selects C and threshold",
            "seed": seed,
            "features": "seed-averaged final graph and CNN-alone log-odds plus log1p genomic gap",
            "hyperparameters": {"stage": "L2 logistic, CS grid, best-F1 threshold on validation"},
            "artifact": str(destination / "frozen_gap_stage.json"),
            "metrics": summary,
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--site-master", type=Path, default=Path("data/derived/site_master.parquet"))
    parser.add_argument("--context", required=True)
    parser.add_argument("--graph-dir", type=Path, action="append", required=True, help="a final graph model folder")
    parser.add_argument("--cnn-dir", type=Path, action="append", required=True, help="a final CNN-alone model folder")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--run-tag")
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(json.dumps(run(root, resolve(args.site_master), args.context, [resolve(p) for p in args.graph_dir],
                         [resolve(p) for p in args.cnn_dir], resolve(args.output_dir), args.batch_size, args.seed,
                         torch.device(args.device), args.run_tag), indent=2))


if __name__ == "__main__":
    main()
