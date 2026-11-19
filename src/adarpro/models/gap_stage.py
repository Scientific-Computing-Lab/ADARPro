"""Gap-only second stage on top of the graph and the graph-free CNN (PNAS integration, D39/D40).

A logistic second stage takes, per canonical development row, the graph's and the CNN alone's log-odds and the pair's
genomic gap, log(1 + start2 - end1) from ``pair_intrinsic_features``. It is fitted with PNAS's gate scheme
(``hierarchical_gate.fit_gate``): for outer fold k the outer-k checkpoints score every row; the stage is fitted on the
three training folds, its penalty and F1 threshold are chosen on the inner fold (k+1)%5, and it is applied to fold k.

Arms (same rows, folds and fitting): ``no_gap`` (the two scores), ``gap`` (+ the gap), ``shuffled_gap`` (+ the gap
values permuted across the endpoint's pairs with a fixed seed). ``equal_average`` is the fixed 50:50 log-odds average
of the two networks with the saved ensemble's inner-fold thresholds (the selected design), reproduced and reported
beside the arms. Development rows only.
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
from torch_geometric.loader import DataLoader

from adarpro.features.genomic_context import pair_intrinsic_features
from adarpro.models.baseline_oof import build_graphs, torch_load, upstream_modules
from adarpro.models.fixed_checkpoint_ensemble import (
    equal_logit_average_many,
    model_for_mode,
)
from adarpro.models.hierarchical_gate import fit_gate
from adarpro.models.multitask_oof import (
    canonical_window_examples,
    check_checkpoint_mode,
    predict_binary,
    run_dir_partner_lane,
    run_tag_suffix,
)
from adarpro.models.residual import bootstrap_differences, metrics
from adarpro.quarantine import sha256_file
from adarpro.registry import register

GRAPH_MODE = "binary_pair_balanced"
CNN_MODE = "binary_pair_balanced_cnn"
ARMS = {
    "no_gap": ["graph_logit", "cnn_logit"],
    "gap": ["graph_logit", "cnn_logit", "gap"],
    "shuffled_gap": ["graph_logit", "cnn_logit", "shuffled_gap"],
}
REFERENCE = "equal_average"
DEFAULT_SHUFFLE_SEED = 20261119  # fixed before any stage was fitted (D40 Qc): one permutation per endpoint
IDENTITY = ["example_id", "context", "pair_id", "pair_local_key", "label", "development_fold"]
PAIR_COLUMNS = ["pair_id", "sequence", "structure", "chr1", "start1", "end1", "chr2", "start2", "end2", "strand"]
EPSILON = np.finfo(np.float64).eps  # the clip the fixed ensemble uses


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1 - EPSILON)
    return np.log(clipped / (1 - clipped))


def pair_gap_table(canonical: pd.DataFrame) -> pd.DataFrame:
    """pair_id -> genomic gap (bp) and log(1 + gap), from PNAS's pair features; label-free."""

    pairs = canonical[PAIR_COLUMNS].drop_duplicates()
    if pairs["pair_id"].duplicated().any():
        raise RuntimeError("a pair has more than one sequence, structure or coordinate set")
    if (pairs["chr1"] != pairs["chr2"]).any():
        raise RuntimeError("a pair's arms lie on different chromosomes")
    rows = []
    for pair in pairs.itertuples(index=False):
        features = pair_intrinsic_features(pd.Series(pair._asdict()))
        rows.append({"pair_id": pair.pair_id, "gap_bp": features["geometry__genomic_gap"],
                     "gap": features["geometry__genomic_gap_log1p"]})
    table = pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)
    if (table["gap_bp"] < 0).any():
        raise RuntimeError("negative genomic gap (overlapping or misordered arms)")
    return table


def shuffled_gap_values(gaps: pd.DataFrame, shuffle_seed: int) -> tuple[pd.DataFrame, str, float]:
    """One fixed permutation of the gap values across pairs (sorted by pair_id first, so it is reproducible).

    Returns the table with ``shuffled_gap``, the permutation's hash and the share of pairs that keep their own value.
    """

    table = gaps.sort_values("pair_id").reset_index(drop=True).copy()
    order = np.random.default_rng(shuffle_seed).permutation(len(table))
    table["shuffled_gap"] = table["gap"].to_numpy()[order]
    digest = hashlib.sha256(
        json.dumps([[pair, int(index)] for pair, index in zip(table["pair_id"], order)]).encode()
    ).hexdigest()
    kept = float(np.mean(table["shuffled_gap"].to_numpy() == table["gap"].to_numpy()))
    return table, digest, kept


def stage_table(
    canonical: pd.DataFrame,
    context: str,
    graph_probability: np.ndarray,
    cnn_probability: np.ndarray,
    gaps: pd.DataFrame,
) -> pd.DataFrame:
    """One row per canonical row: identity, the development reference, both networks' log-odds and the gap columns."""

    if len(graph_probability) != len(canonical) or len(cnn_probability) != len(canonical):
        raise RuntimeError("network score rows differ from the canonical rows")
    if set(canonical["context"]) != {context}:
        raise RuntimeError(f"canonical rows are not all from {context}")
    missing = set(canonical["pair_id"]) - set(gaps["pair_id"])
    if missing:
        raise RuntimeError(f"no gap for {len(missing)} pairs")
    table = canonical[IDENTITY + ["baseline_oof_probability", "baseline_oof_prediction"]].copy()
    table["graph_probability"] = np.asarray(graph_probability, dtype=np.float64)
    table["cnn_probability"] = np.asarray(cnn_probability, dtype=np.float64)
    table["graph_logit"] = logit(graph_probability)
    table["cnn_logit"] = logit(cnn_probability)
    table = table.merge(gaps[["pair_id", "gap", "shuffled_gap"]], on="pair_id", how="left", validate="many_to_one")
    if table[["gap", "shuffled_gap"]].isna().any().any():
        raise RuntimeError("gap lookup failed")
    return table


def fit_arm(table: pd.DataFrame, columns: list[str], outer_fold: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """PNAS's gate fit for one arm, plus the chosen model's standardized weights (refitted identically and checked)."""

    outer, selection = fit_gate(table, columns, outer_fold, seed)
    inner_fold = (outer_fold + 1) % 5
    train = table[~table["development_fold"].isin([outer_fold, inner_fold])]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=selection["C"], penalty="l2", solver="liblinear", max_iter=5000,
                           random_state=seed + outer_fold),
    )
    model.fit(train[columns], train["label"])
    refit = model.predict_proba(outer[columns])[:, 1]
    if not np.allclose(refit, outer["probability"].to_numpy(), rtol=0, atol=1e-12):
        raise RuntimeError("refitted stage does not reproduce the gate fit")
    weights = model[-1].coef_[0]
    selection.update({f"weight_{column}": float(weight) for column, weight in zip(columns, weights)})
    selection["intercept"] = float(model[-1].intercept_[0])
    selection["train_rows"] = len(train)
    return outer, selection


def score_all_rows(checkpoint: Path, mode: str, examples: list, batch_size: int, device: torch.device) -> tuple:
    saved = torch_load(checkpoint, device)
    check_checkpoint_mode(saved, mode, checkpoint)
    model = model_for_mode(mode, device)
    model.load_state_dict(saved["model_state"])
    return predict_binary(model, DataLoader(examples, batch_size=batch_size, shuffle=False), device)


def average_probabilities(arrays: list[np.ndarray]) -> np.ndarray:
    """Average several runs' probabilities in log-odds (equal weights); one run is returned unchanged."""

    if len(arrays) == 1:
        return np.asarray(arrays[0], dtype=np.float64)
    return equal_logit_average_many(arrays)


def score_run(
    repository: Path,
    canonical: pd.DataFrame,
    graph_run_dir: Path,
    cnn_run_dir: Path,
    batch_size: int,
    device: torch.device,
    cache_path: Path | None = None,
    examples: tuple | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Every outer-k checkpoint of one run scores every canonical row (the gate scheme's inputs).

    Returns a long table (example_id, outer_fold, graph_probability, cnn_probability), canonical order within each
    outer fold, and the reproduction maxima against the saved out-of-fold scores. With ``cache_path`` the table is
    read if present (its row identity checked) and written otherwise; scores are kept, never deleted.
    """

    ids = canonical["example_id"].to_numpy()
    if cache_path is not None and cache_path.exists():
        table = pd.read_parquet(cache_path)
        for fold in range(5):
            part = table[table["outer_fold"] == fold]
            if not np.array_equal(part["example_id"].to_numpy(), ids):
                raise RuntimeError(f"score cache {cache_path} does not match the canonical rows (fold {fold})")
        meta = json.loads(cache_path.with_suffix(".json").read_text())
        return table, meta["maximum_reproduction_difference"]
    if examples is None:
        graphs = build_graphs(canonical, upstream_modules(repository))
        windows = canonical_window_examples(canonical, partner_lane=run_dir_partner_lane(cnn_run_dir, device))
    else:
        graphs, windows = examples
    labels = canonical["label"].to_numpy()
    parts, reproduction, hashes = [], {"graph": 0.0, "cnn": 0.0}, {}
    for outer_fold in range(5):
        part = pd.DataFrame({"example_id": ids, "outer_fold": outer_fold})
        for name, mode, run_dir, examples_ in (("graph", GRAPH_MODE, graph_run_dir, graphs),
                                               ("cnn", CNN_MODE, cnn_run_dir, windows)):
            checkpoint = run_dir / f"outer_{outer_fold}" / "best.pth"
            hashes[f"{name}_outer_{outer_fold}"] = sha256_file(checkpoint)
            probability, predicted_labels = score_all_rows(checkpoint, mode, examples_, batch_size, device)
            if not np.array_equal(predicted_labels, labels):
                raise RuntimeError(f"{name} prediction order mismatch")
            saved_outer = pd.read_parquet(run_dir / f"outer_{outer_fold}" / "outer_predictions.parquet")
            outer_rows = canonical.index[canonical["development_fold"] == outer_fold].to_numpy()
            if not np.array_equal(saved_outer["example_id"].to_numpy(), ids[outer_rows]):
                raise RuntimeError(f"{name} saved outer row order mismatch")
            reproduction[name] = max(reproduction[name], float(np.max(np.abs(
                saved_outer["probability"].to_numpy() - probability[outer_rows]))))
            part[f"{name}_probability"] = probability
        parts.append(part)
    if max(reproduction.values()) > 1e-5:
        raise RuntimeError(f"checkpoints do not reproduce the saved out-of-fold scores: {reproduction}")
    table = pd.concat(parts, ignore_index=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(cache_path, index=False)
        cache_path.with_suffix(".json").write_text(json.dumps(
            {"graph_run_dir": str(graph_run_dir), "cnn_run_dir": str(cnn_run_dir), "checkpoints": hashes,
             "maximum_reproduction_difference": reproduction, "canonical_test_access": False},
            indent=2, sort_keys=True) + "\n")
    return table, reproduction


def run(
    repository: Path,
    site_master_path: Path,
    graph_run_dir: Path,
    cnn_run_dir: Path,
    ensemble_dir: Path,
    context: str,
    output_dir: Path,
    batch_size: int,
    bootstrap_replicates: int,
    seed: int,
    device: torch.device,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    run_tag: str | None = None,
) -> dict:
    run_tag_suffix(run_tag)  # refuse a bad tag before any work
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("gap stage received a sealed split")
    canonical = master[master["context"] == context].copy().reset_index(drop=True)
    if canonical.empty:
        raise RuntimeError(f"no canonical rows for {context}")
    gaps, shuffle_hash, kept_share = shuffled_gap_values(pair_gap_table(canonical), shuffle_seed)
    graphs = build_graphs(canonical, upstream_modules(repository))
    windows = canonical_window_examples(canonical, partner_lane=run_dir_partner_lane(cnn_run_dir, device))
    ensemble_folds = pd.read_csv(ensemble_dir / "fold_selections.csv").set_index("outer_fold")
    saved_ensemble = pd.read_parquet(ensemble_dir / "oof_predictions.parquet").set_index("example_id")
    labels = canonical["label"].to_numpy()
    outputs, selections, checkpoint_hashes = [], [], {}
    reproduction = {"graph": 0.0, "cnn": 0.0, REFERENCE: 0.0}
    for outer_fold in range(5):
        probabilities = {}
        for name, mode, run_dir, examples in (("graph", GRAPH_MODE, graph_run_dir, graphs),
                                              ("cnn", CNN_MODE, cnn_run_dir, windows)):
            checkpoint = run_dir / f"outer_{outer_fold}" / "best.pth"
            checkpoint_hashes[f"{name}_outer_{outer_fold}"] = sha256_file(checkpoint)
            probability, predicted_labels = score_all_rows(checkpoint, mode, examples, batch_size, device)
            if not np.array_equal(predicted_labels, labels):
                raise RuntimeError(f"{name} prediction order mismatch")
            saved_outer = pd.read_parquet(run_dir / f"outer_{outer_fold}" / "outer_predictions.parquet")
            outer_rows = canonical.index[canonical["development_fold"] == outer_fold].to_numpy()
            if not np.array_equal(saved_outer["example_id"].to_numpy(), canonical.loc[outer_rows, "example_id"].to_numpy()):
                raise RuntimeError(f"{name} saved outer row order mismatch")
            difference = float(np.max(np.abs(saved_outer["probability"].to_numpy() - probability[outer_rows])))
            reproduction[name] = max(reproduction[name], difference)
            probabilities[name] = probability
        table = stage_table(canonical, context, probabilities["graph"], probabilities["cnn"], gaps)
        if not table["example_id"].equals(canonical["example_id"]):  # positions below index the canonical order
            raise RuntimeError("stage table lost the canonical row order")
        reference_probability = equal_logit_average_many([probabilities["graph"], probabilities["cnn"]])
        reference_threshold = float(ensemble_folds.loc[outer_fold, "threshold"])
        table["expanded_reference_prediction"] = (reference_probability >= reference_threshold).astype(int)
        reference = table[table["development_fold"] == outer_fold].copy()
        reference["probability"] = reference_probability[reference.index]
        reference["threshold"] = reference_threshold
        reference["prediction"] = reference["expanded_reference_prediction"]
        saved_rows = saved_ensemble.loc[reference["example_id"]]
        reproduction[REFERENCE] = max(reproduction[REFERENCE], float(np.max(np.abs(
            saved_rows["probability"].to_numpy() - reference["probability"].to_numpy()))))
        if not np.array_equal(saved_rows["prediction"].to_numpy(), reference["prediction"].to_numpy()):
            raise RuntimeError("equal average does not reproduce the saved ensemble predictions")
        reference["model"] = REFERENCE
        reference["matched_recall_threshold"] = np.nan
        reference["matched_recall_prediction"] = reference["prediction"]
        outputs.append(reference)
        selections.append({"model": REFERENCE, "outer_fold": outer_fold, "inner_fold": (outer_fold + 1) % 5,
                           "threshold": reference_threshold})
        for arm, columns in ARMS.items():
            outer, selection = fit_arm(table, columns, outer_fold, seed)
            outer["model"] = arm
            outputs.append(outer)
            selection["model"] = arm
            selections.append(selection)
    if max(reproduction.values()) > 1e-5:
        raise RuntimeError(f"checkpoints do not reproduce the saved out-of-fold scores: {reproduction}")
    keep = IDENTITY + ["baseline_oof_probability", "baseline_oof_prediction", "graph_probability", "cnn_probability",
                       "gap", "shuffled_gap", "model", "probability", "threshold", "prediction",
                       "matched_recall_threshold", "matched_recall_prediction"]
    predictions = pd.concat([frame[keep] for frame in outputs], ignore_index=True)
    reference_rows = predictions[predictions["model"] == REFERENCE]
    metric_rows, bootstraps = [], []
    for name, group in predictions.groupby("model", sort=False):
        if len(group) != len(canonical) or set(group["example_id"]) != set(canonical["example_id"]):
            raise RuntimeError(f"{name} does not cover every canonical row exactly once")
        row = metrics(group["label"], group["probability"], group["prediction"])
        row["model"] = name
        metric_rows.append(row)
        if name != REFERENCE:
            values = bootstrap_differences(group, reference_rows, bootstrap_replicates, seed + 4000)
            values["model"] = name
            bootstraps.append(values)
    bootstrap = pd.concat(bootstraps, ignore_index=True)
    confidence = (
        bootstrap.groupby(["model", "metric"])["difference"].quantile([0.025, 0.5, 0.975]).unstack().reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    destination = output_dir / context
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(destination / "oof_predictions.parquet", index=False)
    for name, group in predictions.groupby("model", sort=False):  # comparison-ready, one file per arm
        arm_dir = destination / "arms" / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        group.sort_values("example_id").to_parquet(arm_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(destination / "fold_selections.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(destination / "metrics.csv", index=False)
    bootstrap.to_parquet(destination / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(destination / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "gap-only second stage (PNAS gate scheme) on graph + CNN alone; development rows only",
        "context": context,
        "rows": len(canonical),
        "pairs": int(canonical["pair_id"].nunique()),
        "arms": {name: columns for name, columns in ARMS.items()},
        "reference": "fixed equal-logit average, saved ensemble thresholds",
        "graph_run_dir": str(graph_run_dir),
        "cnn_run_dir": str(cnn_run_dir),
        "ensemble_dir": str(ensemble_dir),
        "shuffle_seed": shuffle_seed,
        "shuffle_hash": shuffle_hash,
        "shuffle_share_pairs_keeping_value": kept_share,
        "maximum_reproduction_difference": reproduction,
        "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    register(
        repository,
        {
            "run_id": f"gap_stage_{context}_seed{seed}" + run_tag_suffix(run_tag),
            "experiment_type": "gap_only_second_stage_gate_scheme",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "checkpoints": checkpoint_hashes,
                "ensemble_fold_selections": sha256_file(ensemble_dir / "fold_selections.csv"),
                "ensemble_predictions": sha256_file(ensemble_dir / "oof_predictions.parquet"),
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5; stage fitted on the three training folds",
            "seed": seed,
            "features": "graph and CNN-alone log-odds (outer-k checkpoints) plus log1p genomic gap (or its fixed shuffle)",
            "hyperparameters": {
                "stage": "L2 logistic (hierarchical_gate.fit_gate); C and F1 threshold selected on the inner fold",
                "shuffle_seed": shuffle_seed,
                "bootstrap_replicates": bootstrap_replicates,
            },
            "artifact": str(destination / "oof_predictions.parquet"),
            "metrics": summary,
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--site-master", type=Path, default=Path("data/derived/site_master.parquet"))
    parser.add_argument("--graph-run-dir", type=Path, required=True, help="…/binary_pair_balanced/<context>")
    parser.add_argument("--cnn-run-dir", type=Path, required=True, help="…/binary_pair_balanced_cnn/<context>")
    parser.add_argument("--ensemble-dir", type=Path, required=True, help="…/ensembles/graph_cnn/<context>")
    parser.add_argument("--context", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True, help="the networks' run seed (names the run)")
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--run-tag", help="append _<tag> to the registry run_id (ASCII letters/digits)")
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(json.dumps(run(root, resolve(args.site_master), resolve(args.graph_run_dir), resolve(args.cnn_run_dir),
                         resolve(args.ensemble_dir), args.context, resolve(args.output_dir), args.batch_size,
                         args.bootstrap_replicates, args.seed, torch.device(args.device), args.shuffle_seed,
                         args.run_tag), indent=2))


if __name__ == "__main__":
    main()
