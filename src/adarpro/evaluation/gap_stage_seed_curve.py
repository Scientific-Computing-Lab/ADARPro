"""Seed-averaging curve for the gap-stage predictor, development rows only.

For every subset of the development runs (sizes 1..5), each network's out-of-fold-model scores are averaged over the
subset's runs in log-odds, and one gap stage is fitted on top with PNAS's gate scheme (``gap_stage.fit_arm``: fit on
the three training folds, penalty and threshold on the inner fold, applied to the outer fold). Reported beside it, on
the same averaged scores: the equal average of the two averaged networks (threshold on the inner fold, as
``fixed_checkpoint_ensemble``) and each averaged network alone (threshold on the inner fold).

Anchor: every one-run subset must reproduce that run's gap-stage and equal-average out-of-fold predictions
(experiments/runs/step9) exactly.
Headline: d_avg = metric(all runs averaged) - mean metric of the single runs, with a pair bootstrap (runs fixed).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from adarpro.models.baseline_oof import build_graphs, choose_threshold, upstream_modules
from adarpro.models.fixed_checkpoint_ensemble import equal_logit_average_many
from adarpro.models.gap_stage import (
    ARMS,
    DEFAULT_SHUFFLE_SEED,
    IDENTITY,
    average_probabilities,
    fit_arm,
    pair_gap_table,
    score_run,
    shuffled_gap_values,
    stage_table,
)
from adarpro.models.multitask_oof import (
    canonical_window_examples,
    run_dir_partner_lane,
    run_tag_suffix,
)
from adarpro.models.residual import metrics as pnas_metrics
from adarpro.quarantine import sha256_file
from adarpro.registry import register

MODELS = ("predictor", "equal_average", "graph", "cnn")
METRICS = ("auroc", "auprc", "f1", "brier")


def four(labels: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.array([roc_auc_score(labels, probability), average_precision_score(labels, probability),
                     f1_score(labels, prediction, zero_division=0), float(np.mean((probability - labels) ** 2))])


def subset_predictions(canonical, context, gaps, scores: dict, subset: tuple) -> dict[str, pd.DataFrame]:
    """Out-of-fold predictions of the four models for one subset of runs (canonical order within each fold).

    The stage's solver seed is the subset's first run seed, so a one-run subset is fitted exactly as the gap-stage run fitted
    that run (gap_stage --seed <run seed>)."""
    seed = subset[0]
    labels = canonical["label"].to_numpy()
    out = {name: [] for name in MODELS}
    for outer_fold in range(5):
        inner_fold = (outer_fold + 1) % 5
        graph_p = average_probabilities([scores[s].loc[scores[s]["outer_fold"] == outer_fold, "graph_probability"]
                                         .to_numpy() for s in subset])
        cnn_p = average_probabilities([scores[s].loc[scores[s]["outer_fold"] == outer_fold, "cnn_probability"]
                                       .to_numpy() for s in subset])
        table = stage_table(canonical, context, graph_p, cnn_p, gaps)
        if not table["example_id"].equals(canonical["example_id"]):
            raise RuntimeError("stage table lost the canonical row order")
        inner = (canonical["development_fold"] == inner_fold).to_numpy()
        outer = (canonical["development_fold"] == outer_fold).to_numpy()
        ensemble_p = equal_logit_average_many([graph_p, cnn_p])
        ensemble_threshold, _ = choose_threshold(labels[inner], ensemble_p[inner])
        table["expanded_reference_prediction"] = (ensemble_p >= ensemble_threshold).astype(int)
        stage_outer, _ = fit_arm(table, ARMS["gap"], outer_fold, seed)
        out["predictor"].append(stage_outer[IDENTITY + ["probability", "threshold", "prediction"]])
        for name, probability in (("equal_average", ensemble_p), ("graph", graph_p), ("cnn", cnn_p)):
            threshold = ensemble_threshold if name == "equal_average" else choose_threshold(labels[inner],
                                                                                             probability[inner])[0]
            frame = canonical.loc[outer, IDENTITY].copy()
            frame["probability"] = probability[outer]
            frame["threshold"] = float(threshold)
            frame["prediction"] = (probability[outer] >= threshold).astype(int)
            out[name].append(frame)
    return {name: pd.concat(frames, ignore_index=True).sort_values("example_id").reset_index(drop=True)
            for name, frames in out.items()}


def run(
    repository: Path,
    site_master_path: Path,
    context: str,
    runs: dict[int, Path],
    cache_dir: Path,
    output_dir: Path,
    anchor_dir: Path | None,
    batch_size: int,
    bootstrap_replicates: int,
    seed: int,
    device: torch.device,
    run_tag: str | None = None,
) -> dict:
    run_tag_suffix(run_tag)  # refuse a bad tag before any work
    if len(runs) < 2:
        raise ValueError("a seed-averaging curve needs at least two runs")
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("seed curve received a sealed split")
    canonical = master[master["context"] == context].copy().reset_index(drop=True)
    gaps, _, _ = shuffled_gap_values(pair_gap_table(canonical), DEFAULT_SHUFFLE_SEED)
    lanes = {run_dir_partner_lane(base / "binary_pair_balanced_cnn" / context, device) for base in runs.values()}
    if len(lanes) != 1:
        raise RuntimeError(f"runs disagree on the CNN partner lane: {lanes}")
    examples = (build_graphs(canonical, upstream_modules(repository)),
                canonical_window_examples(canonical, partner_lane=lanes.pop()))
    scores, reproduction = {}, {}
    for s, base in sorted(runs.items()):
        scores[s], reproduction[s] = score_run(
            repository, canonical, base / "binary_pair_balanced" / context, base / "binary_pair_balanced_cnn" / context,
            batch_size, device, cache_path=cache_dir / f"seed{s}" / f"{context}.parquet", examples=examples)
    seeds = sorted(runs)
    rows, kept, anchor = [], {}, {}
    for k in range(1, len(seeds) + 1):
        for subset in itertools.combinations(seeds, k):
            predictions = subset_predictions(canonical, context, gaps, scores, subset)
            for name, frame in predictions.items():
                values = pnas_metrics(frame["label"], frame["probability"], frame["prediction"])
                rows.append({"k": k, "subset": "+".join(map(str, subset)), "model": name,
                             **{m: float(values[m]) for m in METRICS}})
            if k == 1:
                kept[subset[0]] = predictions
                if anchor_dir is not None:
                    for name, arm in (("predictor", "gap"), ("equal_average", "equal_average")):
                        saved = pd.read_parquet(anchor_dir / f"seed{subset[0]}/gap_stage/{context}/arms/{arm}/"
                                                "oof_predictions.parquet").sort_values("example_id")
                        mine = predictions[name]
                        if not np.array_equal(saved["example_id"].to_numpy(), mine["example_id"].to_numpy()):
                            raise RuntimeError(f"anchor rows differ: {subset} {name}")
                        dp = float(np.max(np.abs(saved["probability"].to_numpy() - mine["probability"].to_numpy())))
                        same = bool(np.array_equal(saved["prediction"].to_numpy(), mine["prediction"].to_numpy()))
                        anchor[f"seed{subset[0]}_{name}"] = {"max_abs_probability_difference": dp,
                                                             "predictions_identical": same}
                        if dp > 1e-9 or not same:
                            raise RuntimeError(f"one-run subset does not reproduce its gap-stage run: {subset} {name} ({dp})")
            if k == len(seeds):
                kept["all"] = predictions
    curve = pd.DataFrame(rows)
    summary_rows = []
    for (k, name), group in curve.groupby(["k", "model"]):
        for metric in METRICS:
            values = group[metric].to_numpy()
            summary_rows.append({"k": k, "model": name, "metric": metric, "subsets": len(values),
                                 "mean": float(values.mean()), "sd": float(values.std(ddof=0)),
                                 "min": float(values.min()), "max": float(values.max())})
    curve_summary = pd.DataFrame(summary_rows)
    # headline: all runs averaged vs the mean of single runs, pairs resampled (the same draw for every model)
    reference = kept["all"]["predictor"]
    labels = reference["label"].to_numpy()
    pair_ids = reference["pair_id"].to_numpy()
    pairs = np.unique(pair_ids)
    members = {pair: np.flatnonzero(pair_ids == pair) for pair in pairs}
    rng = np.random.default_rng(seed + 5000)
    boot = {name: [] for name in MODELS}
    for _ in range(bootstrap_replicates):
        rows_ = np.concatenate([members[p] for p in rng.choice(pairs, size=len(pairs), replace=True)])
        for name in MODELS:
            full = four(labels[rows_], kept["all"][name]["probability"].to_numpy()[rows_],
                        kept["all"][name]["prediction"].to_numpy()[rows_])
            singles = np.mean([four(labels[rows_], kept[s][name]["probability"].to_numpy()[rows_],
                                    kept[s][name]["prediction"].to_numpy()[rows_]) for s in seeds], axis=0)
            boot[name].append(full - singles)
    d_rows = []
    for name in MODELS:
        full = four(labels, kept["all"][name]["probability"].to_numpy(), kept["all"][name]["prediction"].to_numpy())
        singles = np.mean([four(labels, kept[s][name]["probability"].to_numpy(), kept[s][name]["prediction"].to_numpy())
                           for s in seeds], axis=0)
        values = np.asarray(boot[name])
        for j, metric in enumerate(METRICS):
            lower, median, upper = np.quantile(values[:, j], [0.025, 0.5, 0.975])
            d_rows.append({"model": name, "metric": metric, "all_runs": float(full[j]), "mean_single_run": float(singles[j]),
                           "d_avg": float(full[j] - singles[j]), "lower": float(lower), "median": float(median),
                           "upper": float(upper)})
    d_avg = pd.DataFrame(d_rows)
    destination = output_dir / context
    destination.mkdir(parents=True, exist_ok=True)
    curve.to_csv(destination / "curve.csv", index=False)
    curve_summary.to_csv(destination / "curve_summary.csv", index=False)
    d_avg.to_csv(destination / "d_avg.csv", index=False)
    for name, frame in kept["all"].items():  # the all-runs averaged models' out-of-fold predictions
        model_dir = destination / "all_runs" / name
        model_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(model_dir / "oof_predictions.parquet", index=False)
    summary = {
        "scope": "seed-averaging curve of the gap-stage predictor; development rows only",
        "context": context, "runs": seeds, "subsets": int(curve["subset"].nunique()),
        "rows": len(canonical), "pairs": int(canonical["pair_id"].nunique()),
        "averaging": "each network's scores averaged over the subset's runs in log-odds, then one gap stage",
        "anchor_step9": anchor, "maximum_reproduction_difference": {str(s): r for s, r in reproduction.items()},
        "bootstrap_replicates": bootstrap_replicates, "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    register(
        repository,
        {
            "run_id": f"gap_stage_seed_curve_{context}_seed{seed}" + run_tag_suffix(run_tag),
            "experiment_type": "gap_stage_seed_averaging_curve",
            "context": context,
            "input_hashes": {"site_master": sha256_file(site_master_path),
                             "score_caches": {str(s): sha256_file(cache_dir / f"seed{s}" / f"{context}.parquet")
                                              for s in seeds}},
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5; stage fitted on the three training folds",
            "seed": seed,
            "features": "per-network log-odds averaged over run subsets, plus log1p genomic gap",
            "hyperparameters": {"subset_sizes": list(range(1, len(seeds) + 1)),
                                "bootstrap_replicates": bootstrap_replicates},
            "artifact": str(destination / "curve.csv"),
            "metrics": summary,
            "canonical_test_access": False,
        },
    )
    return summary


def parse_runs(values: list[str], root: Path) -> dict[int, Path]:
    runs = {}
    for value in values:
        seed, separator, base = value.partition("=")
        if not separator or not seed.isdigit() or not base:
            raise ValueError(f"--run must be SEED=PATH, got {value!r}")
        if int(seed) in runs:
            raise ValueError(f"duplicate run seed {seed}")
        path = Path(base)
        runs[int(seed)] = path if path.is_absolute() else root / path
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--site-master", type=Path, default=Path("data/derived/site_master.parquet"))
    parser.add_argument("--context", required=True)
    parser.add_argument("--run", action="append", required=True,
                        help="SEED=PATH, PATH holding binary_pair_balanced/<context> and binary_pair_balanced_cnn/<context>")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-dir", type=Path, help="the gap-stage runs (experiments/runs/step9); one-run subsets must reproduce them")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--run-tag")
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path | None) -> Path | None:
        return None if path is None else (path if path.is_absolute() else root / path)

    print(json.dumps(run(root, resolve(args.site_master), args.context, parse_runs(args.run, root),
                         resolve(args.cache_dir), resolve(args.output_dir), resolve(args.anchor_dir), args.batch_size,
                         args.bootstrap_replicates, args.seed, torch.device(args.device), args.run_tag), indent=2))


if __name__ == "__main__":
    main()
