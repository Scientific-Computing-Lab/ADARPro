"""The gap stage at natural prevalence, development rows only.

For each outer fold k, the gap stage is fitted exactly as in the gap-stage runs (``gap_stage.fit_arm_model``: PNAS's gate scheme on
the canonical rows, scores from the outer-k checkpoints, read from the run's all-row score cache) and applied to that fold's
raw high/low sites. Those sites' graph and CNN-alone scores are the outer-k models' natural-prevalence predictions
(``expanded_natural_prevalence``), and the gap is the pair's log1p(start2 - end1) from ``pair_intrinsic_features``.
The threshold is the stage's inner-fold F1 threshold, as for every natural-prevalence model.

Anchor (stops on failure): the fitted stage applied to the canonical rows of fold k reproduces the run's gap-stage
out-of-fold predictions (experiments/runs/step9) exactly. Output: an out-of-fold prediction file in the format of PNAS's comparison
script (``richer_supervision``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.models.baseline_oof import choose_threshold
from adarpro.models.fixed_checkpoint_ensemble import equal_logit_average_many
from adarpro.models.gap_stage import (
    ARMS,
    DEFAULT_SHUFFLE_SEED,
    fit_arm_model,
    logit,
    pair_gap_table,
    shuffled_gap_values,
    stage_table,
)
from adarpro.models.multitask_oof import run_tag_suffix
from adarpro.quarantine import sha256_file
from adarpro.registry import register

NP_IDENTITY = ["example_id", "context", "pair_id", "pair_local_key", "development_fold", "editing_index", "label"]
BASELINE = ["baseline_oof_probability", "baseline_oof_threshold", "baseline_oof_prediction"]


def natural_rows(graph_path: Path, cnn_path: Path, context: str) -> pd.DataFrame:
    graph = pd.read_parquet(graph_path).sort_values("example_id").reset_index(drop=True)
    cnn = pd.read_parquet(cnn_path).sort_values("example_id").reset_index(drop=True)
    if not graph[NP_IDENTITY].equals(cnn[NP_IDENTITY]):
        raise RuntimeError("graph and CNN-alone natural-prevalence rows differ")
    if set(graph["context"]) != {context}:
        raise RuntimeError(f"natural-prevalence rows are not all from {context}")
    if not np.allclose(graph[BASELINE].to_numpy(float), cnn[BASELINE].to_numpy(float), rtol=0, atol=1e-12):
        raise RuntimeError("graph and CNN-alone files carry different development references")
    table = graph[NP_IDENTITY + BASELINE].copy()
    table["graph_probability"] = graph["probability"].to_numpy(float)
    table["cnn_probability"] = cnn["probability"].to_numpy(float)
    return table


def run(
    repository: Path,
    site_master_path: Path,
    context: str,
    score_cache: Path,
    graph_natural_path: Path,
    cnn_natural_path: Path,
    anchor_path: Path,
    output_dir: Path,
    seed: int,
    run_tag: str | None = None,
) -> dict:
    run_tag_suffix(run_tag)  # refuse a bad tag before any work
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("natural-prevalence gap stage received a sealed split")
    canonical = master[master["context"] == context].copy().reset_index(drop=True)
    gaps, _, _ = shuffled_gap_values(pair_gap_table(canonical), DEFAULT_SHUFFLE_SEED)  # exactly the gap-stage runs' table
    all_gaps = pair_gap_table(master)  # every development pair, for raw sites of pairs outside this context's rows
    scores = pd.read_parquet(score_cache)
    natural = natural_rows(graph_natural_path, cnn_natural_path, context)
    natural = natural.merge(all_gaps[["pair_id", "gap"]], on="pair_id", how="left", validate="many_to_one")
    if natural["gap"].isna().any():
        raise RuntimeError("a natural-prevalence site has no pair gap")
    anchor = pd.read_parquet(anchor_path).set_index("example_id")
    labels = canonical["label"].to_numpy()
    outputs, selections, anchor_max = [], [], 0.0
    for outer_fold in range(5):
        part = scores[scores["outer_fold"] == outer_fold]
        if not np.array_equal(part["example_id"].to_numpy(), canonical["example_id"].to_numpy()):
            raise RuntimeError(f"score cache rows differ from the canonical rows (fold {outer_fold})")
        graph_p, cnn_p = part["graph_probability"].to_numpy(), part["cnn_probability"].to_numpy()
        table = stage_table(canonical, context, graph_p, cnn_p, gaps)
        inner = (canonical["development_fold"] == (outer_fold + 1) % 5).to_numpy()
        ensemble_p = equal_logit_average_many([graph_p, cnn_p])
        threshold_e, _ = choose_threshold(labels[inner], ensemble_p[inner])
        table["expanded_reference_prediction"] = (ensemble_p >= threshold_e).astype(int)  # matched-recall only
        canonical_outer, selection, model = fit_arm_model(table, ARMS["gap"], outer_fold, seed)
        saved = anchor.loc[canonical_outer["example_id"]]
        anchor_max = max(anchor_max, float(np.max(np.abs(saved["probability"].to_numpy()
                                                           - canonical_outer["probability"].to_numpy()))))
        if not np.array_equal(saved["prediction"].to_numpy(), canonical_outer["prediction"].to_numpy()):
            raise RuntimeError(f"fold {outer_fold}: the refitted stage does not reproduce the gap-stage run's predictions")
        rows = natural[natural["development_fold"] == outer_fold].copy()
        features = pd.DataFrame({"graph_logit": logit(rows["graph_probability"]),
                                 "cnn_logit": logit(rows["cnn_probability"]), "gap": rows["gap"].to_numpy()})
        rows["probability"] = model.predict_proba(features[ARMS["gap"]])[:, 1]
        rows["threshold"] = float(selection["threshold"])
        rows["prediction"] = (rows["probability"] >= rows["threshold"]).astype(int)
        outputs.append(rows)
        selections.append({"outer_fold": outer_fold, **selection})
    if anchor_max > 1e-9:
        raise RuntimeError(f"refitted stage does not reproduce the gap-stage run exactly: {anchor_max}")
    predictions = pd.concat(outputs, ignore_index=True).sort_values("example_id").reset_index(drop=True)
    if len(predictions) != len(natural) or predictions["example_id"].duplicated().any():
        raise RuntimeError("natural-prevalence rows lost or duplicated")
    destination = output_dir / context
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(destination / "oof_predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(destination / "fold_selections.csv", index=False)
    summary = {
        "scope": "gap stage applied to every development high/low site at natural prevalence; held-out global pair "
                 "folds; development rows only",
        "context": context, "rows": len(predictions), "pairs": int(predictions["pair_id"].nunique()),
        "prevalence": float(predictions["label"].mean()), "seed": seed,
        "anchor_step9_max_abs_probability_difference": anchor_max, "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    register(
        repository,
        {
            "run_id": f"gap_stage_natural_prevalence_{context}_seed{seed}" + run_tag_suffix(run_tag),
            "experiment_type": "gap_stage_natural_prevalence_oof",
            "context": context,
            "input_hashes": {"site_master": sha256_file(site_master_path), "score_cache": sha256_file(score_cache),
                             "graph_natural": sha256_file(graph_natural_path),
                             "cnn_natural": sha256_file(cnn_natural_path), "step9_anchor": sha256_file(anchor_path)},
            "fold_definition": "five global pair folds; stage fitted on the canonical training folds of outer k",
            "seed": seed,
            "features": "outer-k graph and CNN-alone log-odds on raw high/low sites plus log1p genomic gap",
            "hyperparameters": {"stage": "L2 logistic (hierarchical_gate.fit_gate); C and threshold on the inner fold"},
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
    parser.add_argument("--context", required=True)
    parser.add_argument("--score-cache", type=Path, required=True, help="the run's all-row score cache (experiments/runs/step11/scores)")
    parser.add_argument("--graph-natural", type=Path, required=True)
    parser.add_argument("--cnn-natural", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True, help="the run's gap-stage oof_predictions.parquet (experiments/runs/step9)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="the run seed (the stage's solver seed, as in the gap-stage run)")
    parser.add_argument("--run-tag")
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(json.dumps(run(root, resolve(args.site_master), args.context, resolve(args.score_cache),
                         resolve(args.graph_natural), resolve(args.cnn_natural), resolve(args.anchor),
                         resolve(args.output_dir), args.seed, args.run_tag), indent=2))


if __name__ == "__main__":
    main()
