"""Matched development-only analysis of RBP evidence at baseline false positives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from adarpro.evaluation.error_atlas import add_intrinsic_descriptors

EXPOSURES = {
    "ILF3_eclip_near25": "rbp__ILF3__eclip_focal_near_w25_any",
    "ILF3_eclip_duplex": "rbp__ILF3__eclip_duplex_peak_any",
    "ILF3_rbpnet_local": "pred__ILF3__target_focal_mean_w25",
    "ILF3_rbpnet_duplex": "pred__ILF3__target_duplex_max",
    "ILF3_prismnet_local": "prism__ILF3__prismnet_focal_max_score_w25",
    "ILF3_prismnet_duplex": "prism__ILF3__prismnet_duplex_max_score",
    "ILF3_prismnet_local_x_expression": (
        "prism__ILF3__prismnet_focal_max_score_w25_x_log1p_tpm"
    ),
}
MATCH_COLUMNS = (
    "baseline_oof_logit",
    "focal_is_paired",
    "local_gc_21",
    "local_gc_51",
    "pair_span_log1p",
    "sequence_length_log1p",
)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return pd.Series(output, index=p_values.index)


def prepare_table(matrix: pd.DataFrame) -> pd.DataFrame:
    if set(matrix["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("false-positive analysis contains a sealed split")
    required = [
        "baseline_oof_probability",
        "baseline_oof_logit",
        "baseline_oof_prediction",
        *EXPOSURES.values(),
    ]
    missing = set(required) - set(matrix.columns)
    if missing:
        raise RuntimeError(f"false-positive matrix lacks {sorted(missing)}")
    if matrix[["baseline_oof_probability", "baseline_oof_logit"]].isna().any(axis=None):
        raise RuntimeError("false-positive analysis requires complete OOF predictions")
    table = add_intrinsic_descriptors(matrix.copy())
    table["pair_span_log1p"] = np.log1p(table["pair_span"].clip(lower=0))
    table["sequence_length_log1p"] = np.log1p(table["sequence_length"])
    table["baseline_status"] = np.select(
        [
            (table["label"] == 1) & (table["baseline_oof_prediction"] == 1),
            (table["label"] == 0) & (table["baseline_oof_prediction"] == 1),
            (table["label"] == 0) & (table["baseline_oof_prediction"] == 0),
        ],
        ["TP", "FP", "TN"],
        default="FN",
    )
    return table


def match_controls(table: pd.DataFrame, control_status: str) -> pd.DataFrame:
    matches = []
    for context, context_table in table.groupby("context", sort=True):
        focal = context_table[context_table["baseline_status"] == "FP"].copy()
        control = context_table[context_table["baseline_status"] == control_status].copy()
        if focal.empty or control.empty:
            continue
        combined = pd.concat([focal[list(MATCH_COLUMNS)], control[list(MATCH_COLUMNS)]])
        combined = combined.replace([np.inf, -np.inf], np.nan)
        medians = combined.median().fillna(0)
        focal_values = focal[list(MATCH_COLUMNS)].replace([np.inf, -np.inf], np.nan).fillna(medians)
        control_values = control[list(MATCH_COLUMNS)].replace([np.inf, -np.inf], np.nan).fillna(medians)
        scaler = StandardScaler().fit(combined.fillna(medians))
        focal_scaled = scaler.transform(focal_values)
        control_scaled = scaler.transform(control_values)
        # Baseline confidence is the primary matching variable.
        focal_scaled[:, 0] *= 2.0
        control_scaled[:, 0] *= 2.0
        distances, indices = NearestNeighbors(n_neighbors=1).fit(control_scaled).kneighbors(
            focal_scaled
        )
        chosen = control.iloc[indices[:, 0]].reset_index(drop=True)
        focal = focal.reset_index(drop=True)
        for index in range(len(focal)):
            row = {
                "context": context,
                "control_status": control_status,
                "focal_example_id": focal.loc[index, "example_id"],
                "focal_pair_id": int(focal.loc[index, "pair_id"]),
                "control_example_id": chosen.loc[index, "example_id"],
                "control_pair_id": int(chosen.loc[index, "pair_id"]),
                "match_distance": float(distances[index, 0]),
                "baseline_probability_difference": float(
                    focal.loc[index, "baseline_oof_probability"]
                    - chosen.loc[index, "baseline_oof_probability"]
                ),
            }
            for name, column in EXPOSURES.items():
                row[f"focal__{name}"] = focal.loc[index, column]
                row[f"control__{name}"] = chosen.loc[index, column]
                row[f"difference__{name}"] = focal.loc[index, column] - chosen.loc[index, column]
            matches.append(row)
    if not matches:
        raise RuntimeError(f"no matched FP/{control_status} examples")
    return pd.DataFrame(matches)


def clustered_bootstrap(
    matches: pd.DataFrame, replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, summaries = [], []
    for control_status, comparison in matches.groupby("control_status", sort=True):
        pairs = comparison["focal_pair_id"].unique()
        groups = {
            pair_id: comparison[comparison["focal_pair_id"] == pair_id]
            for pair_id in pairs
        }
        for name in EXPOSURES:
            column = f"difference__{name}"
            observed = float(comparison[column].mean())
            boot = np.empty(replicates, dtype=float)
            for replicate in range(replicates):
                sampled = rng.choice(pairs, size=len(pairs), replace=True)
                values = pd.concat([groups[pair_id][column] for pair_id in sampled])
                boot[replicate] = values.mean()
                rows.append(
                    {
                        "control_status": control_status,
                        "exposure": name,
                        "replicate": replicate,
                        "mean_difference": boot[replicate],
                    }
                )
            tail = min(np.mean(boot <= 0), np.mean(boot >= 0))
            summaries.append(
                {
                    "comparison": f"FP_minus_{control_status}",
                    "control_status": control_status,
                    "exposure": name,
                    "matched_sets": len(comparison),
                    "focal_pairs": len(pairs),
                    "mean_difference": observed,
                    "ci_lower": float(np.quantile(boot, 0.025)),
                    "ci_median": float(np.quantile(boot, 0.5)),
                    "ci_upper": float(np.quantile(boot, 0.975)),
                    "two_sided_bootstrap_p": float(min(1.0, 2 * tail)),
                }
            )
    summary = pd.DataFrame(summaries)
    summary["bh_q"] = benjamini_hochberg(summary["two_sided_bootstrap_p"])
    return pd.DataFrame(rows), summary


def run(
    matrix_path: Path,
    context: str,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    matrix = pd.read_parquet(matrix_path)
    if context == "IndividualPooled":
        matrix = matrix[matrix["context"] != "Combined"].copy()
    else:
        matrix = matrix[matrix["context"] == context].copy()
    table = prepare_table(matrix)
    matches = pd.concat(
        [match_controls(table, "TP"), match_controls(table, "TN")],
        ignore_index=True,
    )
    bootstrap, summary = clustered_bootstrap(matches, bootstrap_replicates, seed)
    output = output_dir / context
    output.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(output / "matches.parquet", index=False)
    bootstrap.to_parquet(output / "pair_bootstrap.parquet", index=False)
    summary.to_csv(output / "enrichment_summary.csv", index=False)
    balance = (
        matches.groupby("control_status", as_index=False)
        .agg(
            matched_sets=("focal_example_id", "size"),
            focal_pairs=("focal_pair_id", "nunique"),
            mean_match_distance=("match_distance", "mean"),
            median_absolute_probability_difference=(
                "baseline_probability_difference",
                lambda values: np.median(np.abs(values)),
            ),
            maximum_absolute_probability_difference=(
                "baseline_probability_difference",
                lambda values: np.max(np.abs(values)),
            ),
        )
    )
    balance.to_csv(output / "match_balance.csv", index=False)
    result = {
        "scope": "development train+valid grouped-OOF predictions only",
        "context": context,
        "false_positives": int((table["baseline_status"] == "FP").sum()),
        "false_positive_pairs": int(
            table.loc[table["baseline_status"] == "FP", "pair_id"].nunique()
        ),
        "exposures": EXPOSURES,
        "matching": {
            "exact": "context",
            "nearest_neighbor_with_replacement": list(MATCH_COLUMNS),
            "baseline_logit_weight": 2.0,
        },
        "bootstrap_replicates": bootstrap_replicates,
        "canonical_test_access": False,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("data/derived/features/residual_model_matrix.parquet"),
    )
    parser.add_argument(
        "--context",
        choices=("Artery", "Brain", "Liver", "MuscleSkeletal", "Combined", "IndividualPooled"),
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/false_positive_enrichment"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(
        json.dumps(
            run(
                resolve(args.matrix),
                args.context,
                resolve(args.output_dir),
                args.bootstrap_replicates,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
