"""Describe within-site editing variation across individual tissues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.quarantine import sha256_file
from adarpro.registry import register

TISSUES = ("Artery", "Brain", "Liver", "MuscleSkeletal")
RANGE_THRESHOLDS = (1.0, 5.0, 10.0, 15.0)


def add_variation(table: pd.DataFrame) -> pd.DataFrame:
    """Add site-level summaries without imputing unobserved tissues."""

    missing = (
        set(TISSUES)
        | {
            "site_id",
            "pair_id",
            "observed_tissues",
            "switch_category",
        }
    ) - set(table)
    if missing:
        raise RuntimeError(f"matched-site table lacks columns: {sorted(missing)}")
    result = table.copy()
    values = result.loc[:, TISSUES]
    result["editing_index_min"] = values.min(axis=1, skipna=True)
    result["editing_index_max"] = values.max(axis=1, skipna=True)
    result["editing_index_range"] = (
        result["editing_index_max"] - result["editing_index_min"]
    )
    result["editing_index_sd"] = values.std(axis=1, skipna=True, ddof=0)
    result["strict_switch"] = result["switch_category"].eq(
        "strict_edited_nonedited_switch"
    )
    return result


def point_metrics(table: pd.DataFrame) -> dict[str, float]:
    metrics = {
        "sites": float(len(table)),
        "pairs": float(table["pair_id"].nunique()),
        "mean_range": float(table["editing_index_range"].mean()),
        "median_range": float(table["editing_index_range"].median()),
        "strict_switch_fraction": float(table["strict_switch"].mean()),
    }
    for threshold in RANGE_THRESHOLDS:
        metrics[f"fraction_range_ge_{threshold:g}"] = float(
            table["editing_index_range"].ge(threshold).mean()
        )
    return metrics


def pair_bootstrap(
    table: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    """Resample whole Alu pairs while retaining all their matched sites."""

    pairs = table["pair_id"].drop_duplicates().to_numpy()
    pair_values = table["pair_id"].to_numpy()
    groups = {pair: np.flatnonzero(pair_values == pair) for pair in pairs}
    ranges = table["editing_index_range"].to_numpy(dtype=float)
    switches = table["strict_switch"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        indices = np.concatenate([groups[pair] for pair in sampled])
        sampled_ranges = ranges[indices]
        metrics = {
            "mean_range": float(sampled_ranges.mean()),
            "median_range": float(np.median(sampled_ranges)),
            "strict_switch_fraction": float(switches[indices].mean()),
            **{
                f"fraction_range_ge_{threshold:g}": float(
                    np.mean(sampled_ranges >= threshold)
                )
                for threshold in RANGE_THRESHOLDS
            },
        }
        rows.extend(
            {"replicate": replicate, "metric": metric, "value": value}
            for metric, value in metrics.items()
            if metric not in {"sites", "pairs"}
        )
    return pd.DataFrame(rows)


def confidence_table(
    table: pd.DataFrame, groups: dict[str, pd.DataFrame], replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    draws = []
    for group_index, (group, subset) in enumerate(groups.items()):
        observed = point_metrics(subset)
        bootstrap = pair_bootstrap(subset, replicates, seed + 1000 * group_index)
        bootstrap.insert(0, "group", group)
        draws.append(bootstrap)
        intervals = (
            bootstrap.groupby("metric")["value"]
            .quantile([0.025, 0.5, 0.975])
            .unstack()
        )
        for metric, value in observed.items():
            row = {
                "group": group,
                "metric": metric,
                "value": value,
                "lower": np.nan,
                "bootstrap_median": np.nan,
                "upper": np.nan,
            }
            if metric in intervals.index:
                row.update(
                    {
                        "lower": float(intervals.loc[metric, 0.025]),
                        "bootstrap_median": float(intervals.loc[metric, 0.5]),
                        "upper": float(intervals.loc[metric, 0.975]),
                    }
                )
            summaries.append(row)
    return pd.DataFrame(summaries), pd.concat(draws, ignore_index=True)


def switch_matrix(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for edited in TISSUES:
        for nonedited in TISSUES:
            if edited == nonedited:
                continue
            shared = table[table[edited].notna() & table[nonedited].notna()]
            count = int(
                (
                    shared[f"{edited}_state"].eq("edited")
                    & shared[f"{nonedited}_state"].eq("nonedited")
                ).sum()
            )
            rows.append(
                {
                    "edited_tissue": edited,
                    "nonedited_tissue": nonedited,
                    "shared_sites": len(shared),
                    "strict_switch_sites": count,
                    "strict_switch_per_10000_shared": 10000 * count / len(shared),
                }
            )
    return pd.DataFrame(rows)


def run(
    repository: Path,
    matched_path: Path,
    output_dir: Path,
    source_dir: Path,
    replicates: int,
    seed: int,
) -> dict:
    matched = add_variation(pd.read_parquet(matched_path))
    if (matched["observed_tissues"] < 2).any():
        raise RuntimeError("matched-site table contains fewer than two tissues")
    groups = {
        "at_least_two_tissues": matched,
        "all_four_tissues": matched.loc[matched["observed_tissues"].eq(4)],
    }
    summary, bootstrap = confidence_table(matched, groups, replicates, seed)
    matrix = switch_matrix(matched)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    retained = matched[
        [
            "site_id",
            "pair_id",
            "development_fold",
            "observed_tissues",
            "switch_category",
            "strict_switch",
            "editing_index_min",
            "editing_index_max",
            "editing_index_range",
            "editing_index_sd",
        ]
    ]
    retained.to_parquet(output_dir / "per_site_variation.parquet", index=False)
    summary.to_csv(output_dir / "variation_summary.csv", index=False)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    matrix.to_csv(output_dir / "strict_switch_matrix.csv", index=False)
    summary.to_csv(source_dir / "matched_site_tissue_variation.csv", index=False)
    matrix.to_csv(source_dir / "matched_site_strict_switch_matrix.csv", index=False)

    primary = summary.loc[summary["group"].eq("all_four_tissues")]
    payload = {
        "scope": "post-hoc descriptive matched-site analysis on development pairs",
        "individual_tissues": list(TISSUES),
        "combined_excluded": True,
        "bootstrap_replicates": replicates,
        "seed": seed,
        "canonical_test_access": False,
        "sites_observed_in_at_least_two_tissues": len(matched),
        "sites_observed_in_all_four_tissues": len(groups["all_four_tissues"]),
        "unique_strict_switch_sites": int(matched["strict_switch"].sum()),
        "all_four_tissue_metrics": {
            row.metric: row.value for row in primary.itertuples(index=False)
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": "post_analysis_matched_site_tissue_variation_v1",
            "experiment_type": "post_hoc_matched_site_tissue_variation",
            "context": "four individual GTEx tissues; Combined excluded",
            "input_hashes": {"matched_sites": sha256_file(matched_path)},
            "fold_definition": "development pairs only; descriptive matched-site analysis",
            "seed": seed,
            "features": "within-site EditingIndex range and strict state switches",
            "hyperparameters": {
                "bootstrap_replicates": replicates,
                "range_thresholds": list(RANGE_THRESHOLDS),
            },
            "artifact": str((output_dir / "summary.json").relative_to(repository)),
            "metrics": payload,
            "canonical_test_access": False,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--matched-sites",
        type=Path,
        default=Path("data/derived/cross_tissue/matched_sites.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/post_analysis/tissue_variation"),
    )
    parser.add_argument(
        "--source-dir", type=Path, default=Path("paper/source_tables")
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    repository = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repository / path

    print(
        json.dumps(
            run(
                repository,
                resolve(args.matched_sites),
                resolve(args.output_dir),
                resolve(args.source_dir),
                args.bootstrap_replicates,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
