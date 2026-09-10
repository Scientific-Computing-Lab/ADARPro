"""Assess substrate-level editing stability on matched sites across tissues."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from adarpro.quarantine import sha256_file
from adarpro.registry import register

TISSUES = ("Artery", "Brain", "Liver", "MuscleSkeletal")


def pair_summaries(
    matched: pd.DataFrame, first: str, second: str, min_sites: int
) -> pd.DataFrame:
    """Summarize both tissues on the exact same sites within each Alu pair."""

    shared = matched.loc[matched[first].notna() & matched[second].notna()].copy()
    shared[f"{first}_high"] = shared[first].ge(15).astype(float)
    shared[f"{second}_high"] = shared[second].ge(15).astype(float)
    result = (
        shared.groupby("pair_id")
        .agg(
            matched_sites=("site_id", "size"),
            first_mean_editing=(first, "mean"),
            second_mean_editing=(second, "mean"),
            first_high_fraction=(f"{first}_high", "mean"),
            second_high_fraction=(f"{second}_high", "mean"),
        )
        .reset_index()
    )
    result = result.loc[result["matched_sites"].ge(min_sites)].copy()
    result.insert(0, "first_tissue", first)
    result.insert(1, "second_tissue", second)
    return result


def correlations(table: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_editing_spearman": float(
            spearmanr(table["first_mean_editing"], table["second_mean_editing"]).statistic
        ),
        "high_fraction_spearman": float(
            spearmanr(
                table["first_high_fraction"], table["second_high_fraction"]
            ).statistic
        ),
    }


def bootstrap_correlations(
    table: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        draw = table.iloc[rng.integers(0, len(table), size=len(table))]
        rows.extend(
            {
                "replicate": replicate,
                "metric": metric,
                "spearman": value,
            }
            for metric, value in correlations(draw).items()
        )
    return pd.DataFrame(rows)


def run(
    repository: Path,
    matched_path: Path,
    output_dir: Path,
    source_dir: Path,
    min_sites: int,
    replicates: int,
    seed: int,
) -> dict:
    matched = pd.read_parquet(matched_path)
    all_pairs = []
    all_bootstrap = []
    summary_rows = []
    for comparison_index, (first, second) in enumerate(combinations(TISSUES, 2)):
        pair_table = pair_summaries(matched, first, second, min_sites)
        if pair_table.empty:
            raise RuntimeError(f"no eligible pairs for {first} versus {second}")
        observed = correlations(pair_table)
        bootstrap = bootstrap_correlations(
            pair_table, replicates, seed + 1000 * comparison_index
        )
        bootstrap.insert(0, "first_tissue", first)
        bootstrap.insert(1, "second_tissue", second)
        all_pairs.append(pair_table)
        all_bootstrap.append(bootstrap)
        intervals = (
            bootstrap.groupby("metric")["spearman"]
            .quantile([0.025, 0.5, 0.975])
            .unstack()
        )
        for metric, value in observed.items():
            summary_rows.append(
                {
                    "first_tissue": first,
                    "second_tissue": second,
                    "metric": metric,
                    "eligible_pairs": len(pair_table),
                    "median_matched_sites": float(pair_table["matched_sites"].median()),
                    "spearman": value,
                    "lower": float(intervals.loc[metric, 0.025]),
                    "bootstrap_median": float(intervals.loc[metric, 0.5]),
                    "upper": float(intervals.loc[metric, 0.975]),
                }
            )

    pair_tables = pd.concat(all_pairs, ignore_index=True)
    bootstraps = pd.concat(all_bootstrap, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    pair_tables.to_parquet(output_dir / "matched_pair_summaries.parquet", index=False)
    bootstraps.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    summary.to_csv(output_dir / "cross_tissue_pair_stability.csv", index=False)
    summary.to_csv(source_dir / "cross_tissue_pair_stability.csv", index=False)

    retains = bool(
        summary["eligible_pairs"].ge(100).all()
        and summary["spearman"].gt(0).all()
        and summary["lower"].gt(0).all()
    )
    payload = {
        "scope": "post-hoc descriptive matched-site analysis on development pairs",
        "individual_tissues": list(TISSUES),
        "combined_excluded": True,
        "minimum_matched_sites_per_pair": min_sites,
        "bootstrap_replicates": replicates,
        "seed": seed,
        "canonical_test_access": False,
        "prespecified_retention_rule_passed": retains,
        "comparisons": summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": "post_analysis_cross_tissue_pair_stability_v1",
            "experiment_type": "post_hoc_matched_site_pair_propensity_stability",
            "context": "six pairwise comparisons among individual GTEx tissues",
            "input_hashes": {"matched_sites": sha256_file(matched_path)},
            "fold_definition": "development pairs only; same sites in each tissue pair",
            "seed": seed,
            "features": "pair-level mean EditingIndex and high-editing-site fraction",
            "hyperparameters": {
                "minimum_matched_sites_per_pair": min_sites,
                "bootstrap_replicates": replicates,
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
        default=Path("experiments/runs/post_analysis/pair_stability"),
    )
    parser.add_argument(
        "--source-dir", type=Path, default=Path("paper/source_tables")
    )
    parser.add_argument("--minimum-matched-sites", type=int, default=5)
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
                args.minimum_matched_sites,
                args.bootstrap_replicates,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
