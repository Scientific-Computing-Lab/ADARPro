"""Aggregate eCLIP tracks and join tissue abundance without test-set access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.rbp.eclip_features import WINDOWS


def aggregate_tracks(features: pd.DataFrame) -> pd.DataFrame:
    keys = ["pair_local_key", "gene"]
    work = features.copy()
    binary_sources = {
        "focal_overlap_any": work["focal_overlap"] > 0,
        "partner_overlap_any": work["partner_overlap"] > 0,
        "duplex_peak_any": work["duplex_peak_count"] > 0,
        "focal_arm_peak_any": work["focal_arm_peak_count"] > 0,
        "opposite_arm_peak_any": work["opposite_arm_peak_count"] > 0,
        "shifted_focal_overlap_any": work["shifted_focal_overlap"] > 0,
        "shifted_partner_overlap_any": work["shifted_partner_overlap"] > 0,
        "shifted_duplex_peak_any": work["shifted_duplex_peak_count"] > 0,
    }
    for window in WINDOWS:
        binary_sources[f"focal_near_w{window}_any"] = (
            work[f"focal_count_w{window}"] > 0
        )
        binary_sources[f"partner_near_w{window}_any"] = (
            work[f"partner_count_w{window}"] > 0
        )
        binary_sources[f"shifted_focal_near_w{window}_any"] = (
            work[f"shifted_focal_count_w{window}"] > 0
        )
    for name, values in binary_sources.items():
        work[name] = values.astype(float)
    work["valid_focal_nearest_distance"] = work["focal_nearest_distance"].where(
        work["focal_nearest_distance"] >= 0
    )
    named_aggregations: dict[str, tuple[str, str]] = {
        "eclip_track_count": ("file_accession", "nunique"),
        "eclip_cell_line_count": ("cell_line", "nunique"),
        "eclip_focal_overlap_any": ("focal_overlap_any", "max"),
        "eclip_focal_overlap_track_fraction": ("focal_overlap_any", "mean"),
        "eclip_partner_overlap_any": ("partner_overlap_any", "max"),
        "eclip_partner_overlap_track_fraction": ("partner_overlap_any", "mean"),
        "eclip_duplex_peak_any": ("duplex_peak_any", "max"),
        "eclip_duplex_peak_track_fraction": ("duplex_peak_any", "mean"),
        "eclip_focal_arm_peak_any": ("focal_arm_peak_any", "max"),
        "eclip_opposite_arm_peak_any": ("opposite_arm_peak_any", "max"),
        "eclip_shifted_focal_overlap_any": ("shifted_focal_overlap_any", "max"),
        "eclip_shifted_partner_overlap_any": (
            "shifted_partner_overlap_any",
            "max",
        ),
        "eclip_shifted_duplex_peak_any": ("shifted_duplex_peak_any", "max"),
        "eclip_shifted_duplex_coverage_fraction_max": (
            "shifted_duplex_coverage_fraction",
            "max",
        ),
        "eclip_nearest_peak_distance": ("valid_focal_nearest_distance", "min"),
        "eclip_duplex_coverage_fraction_max": ("duplex_coverage_fraction", "max"),
        "eclip_duplex_coverage_fraction_mean": ("duplex_coverage_fraction", "mean"),
        "eclip_duplex_max_signal": ("duplex_max_signal", "max"),
        "eclip_focal_arm_coverage_fraction_max": (
            "focal_arm_coverage_fraction",
            "max",
        ),
        "eclip_opposite_arm_coverage_fraction_max": (
            "opposite_arm_coverage_fraction",
            "max",
        ),
    }
    for window in WINDOWS:
        named_aggregations.update(
            {
                f"eclip_focal_near_w{window}_any": (
                    f"focal_near_w{window}_any",
                    "max",
                ),
                f"eclip_focal_near_w{window}_track_fraction": (
                    f"focal_near_w{window}_any",
                    "mean",
                ),
                f"eclip_focal_max_signal_w{window}": (
                    f"focal_max_signal_w{window}",
                    "max",
                ),
                f"eclip_partner_near_w{window}_any": (
                    f"partner_near_w{window}_any",
                    "max",
                ),
                f"eclip_shifted_focal_near_w{window}_any": (
                    f"shifted_focal_near_w{window}_any",
                    "max",
                ),
            }
        )
    return work.groupby(keys, as_index=False).agg(**named_aggregations)


def complete_panel(
    aggregated: pd.DataFrame, site_keys: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    genes = panel[["gene", "role", "prior_set", "prior_total"]].drop_duplicates()
    grid = site_keys.assign(_join=1).merge(genes.assign(_join=1), on="_join").drop(
        columns="_join"
    )
    result = grid.merge(
        aggregated, on=["pair_local_key", "gene"], how="left", validate="one_to_one"
    )
    result["eclip_assay_available"] = result["eclip_track_count"].notna().astype(int)
    distance = result["eclip_nearest_peak_distance"]
    result["eclip_nearest_peak_log1p"] = np.log1p(distance)
    eclip_columns = [
        column
        for column in result.columns
        if column.startswith("eclip_")
        and column not in {"eclip_nearest_peak_distance", "eclip_nearest_peak_log1p"}
    ]
    result[eclip_columns] = result[eclip_columns].fillna(0)
    result["negative_peak_call_is_uncertain_in_repeats"] = True
    return result


def add_expression(
    site_gene: pd.DataFrame, examples: pd.DataFrame, expression: pd.DataFrame
) -> pd.DataFrame:
    rows = examples.merge(site_gene, on="pair_local_key", how="left", validate="many_to_many")
    expression_columns = [
        "context",
        "gene",
        "tpm",
        "log1p_tpm",
        "expression_percentile_across_gtex",
        "tpm_over_adar",
        "tpm_over_adarb1",
    ]
    rows = rows.merge(
        expression[expression_columns],
        on=["context", "gene"],
        how="left",
        validate="many_to_one",
    )
    rows["expression_applicable"] = rows["context"] != "Combined"
    if rows.loc[rows["expression_applicable"], "tpm"].isna().any():
        raise RuntimeError("missing expression for an individual tissue context")
    if rows.loc[~rows["expression_applicable"], "tpm"].notna().any():
        raise RuntimeError("Combined was assigned a fabricated tissue-expression value")
    binding_columns = [
        "eclip_focal_overlap_any",
        "eclip_focal_near_w25_any",
        "eclip_focal_near_w100_any",
        "eclip_partner_overlap_any",
        "eclip_duplex_peak_any",
        "eclip_duplex_coverage_fraction_max",
    ]
    for column in binding_columns:
        rows[f"{column}_x_log1p_tpm"] = rows[column] * rows["log1p_tpm"]
    return rows


def wide_table(context_long: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["example_id", "context", "pair_id", "pair_local_key"]
    feature_columns = [
        column
        for column in context_long.columns
        if column.startswith("eclip_")
        or column
        in {
            "tpm",
            "log1p_tpm",
            "expression_percentile_across_gtex",
            "tpm_over_adar",
            "tpm_over_adarb1",
        }
    ]
    feature_columns = [
        column
        for column in feature_columns
        if column
        not in {
            "eclip_nearest_peak_distance",
        }
    ]
    indexed = context_long.set_index([*identifiers, "gene"])[feature_columns]
    wide = indexed.unstack("gene")
    wide.columns = [f"rbp__{gene}__{feature}" for feature, gene in wide.columns]
    return wide.reset_index()


def run(
    site_master_path: Path,
    track_features_path: Path,
    panel_path: Path,
    expression_path: Path,
    output_dir: Path,
) -> dict:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("context feature input contains a sealed split")
    examples = master[["example_id", "context", "pair_id", "pair_local_key"]].copy()
    site_keys = examples[["pair_local_key"]].drop_duplicates()
    tracks = pd.read_parquet(track_features_path)
    panel = pd.read_csv(panel_path)
    expression = pd.read_csv(expression_path)
    aggregated = aggregate_tracks(tracks)
    site_gene = complete_panel(aggregated, site_keys, panel)
    context_long = add_expression(site_gene, examples, expression)
    wide = wide_table(context_long)
    if len(wide) != len(examples):
        raise RuntimeError("context feature table does not have one row per development example")
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated.to_parquet(output_dir / "eclip_site_gene.parquet", index=False)
    context_long.to_parquet(output_dir / "rbp_context_long.parquet", index=False)
    wide.to_parquet(output_dir / "rbp_context_wide.parquet", index=False)
    summary = {
        "scope": "development train+valid only",
        "example_count": len(examples),
        "site_count": int(site_keys["pair_local_key"].nunique()),
        "panel_gene_count": int(panel["gene"].nunique()),
        "genes_with_eclip": sorted(aggregated["gene"].unique()),
        "wide_feature_count": len(wide.columns) - 4,
        "combined_expression": "not applicable; values remain missing",
        "interaction_definition": "prespecified binding summaries multiplied by log1p GTEx v8 TPM",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument(
        "--track-features",
        type=Path,
        default=Path("data/derived/rbp/eclip_track_features.parquet"),
    )
    parser.add_argument(
        "--panel", type=Path, default=Path("context/rbp/candidate_panel.csv")
    )
    parser.add_argument(
        "--expression",
        type=Path,
        default=Path("data/external/gtex_expression/gtex_v8_context_expression.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/derived/features")
    )
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    summary = run(
        resolve(args.site_master),
        resolve(args.track_features),
        resolve(args.panel),
        resolve(args.expression),
        resolve(args.output_dir),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
