"""Assemble predicted RBP binding and abundance into a development model matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

INTERACTION_FEATURES = (
    "target_focal_value",
    "target_focal_mean_w25",
    "target_focal_mass_w25",
    "target_partner_value",
    "target_partner_mean_w25",
    "target_duplex_max",
    "target_focal_arm_mass",
    "target_opposite_arm_mass",
)
PRISMNET_INTERACTION_FEATURES = (
    "prismnet_focal_overlap_any",
    "prismnet_focal_near_w25_any",
    "prismnet_focal_max_score_w25",
    "prismnet_partner_overlap_any",
    "prismnet_duplex_peak_any",
    "prismnet_duplex_coverage_fraction_max",
    "prismnet_duplex_max_score",
)


def build_long(
    examples: pd.DataFrame, predictions: pd.DataFrame, expression: pd.DataFrame
) -> pd.DataFrame:
    rows = examples.merge(
        predictions, on=["pair_id", "pair_local_key", "site_id"], validate="many_to_many"
    )
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
        raise RuntimeError("missing predicted-RBP expression for an individual context")
    if rows.loc[~rows["expression_applicable"], "tpm"].notna().any():
        raise RuntimeError("Combined received a tissue-expression vector")
    for feature in INTERACTION_FEATURES:
        rows[f"{feature}_x_log1p_tpm"] = rows[feature] * rows["log1p_tpm"]
    return rows


def build_wide(rows: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["example_id", "context", "pair_id", "pair_local_key"]
    excluded = {
        *identifiers,
        "site_id",
        "gene",
        "cell_line",
        "model_output_semantics",
        "expression_applicable",
    }
    features = [
        column
        for column in rows.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(rows[column])
    ]
    wide = rows.set_index([*identifiers, "gene"])[features].unstack("gene")
    wide.columns = [f"pred__{gene}__{feature}" for feature, gene in wide.columns]
    return wide.reset_index()


def build_prismnet_long(
    examples: pd.DataFrame, predictions: pd.DataFrame, expression: pd.DataFrame
) -> pd.DataFrame:
    rows = examples.merge(
        predictions, on="pair_local_key", how="left", validate="many_to_many"
    )
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
        raise RuntimeError("missing PrismNet-RBP expression in an individual context")
    if rows.loc[~rows["expression_applicable"], "tpm"].notna().any():
        raise RuntimeError("Combined received PrismNet tissue-expression values")
    for feature in PRISMNET_INTERACTION_FEATURES:
        rows[f"{feature}_x_log1p_tpm"] = rows[feature] * rows["log1p_tpm"]
    return rows


def build_prismnet_wide(rows: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["example_id", "context", "pair_id", "pair_local_key"]
    excluded = {*identifiers, "gene", "expression_applicable"}
    features = [
        column
        for column in rows.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(rows[column])
    ]
    wide = rows.set_index([*identifiers, "gene"])[features].unstack("gene")
    wide.columns = [f"prism__{gene}__{feature}" for feature, gene in wide.columns]
    return wide.reset_index()


def run(
    site_master_path: Path,
    predicted_path: Path,
    prismnet_path: Path,
    expression_path: Path,
    eclip_wide_path: Path,
    mappability_path: Path,
    output_dir: Path,
) -> dict:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("residual matrix input contains a sealed split")
    identifiers = [
        "example_id",
        "context",
        "pair_id",
        "pair_local_key",
        "site_id",
        "label",
        "development_fold",
        "source_split",
        "editing_index",
    ]
    intrinsic_metadata = ["local_position", "sequence", "structure"]
    oof_columns = [column for column in master if column.startswith("baseline_oof_")]
    examples = master[identifiers]
    predictions = pd.read_parquet(predicted_path)
    prismnet = pd.read_parquet(prismnet_path)
    expression = pd.read_csv(expression_path)
    long = build_long(examples, predictions, expression)
    predicted_wide = build_wide(long)
    prismnet_long = build_prismnet_long(examples, prismnet, expression)
    prismnet_wide = build_prismnet_wide(prismnet_long)
    eclip_wide = pd.read_parquet(eclip_wide_path)
    mappability = pd.read_parquet(mappability_path).drop(columns="site_id")
    matrix = master[[*identifiers, *intrinsic_metadata, *oof_columns]].merge(
        eclip_wide,
        on=["example_id", "context", "pair_id", "pair_local_key"],
        validate="one_to_one",
    )
    matrix = matrix.merge(
        predicted_wide,
        on=["example_id", "context", "pair_id", "pair_local_key"],
        validate="one_to_one",
    ).merge(
        prismnet_wide,
        on=["example_id", "context", "pair_id", "pair_local_key"],
        validate="one_to_one",
    ).merge(mappability, on="pair_local_key", validate="many_to_one")
    output_dir.mkdir(parents=True, exist_ok=True)
    long.to_parquet(output_dir / "predicted_rbp_context_long.parquet", index=False)
    predicted_wide.to_parquet(output_dir / "predicted_rbp_context_wide.parquet", index=False)
    prismnet_long.to_parquet(output_dir / "prismnet_context_long.parquet", index=False)
    prismnet_wide.to_parquet(output_dir / "prismnet_context_wide.parquet", index=False)
    matrix.to_parquet(output_dir / "residual_model_matrix.parquet", index=False)
    result = {
        "scope": "development train+valid only",
        "rows": len(matrix),
        "pairs": int(matrix["pair_id"].nunique()),
        "contexts": matrix.groupby("context").size().astype(int).to_dict(),
        "predicted_rbp_genes": sorted(predictions["gene"].unique()),
        "predicted_feature_count": len(predicted_wide.columns) - 4,
        "prismnet_genes": sorted(prismnet["gene"].unique()),
        "prismnet_feature_count": len(prismnet_wide.columns) - 4,
        "total_columns": len(matrix.columns),
        "oof_contexts_available": sorted(
            matrix.loc[matrix.get("baseline_oof_probability", pd.Series(index=matrix.index)).notna(), "context"].unique()
        ),
        "combined_expression": "not assigned",
    }
    (output_dir / "residual_model_matrix.summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument(
        "--predicted",
        type=Path,
        default=Path("data/derived/rbp/rbpnet/rbpnet_site_features.parquet"),
    )
    parser.add_argument(
        "--prismnet",
        type=Path,
        default=Path("data/derived/rbp/prismnet/prismnet_site_gene.parquet"),
    )
    parser.add_argument(
        "--expression",
        type=Path,
        default=Path("data/external/gtex_expression/gtex_v8_context_expression.csv"),
    )
    parser.add_argument(
        "--eclip-wide",
        type=Path,
        default=Path("data/derived/features/rbp_context_wide.parquet"),
    )
    parser.add_argument(
        "--mappability",
        type=Path,
        default=Path("data/derived/genomic_mapping/umap100.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/derived/features")
    )
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    result = run(
        resolve(args.site_master),
        resolve(args.predicted),
        resolve(args.prismnet),
        resolve(args.expression),
        resolve(args.eclip_wide),
        resolve(args.mappability),
        resolve(args.output_dir),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
