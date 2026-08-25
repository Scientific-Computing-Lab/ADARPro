"""Recover foldwise standardized coefficients for the H3K36me3 residual model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adarpro.models.residual import make_model


def extract_coefficients(
    site_master_path: Path,
    chromatin_path: Path,
    selections_path: Path,
    context: str,
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    master = pd.read_parquet(
        site_master_path,
        columns=[
            "example_id",
            "context",
            "pair_id",
            "label",
            "development_fold",
            "source_split",
            "baseline_oof_logit",
        ],
    )
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("coefficient extraction received a sealed split")
    annotations = pd.read_parquet(chromatin_path)
    table = master[master["context"] == context].merge(
        annotations[annotations["context"] == context].drop(columns="context"),
        on=["example_id", "pair_id"],
        validate="one_to_one",
    )
    if model_name != "baseline_plus_h3k36me3_pair":
        raise ValueError(f"unsupported coefficient model: {model_name}")
    features = ["baseline_oof_logit"] + sorted(
        column
        for column in table
        if column.startswith("chromatin__h3k36me3__") and "__pair__" in column
    )
    selections = pd.read_csv(selections_path)
    selections = selections[selections["model"] == model_name].set_index("outer_fold")
    rows = []
    for outer_fold in range(5):
        inner_fold = (outer_fold + 1) % 5
        train = ~table["development_fold"].isin([outer_fold, inner_fold])
        model = make_model(float(selections.loc[outer_fold, "C"]), seed + outer_fold)
        model.fit(table.loc[train, features], table.loc[train, "label"])
        coefficients = model.named_steps["logisticregression"].coef_[0]
        for feature, coefficient in zip(features, coefficients):
            rows.append(
                {
                    "context": context,
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                    "canonical_test_access": False,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-master", type=Path, default=Path("data/derived/site_master.parquet"))
    parser.add_argument("--chromatin", type=Path, default=Path("data/derived/features/chromatin_h3k36me3.parquet"))
    parser.add_argument("--selections", type=Path, default=Path("experiments/runs/chromatin_residual/MuscleSkeletal/fold_selections.csv"))
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument("--model", default="baseline_plus_h3k36me3_pair")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, default=Path("experiments/runs/chromatin_residual/MuscleSkeletal/fold_coefficients.csv"))
    args = parser.parse_args()
    table = extract_coefficients(
        args.site_master,
        args.chromatin,
        args.selections,
        args.context,
        args.model,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.groupby("feature")["standardized_coefficient"].median().to_string())


if __name__ == "__main__":
    main()
