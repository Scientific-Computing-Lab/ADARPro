"""Model paired cross-tissue EditingIndex differences beyond tissue-pair means."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adarpro.models.switch_direction import (
    NEGATIVE_CONTROLS,
    SUPPRESSORS,
    TISSUES,
    binding_table,
)
from adarpro.quarantine import sha256_file
from adarpro.registry import register

ALPHAS = (0.1, 1.0, 10.0, 100.0)


def construct_differences(
    observations: pd.DataFrame,
    expression: pd.DataFrame,
    binding: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    common = set.intersection(*(set(table.index) for table in binding.values()))
    observations = observations[observations["pair_local_key"].isin(common)]
    values = observations.pivot(
        index="pair_local_key", columns="context", values="editing_index"
    )
    metadata = observations.drop_duplicates("pair_local_key").set_index("pair_local_key")
    expression_lookup = expression.set_index(["context", "gene"])["log1p_tpm"]
    genes = sorted(expression["gene"].unique())
    blocks = []
    for tissue_a, tissue_b in combinations(TISSUES, 2):
        available = values[[tissue_a, tissue_b]].dropna()
        block = pd.DataFrame(index=available.index)
        block["pair_local_key"] = available.index
        block["pair_id"] = metadata.loc[available.index, "pair_id"].astype(int).to_numpy()
        block["development_fold"] = (
            metadata.loc[available.index, "development_fold"].astype(int).to_numpy()
        )
        block["tissue_a"] = tissue_a
        block["tissue_b"] = tissue_b
        block["tissue_pair"] = f"{tissue_a}|{tissue_b}"
        block["editing_difference"] = (
            available[tissue_a] - available[tissue_b]
        ).to_numpy()
        expression_delta = {
            gene: float(expression_lookup[tissue_a, gene] - expression_lookup[tissue_b, gene])
            for gene in genes
        }
        binding_features = {}
        for modality, table in binding.items():
            aligned = table.loc[available.index]
            for gene in aligned.columns:
                potential = aligned[gene].to_numpy(dtype=float)
                binding_features[f"binding__{modality}__{gene}"] = potential
                binding_features[f"interaction__{modality}__{gene}"] = (
                    potential * expression_delta[gene]
                )
        blocks.append(
            pd.concat(
                [
                    block.reset_index(drop=True),
                    pd.DataFrame(binding_features),
                ],
                axis=1,
            )
        )
    output = pd.concat(blocks, ignore_index=True)
    per_site = output.groupby("pair_local_key").size()
    output["site_weight"] = output["pair_local_key"].map(1 / per_site)
    pair_dummies = pd.get_dummies(output["tissue_pair"], prefix="pair", dtype=float)
    return pd.concat([output, pair_dummies], axis=1)


def select_features(table: pd.DataFrame) -> dict[str, list[str]]:
    tissue_pair = sorted(
        column for column in table if column.startswith("pair_") and "|" in column
    )
    interactions = sorted(
        column for column in table if column.startswith("interaction__")
    )
    binding = sorted(column for column in table if column.startswith("binding__"))

    def modality(prefix: str) -> list[str]:
        return [column for column in interactions if column.startswith(prefix)]

    def genes(wanted: set[str]) -> list[str]:
        return [column for column in interactions if column.rsplit("__", 1)[-1] in wanted]

    return {
        "tissue_pair": tissue_pair,
        "binding_without_expression": tissue_pair + binding,
        "tissue_pair_plus_eclip": tissue_pair + modality("interaction__eclip"),
        "tissue_pair_plus_prismnet": tissue_pair + modality("interaction__prism"),
        "tissue_pair_plus_rbpnet": tissue_pair + modality("interaction__rbpnet"),
        "tissue_pair_plus_all_rbp": tissue_pair + interactions,
        "tissue_pair_plus_suppressors": tissue_pair + genes(SUPPRESSORS),
        "tissue_pair_plus_negative_controls": tissue_pair
        + genes(NEGATIVE_CONTROLS),
    }


def model(alpha: float):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=alpha, solver="lsqr", tol=1e-5),
    )


def weighted_metrics(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame["editing_difference"]
    prediction = frame["predicted_difference"]
    weight = frame["site_weight"]
    truth_mean = np.average(truth, weights=weight)
    prediction_mean = np.average(prediction, weights=weight)
    covariance = np.average(
        (truth - truth_mean) * (prediction - prediction_mean), weights=weight
    )
    truth_variance = np.average((truth - truth_mean) ** 2, weights=weight)
    prediction_variance = np.average(
        (prediction - prediction_mean) ** 2, weights=weight
    )
    correlation = covariance / np.sqrt(truth_variance * prediction_variance)
    return {
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction, sample_weight=weight))),
        "mae": float(mean_absolute_error(truth, prediction, sample_weight=weight)),
        "r2": float(r2_score(truth, prediction, sample_weight=weight)),
        "pearson": float(correlation),
    }


def nested_predictions(
    table: pd.DataFrame,
    columns: list[str],
    name: str,
    seed: int,
    random_dimension: int = 0,
) -> tuple[pd.DataFrame, list[dict]]:
    output, selections = [], []
    random_values = None
    if random_dimension:
        tissue_columns = sorted(
            column for column in table if column.startswith("pair_") and "|" in column
        )
        random_values = np.column_stack(
            [
                table[tissue_columns].to_numpy(),
                np.random.default_rng(seed).normal(
                    size=(len(table), random_dimension)
                ),
            ]
        )
        columns = [*tissue_columns, *[f"random_{i}" for i in range(random_dimension)]]
    for outer_fold in range(5):
        inner_fold = (outer_fold + 1) % 5
        train = ~table["development_fold"].isin([outer_fold, inner_fold])
        inner = table["development_fold"] == inner_fold
        outer = table["development_fold"] == outer_fold
        if random_values is None:
            x_train, x_inner, x_outer = (
                table.loc[train, columns],
                table.loc[inner, columns],
                table.loc[outer, columns],
            )
        else:
            x_train, x_inner, x_outer = (
                random_values[train.to_numpy()],
                random_values[inner.to_numpy()],
                random_values[outer.to_numpy()],
            )
        best = None
        for alpha in ALPHAS:
            fitted = model(alpha)
            fitted.fit(
                x_train,
                table.loc[train, "editing_difference"],
                ridge__sample_weight=table.loc[train, "site_weight"],
            )
            prediction = fitted.predict(x_inner)
            rmse = np.sqrt(
                mean_squared_error(
                    table.loc[inner, "editing_difference"],
                    prediction,
                    sample_weight=table.loc[inner, "site_weight"],
                )
            )
            candidate = (-rmse, -alpha, fitted)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        negative_rmse, negative_alpha, fitted = best
        frame = table.loc[
            outer,
            [
                "pair_local_key",
                "pair_id",
                "development_fold",
                "tissue_a",
                "tissue_b",
                "editing_difference",
                "site_weight",
            ],
        ].copy()
        frame["model"] = name
        frame["predicted_difference"] = fitted.predict(x_outer)
        output.append(frame)
        selections.append(
            {
                "model": name,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "alpha": -negative_alpha,
                "inner_rmse": -negative_rmse,
                "feature_count": len(columns),
            }
        )
    return pd.concat(output, ignore_index=True), selections


def pair_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    keys = ["pair_local_key", "tissue_a", "tissue_b"]
    merged = candidate.merge(
        baseline[[*keys, "predicted_difference"]],
        on=keys,
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    def components(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
        weight = frame["site_weight"].to_numpy()
        truth = frame["editing_difference"].to_numpy()
        prediction = frame[prediction_column].to_numpy()
        values = pd.DataFrame(
            {
                "pair_id": frame["pair_id"].to_numpy(),
                "weight": weight,
                "y": weight * truth,
                "prediction": weight * prediction,
                "y2": weight * truth**2,
                "prediction2": weight * prediction**2,
                "cross": weight * truth * prediction,
                "squared_error": weight * (truth - prediction) ** 2,
                "absolute_error": weight * np.abs(truth - prediction),
            }
        )
        return values.groupby("pair_id").sum()

    def component_metrics(sums: np.ndarray, columns: list[str]) -> dict[str, float]:
        values = dict(zip(columns, sums))
        weight = values["weight"]
        y_mean = values["y"] / weight
        prediction_mean = values["prediction"] / weight
        y_variance_sum = values["y2"] - weight * y_mean**2
        prediction_variance_sum = (
            values["prediction2"] - weight * prediction_mean**2
        )
        covariance_sum = values["cross"] - weight * y_mean * prediction_mean
        return {
            "rmse": float(np.sqrt(values["squared_error"] / weight)),
            "mae": float(values["absolute_error"] / weight),
            "r2": float(1 - values["squared_error"] / y_variance_sum),
            "pearson": float(
                covariance_sum
                / np.sqrt(y_variance_sum * prediction_variance_sum)
            ),
        }

    candidate_components = components(merged, "predicted_difference")
    baseline_components = components(merged, "predicted_difference_baseline")
    pairs = candidate_components.index.to_numpy()
    columns = candidate_components.columns.tolist()
    baseline_components = baseline_components.loc[pairs, columns]
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        counts = np.bincount(
            rng.integers(0, len(pairs), size=len(pairs)), minlength=len(pairs)
        )
        candidate_metric = component_metrics(
            counts @ candidate_components.to_numpy(), columns
        )
        baseline_metric = component_metrics(
            counts @ baseline_components.to_numpy(), columns
        )
        for metric in candidate_metric:
            rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "difference": candidate_metric[metric] - baseline_metric[metric],
                }
            )
    return pd.DataFrame(rows)


def run(
    observations_path: Path,
    expression_path: Path,
    eclip_path: Path,
    prismnet_path: Path,
    rbpnet_path: Path,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    observations = pd.read_parquet(observations_path)
    expression = pd.read_csv(expression_path)
    binding = {
        "eclip25": binding_table(eclip_path, "eclip_focal_near_w25_any"),
        "eclip_duplex": binding_table(eclip_path, "eclip_duplex_peak_any"),
        "prism25": binding_table(prismnet_path, "prismnet_focal_max_score_w25"),
        "prism_duplex": binding_table(prismnet_path, "prismnet_duplex_peak_any"),
        "rbpnet25": binding_table(rbpnet_path, "target_focal_mean_w25"),
        "rbpnet_duplex": binding_table(rbpnet_path, "target_duplex_max"),
    }
    differences = construct_differences(observations, expression, binding)
    sets = select_features(differences)
    prediction_tables, selections = [], []
    for name, columns in sets.items():
        prediction, selection = nested_predictions(differences, columns, name, seed)
        prediction_tables.append(prediction)
        selections.extend(selection)
    random_dimension = len(sets["tissue_pair_plus_all_rbp"]) - len(
        sets["tissue_pair"]
    )
    prediction, selection = nested_predictions(
        differences,
        [],
        "tissue_pair_plus_matched_random",
        seed + 99,
        random_dimension=random_dimension,
    )
    prediction_tables.append(prediction)
    selections.extend(selection)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    metric_table = pd.DataFrame(
        [
            {"model": name, **weighted_metrics(group)}
            for name, group in predictions.groupby("model")
        ]
    ).sort_values("rmse")
    baseline = predictions[predictions["model"] == "tissue_pair"]
    bootstraps = []
    for name, group in predictions.groupby("model"):
        if name == "tissue_pair":
            continue
        bootstrap = pair_bootstrap(group, baseline, bootstrap_replicates, seed + 1000)
        bootstrap["model"] = name
        bootstraps.append(bootstrap)
    bootstrap = pd.concat(bootstraps, ignore_index=True)
    ci = (
        bootstrap.groupby(["model", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    differences.to_parquet(output_dir / "editing_differences.parquet", index=False)
    predictions.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(output_dir / "fold_selections.csv", index=False)
    metric_table.to_csv(output_dir / "metrics.csv", index=False)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    ci.to_csv(output_dir / "pair_bootstrap_ci.csv", index=False)
    result = {
        "scope": "continuous EditingIndex differences on individual tissues; Combined excluded",
        "sites": int(differences["pair_local_key"].nunique()),
        "pairs": int(differences["pair_id"].nunique()),
        "paired_observations": len(differences),
        "models": sorted(predictions["model"].unique()),
        "bootstrap_replicates": bootstrap_replicates,
        "canonical_test_access": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    repository = observations_path.parents[3]
    register(
        repository,
        {
            "run_id": f"continuous_tissue_differences_seed{seed}",
            "experiment_type": "paired_continuous_editing_differences",
            "context": "four_individual_tissues",
            "input_hashes": {
                "observations": sha256_file(observations_path),
                "eclip": sha256_file(eclip_path),
                "prismnet": sha256_file(prismnet_path),
                "rbpnet": sha256_file(rbpnet_path),
            },
            "fold_definition": "canonical five pair folds; all site/tissue pairs grouped by Alu pair",
            "seed": seed,
            "features": sorted(sets),
            "hyperparameters": {
                "model": "weighted Ridge regression",
                "alpha_grid": list(ALPHAS),
                "selection": "inner-fold RMSE",
                "bootstrap_replicates": bootstrap_replicates,
            },
            "artifact": str(output_dir / "oof_predictions.parquet"),
            "metrics": metric_table.to_dict(orient="records"),
            "canonical_test_access": False,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path(
            "data/derived/cross_tissue/development_raw_individual_tissue_sites.parquet"
        ),
    )
    parser.add_argument(
        "--expression",
        type=Path,
        default=Path("data/external/gtex_expression/gtex_v8_context_expression.csv"),
    )
    parser.add_argument(
        "--eclip",
        type=Path,
        default=Path("data/derived/features/eclip_site_gene.parquet"),
    )
    parser.add_argument(
        "--prismnet",
        type=Path,
        default=Path("data/derived/rbp/prismnet/prismnet_site_gene.parquet"),
    )
    parser.add_argument(
        "--rbpnet",
        type=Path,
        default=Path("data/derived/rbp/rbpnet/rbpnet_site_features.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/continuous_tissue_differences"),
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
                resolve(args.observations),
                resolve(args.expression),
                resolve(args.eclip),
                resolve(args.prismnet),
                resolve(args.rbpnet),
                resolve(args.output_dir),
                args.bootstrap_replicates,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
