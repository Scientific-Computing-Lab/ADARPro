"""Predict cross-tissue edit direction while holding biological site identity fixed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from adarpro.models.residual import CS, make_model
from adarpro.quarantine import sha256_file
from adarpro.registry import register

TISSUES = ("Artery", "Brain", "Liver", "MuscleSkeletal")
BINDING_SPECS = {
    "eclip25": ("eclip_focal_near_w25_any",),
    "eclip_duplex": ("eclip_duplex_peak_any",),
    "prism25": ("prismnet_focal_max_score_w25",),
    "prism_duplex": ("prismnet_duplex_peak_any",),
    "rbpnet25": ("target_focal_mean_w25",),
    "rbpnet_duplex": ("target_duplex_max",),
}
SUPPRESSORS = {"ILF3", "ILF2", "STRBP"}
NEGATIVE_CONTROLS = {"RBFOX2", "TIA1", "IGF2BP1"}
SHUFFLED_TISSUE = {
    "Artery": "Brain",
    "Brain": "Liver",
    "Liver": "MuscleSkeletal",
    "MuscleSkeletal": "Artery",
}


def binding_table(path: Path, feature: str) -> pd.DataFrame:
    table = pd.read_parquet(path, columns=["pair_local_key", "gene", feature])
    if table.duplicated(["pair_local_key", "gene"]).any():
        raise RuntimeError(f"non-unique site/gene binding feature in {path}")
    return table.pivot(index="pair_local_key", columns="gene", values=feature)


def randomized_order(site_id: str, edited: str, nonedited: str) -> tuple[str, str, int]:
    digest = hashlib.sha256(f"{site_id}|{edited}|{nonedited}".encode()).digest()
    return (edited, nonedited, 1) if digest[0] % 2 == 0 else (nonedited, edited, 0)


def construct_comparisons(
    matched: pd.DataFrame,
    expression: pd.DataFrame,
    binding: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    expression_lookup = expression.set_index(["context", "gene"])["log1p_tpm"]
    genes = sorted(expression["gene"].unique())
    rows = []
    switches = matched[matched["switch_category"] == "strict_edited_nonedited_switch"]
    common_keys = set.intersection(
        set(switches["pair_local_key"]),
        *(set(table.index) for table in binding.values()),
    )
    switches = switches[switches["pair_local_key"].isin(common_keys)]
    for site in switches.itertuples(index=False):
        edited = [
            tissue for tissue in TISSUES if getattr(site, f"{tissue}_state") == "edited"
        ]
        nonedited = [
            tissue
            for tissue in TISSUES
            if getattr(site, f"{tissue}_state") == "nonedited"
        ]
        for edited_tissue in edited:
            for nonedited_tissue in nonedited:
                tissue_a, tissue_b, label = randomized_order(
                    site.site_id, edited_tissue, nonedited_tissue
                )
                row = {
                    "site_id": site.site_id,
                    "pair_local_key": site.pair_local_key,
                    "pair_id": int(site.pair_id),
                    "development_fold": int(site.development_fold),
                    "tissue_a": tissue_a,
                    "tissue_b": tissue_b,
                    "label_a_edited": label,
                }
                for tissue in TISSUES:
                    row[f"tissue_delta__{tissue}"] = int(tissue_a == tissue) - int(
                        tissue_b == tissue
                    )
                expression_delta = {
                    gene: float(expression_lookup[tissue_a, gene] - expression_lookup[tissue_b, gene])
                    for gene in genes
                }
                shuffled_delta = {
                    gene: float(
                        expression_lookup[SHUFFLED_TISSUE[tissue_a], gene]
                        - expression_lookup[SHUFFLED_TISSUE[tissue_b], gene]
                    )
                    for gene in genes
                }
                for gene in genes:
                    row[f"expression_delta__{gene}"] = expression_delta[gene]
                for modality, table in binding.items():
                    modality_genes = sorted(table.columns)
                    permutation = {
                        gene: modality_genes[(index + 1) % len(modality_genes)]
                        for index, gene in enumerate(modality_genes)
                    }
                    for gene in modality_genes:
                        potential = float(table.at[site.pair_local_key, gene])
                        row[f"binding__{modality}__{gene}"] = potential
                        row[f"interaction__{modality}__{gene}"] = (
                            potential * expression_delta[gene]
                        )
                        row[f"shuffled_tissue__{modality}__{gene}"] = (
                            potential * shuffled_delta[gene]
                        )
                        permuted_potential = float(
                            table.at[site.pair_local_key, permutation[gene]]
                        )
                        row[f"permuted_identity__{modality}__{gene}"] = (
                            permuted_potential * expression_delta[gene]
                        )
                rows.append(row)
    output = pd.DataFrame(rows)
    comparisons_per_site = output.groupby("site_id").size()
    output["site_weight"] = output["site_id"].map(1 / comparisons_per_site)
    return output


def feature_sets(table: pd.DataFrame) -> dict[str, list[str]]:
    tissue = sorted(column for column in table if column.startswith("tissue_delta__"))
    expression = sorted(
        column for column in table if column.startswith("expression_delta__")
    )
    binding = sorted(column for column in table if column.startswith("binding__"))
    eclip = sorted(
        column for column in table if column.startswith("interaction__eclip")
    )
    prism = sorted(
        column for column in table if column.startswith("interaction__prism")
    )
    rbpnet = sorted(
        column for column in table if column.startswith("interaction__rbpnet")
    )
    interactions = sorted(
        column for column in table if column.startswith("interaction__")
    )
    shuffled = sorted(
        column for column in table if column.startswith("shuffled_tissue__")
    )
    permuted = sorted(
        column for column in table if column.startswith("permuted_identity__")
    )

    def genes(columns: list[str], wanted: set[str]) -> list[str]:
        return [column for column in columns if column.rsplit("__", 1)[-1] in wanted]

    return {
        "tissue_id": tissue,
        "expression_only": expression,
        "adar_expression": [
            column
            for column in expression
            if column.endswith(("__ADAR", "__ADARB1"))
        ],
        "binding_without_expression": binding,
        "tissue_plus_eclip_interaction": tissue + eclip,
        "tissue_plus_prismnet_interaction": tissue + prism,
        "tissue_plus_rbpnet_interaction": tissue + rbpnet,
        "tissue_plus_all_interactions": tissue + interactions,
        "tissue_plus_suppressor_interactions": tissue
        + genes(interactions, SUPPRESSORS),
        "tissue_plus_negative_control_interactions": tissue
        + genes(interactions, NEGATIVE_CONTROLS),
        "tissue_plus_shuffled_expression": tissue + shuffled,
        "tissue_plus_permuted_rbp_identity": tissue + permuted,
    }


def nested_predictions(
    table: pd.DataFrame,
    columns: list[str],
    name: str,
    seed: int,
    random_dimension: int = 0,
) -> tuple[pd.DataFrame, list[dict], list[pd.DataFrame]]:
    predictions, selections, coefficients = [], [], []
    random_values = None
    if random_dimension:
        tissue_columns = sorted(
            column for column in table if column.startswith("tissue_delta__")
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
        y_train = table.loc[train, "label_a_edited"].to_numpy()
        best = None
        for c_value in CS:
            model = make_model(c_value, seed + outer_fold)
            model.fit(
                x_train,
                y_train,
                logisticregression__sample_weight=table.loc[
                    train, "site_weight"
                ].to_numpy(),
            )
            probability = model.predict_proba(x_inner)[:, 1]
            score = roc_auc_score(table.loc[inner, "label_a_edited"], probability)
            candidate = (score, -c_value, model)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        score, negative_c, model = best
        probability = model.predict_proba(x_outer)[:, 1]
        frame = table.loc[
            outer,
            [
                "site_id",
                "pair_local_key",
                "pair_id",
                "development_fold",
                "tissue_a",
                "tissue_b",
                "label_a_edited",
                "site_weight",
            ],
        ].copy()
        frame["model"] = name
        frame["probability_a_edited"] = probability
        frame["prediction_a_edited"] = (probability >= 0.5).astype(int)
        predictions.append(frame)
        selections.append(
            {
                "model": name,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "C": -negative_c,
                "inner_auroc": score,
                "feature_count": len(columns),
            }
        )
        values = model.named_steps["logisticregression"].coef_[0]
        coefficients.append(
            pd.DataFrame(
                {
                    "model": name,
                    "outer_fold": outer_fold,
                    "feature": columns,
                    "standardized_coefficient": values,
                }
            )
        )
    return pd.concat(predictions, ignore_index=True), selections, coefficients


def weighted_metrics(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["label_a_edited"]
    probability = frame["probability_a_edited"]
    prediction = frame["prediction_a_edited"]
    weights = frame["site_weight"]
    return {
        "auroc": float(roc_auc_score(labels, probability, sample_weight=weights)),
        "auprc": float(average_precision_score(labels, probability, sample_weight=weights)),
        "accuracy": float(accuracy_score(labels, prediction, sample_weight=weights)),
    }


def pair_bootstrap(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    merged = predictions.merge(
        baseline[
            [
                "site_id",
                "tissue_a",
                "tissue_b",
                "probability_a_edited",
                "prediction_a_edited",
            ]
        ],
        on=["site_id", "tissue_a", "tissue_b"],
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    pairs = merged["pair_id"].unique()
    indices = {
        pair: np.flatnonzero(merged["pair_id"].to_numpy() == pair) for pair in pairs
    }
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(pairs, len(pairs), replace=True)
        sample = merged.iloc[np.concatenate([indices[pair] for pair in sampled])].copy()
        candidate = weighted_metrics(sample)
        base = sample.copy()
        base["probability_a_edited"] = sample[
            "probability_a_edited_baseline"
        ]
        base["prediction_a_edited"] = sample["prediction_a_edited_baseline"]
        baseline_metric = weighted_metrics(base)
        for metric in candidate:
            rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "difference": candidate[metric] - baseline_metric[metric],
                }
            )
    return pd.DataFrame(rows)


def run(
    matched_path: Path,
    expression_path: Path,
    eclip_path: Path,
    prismnet_path: Path,
    rbpnet_path: Path,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    matched = pd.read_parquet(matched_path)
    expression = pd.read_csv(expression_path)
    binding = {
        "eclip25": binding_table(eclip_path, "eclip_focal_near_w25_any"),
        "eclip_duplex": binding_table(eclip_path, "eclip_duplex_peak_any"),
        "prism25": binding_table(prismnet_path, "prismnet_focal_max_score_w25"),
        "prism_duplex": binding_table(prismnet_path, "prismnet_duplex_peak_any"),
        "rbpnet25": binding_table(rbpnet_path, "target_focal_mean_w25"),
        "rbpnet_duplex": binding_table(rbpnet_path, "target_duplex_max"),
    }
    comparisons = construct_comparisons(matched, expression, binding)
    sets = feature_sets(comparisons)
    prediction_tables, selections, coefficient_tables = [], [], []
    for name, columns in sets.items():
        prediction, selection, coefficients = nested_predictions(
            comparisons, columns, name, seed
        )
        prediction_tables.append(prediction)
        selections.extend(selection)
        coefficient_tables.extend(coefficients)
    interaction_dimension = len(sets["tissue_plus_all_interactions"]) - len(
        sets["tissue_id"]
    )
    random_prediction, random_selection, random_coefficients = nested_predictions(
        comparisons,
        [],
        "tissue_plus_matched_random",
        seed + 99,
        random_dimension=interaction_dimension,
    )
    prediction_tables.append(random_prediction)
    selections.extend(random_selection)
    coefficient_tables.extend(random_coefficients)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    coefficients = pd.concat(coefficient_tables, ignore_index=True)
    metrics = []
    for name, group in predictions.groupby("model"):
        metrics.append({"model": name, **weighted_metrics(group)})
    metric_table = pd.DataFrame(metrics).sort_values("auroc", ascending=False)
    baseline = predictions[predictions["model"] == "tissue_id"]
    bootstraps = []
    for name, group in predictions.groupby("model"):
        if name == "tissue_id":
            continue
        bootstrap = pair_bootstrap(
            group, baseline, bootstrap_replicates, seed + 1000
        )
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
    comparisons.to_parquet(output_dir / "switch_comparisons.parquet", index=False)
    predictions.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(output_dir / "fold_selections.csv", index=False)
    coefficients.to_csv(output_dir / "standardized_coefficients.csv", index=False)
    metric_table.to_csv(output_dir / "metrics.csv", index=False)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    ci.to_csv(output_dir / "pair_bootstrap_ci.csv", index=False)
    result = {
        "scope": "strict development-pair cross-tissue switches; Combined excluded",
        "sites": int(comparisons["site_id"].nunique()),
        "pairs": int(comparisons["pair_id"].nunique()),
        "tissue_comparisons": len(comparisons),
        "models": sorted(predictions["model"].unique()),
        "bootstrap_replicates": bootstrap_replicates,
        "canonical_test_access": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    repository = matched_path.parents[3]
    register(
        repository,
        {
            "run_id": f"cross_tissue_switch_seed{seed}",
            "experiment_type": "matched_switch_direction",
            "context": "four_individual_tissues",
            "input_hashes": {
                "matched_sites": sha256_file(matched_path),
                "eclip": sha256_file(eclip_path),
                "prismnet": sha256_file(prismnet_path),
                "rbpnet": sha256_file(rbpnet_path),
            },
            "fold_definition": "canonical five pair folds; site/tissue comparisons grouped by pair",
            "seed": seed,
            "features": sorted(sets),
            "hyperparameters": {
                "model": "weighted L2 logistic regression",
                "C_grid": list(CS),
                "selection": "inner-fold AUROC",
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
        "--matched",
        type=Path,
        default=Path("data/derived/cross_tissue/matched_sites.parquet"),
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
        default=Path("experiments/runs/cross_tissue_switch"),
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
                resolve(args.matched),
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
