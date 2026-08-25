"""Describe where target-local models change pair-balanced OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.quarantine import sha256_file
from adarpro.registry import register

IDENTITY = ["example_id", "context", "pair_id", "pair_local_key", "label", "development_fold"]
PREDICTION_COLUMNS = IDENTITY + ["probability", "prediction"]


def load_predictions(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    tables = {
        name: pd.read_parquet(path).sort_values("example_id").reset_index(drop=True)
        for name, path in paths.items()
    }
    reference = tables["pair"]
    for name, table in tables.items():
        missing = set(PREDICTION_COLUMNS) - set(table)
        if missing:
            raise RuntimeError(f"{name} lacks prediction columns: {sorted(missing)}")
        if not reference[IDENTITY].equals(table[IDENTITY]):
            raise RuntimeError(f"prediction identity mismatch: {name}")
    if set(reference["development_fold"].unique()) != set(range(5)):
        raise RuntimeError("predictions do not contain five development folds")
    return tables


def transition_row(
    label: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_name: str,
    candidate_name: str,
) -> dict[str, float | int | str]:
    reference_fp = (label == 0) & (reference == 1)
    reference_tp = (label == 1) & (reference == 1)
    rescued = reference_fp & (candidate == 0)
    damaged = reference_tp & (candidate == 0)
    return {
        "reference": reference_name,
        "candidate": candidate_name,
        "reference_fp": int(reference_fp.sum()),
        "fp_rescued": int(rescued.sum()),
        "fp_rescue_rate": float(rescued.sum() / max(reference_fp.sum(), 1)),
        "new_fp": int(((label == 0) & (reference == 0) & (candidate == 1)).sum()),
        "reference_tp": int(reference_tp.sum()),
        "tp_damaged": int(damaged.sum()),
        "tp_damage_rate": float(damaged.sum() / max(reference_tp.sum(), 1)),
        "fn_recovered": int(
            ((label == 1) & (reference == 0) & (candidate == 1)).sum()
        ),
    }


def add_strata(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    partner_columns = [
        f"target_geometry__partner_base_{base}" for base in ("A", "C", "G", "T", "N")
    ]
    partner = result[partner_columns].to_numpy().argmax(axis=1)
    result["partner_base"] = np.asarray(["A", "C", "G", "T", "N"])[partner]
    result.loc[result["target_geometry__target_paired"] < 0.5, "partner_base"] = "unpaired"
    result["target_paired"] = np.where(
        result["target_geometry__target_paired"] >= 0.5, "paired", "unpaired"
    )
    result["stem_extent"] = pd.cut(
        result["target_geometry__stem_extent_total"],
        bins=[-np.inf, 0, 5, 10, 20, np.inf],
        labels=["0", "1-5", "6-10", "11-20", ">20"],
    ).astype(str)
    result["helix_intact_w10"] = pd.cut(
        result["target_geometry__w10_helix_intact_fraction"],
        bins=[-np.inf, 0.25, 0.5, 0.75, np.inf],
        labels=["<=0.25", "0.25-0.50", "0.50-0.75", ">0.75"],
    ).astype(str)
    return result


def loss_contrasts(table: pd.DataFrame, candidates: tuple[str, ...]) -> pd.DataFrame:
    epsilon = np.finfo(np.float64).eps
    label = table["label"].to_numpy(dtype=float)
    for model in ("pair", *candidates):
        probability = np.clip(
            table[f"{model}_probability"].to_numpy(dtype=float),
            epsilon,
            1 - epsilon,
        )
        table[f"{model}_log_loss"] = -(
            label * np.log(probability) + (1 - label) * np.log(1 - probability)
        )
        table[f"{model}_brier"] = (label - probability) ** 2
    return table


def stratum_summary(table: pd.DataFrame, candidates: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    strata = [("all", "all", np.ones(len(table), dtype=bool))]
    for column in ("target_paired", "partner_base", "stem_extent", "helix_intact_w10"):
        strata.extend(
            (column, str(value), table[column].eq(value).to_numpy())
            for value in sorted(table[column].dropna().unique())
        )
    for stratum, level, mask in strata:
        subset = table.loc[mask]
        if subset.empty:
            continue
        for candidate in candidates:
            rows.append(
                {
                    "stratum": stratum,
                    "level": level,
                    "candidate": candidate,
                    "rows": len(subset),
                    "pairs": int(subset["pair_id"].nunique()),
                    "positive_fraction": float(subset["label"].mean()),
                    "mean_log_loss_difference": float(
                        (subset[f"{candidate}_log_loss"] - subset["pair_log_loss"]).mean()
                    ),
                    "mean_brier_difference": float(
                        (subset[f"{candidate}_brier"] - subset["pair_brier"]).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def pair_bootstrap_all(
    table: pd.DataFrame,
    candidates: tuple[str, ...],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    pairs = table["pair_id"].drop_duplicates().to_numpy()
    groups = {
        pair: np.flatnonzero(table["pair_id"].to_numpy() == pair) for pair in pairs
    }
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        indices = np.concatenate([groups[pair] for pair in sampled])
        for candidate in candidates:
            for metric in ("log_loss", "brier"):
                difference = (
                    table.iloc[indices][f"{candidate}_{metric}"].to_numpy()
                    - table.iloc[indices][f"pair_{metric}"].to_numpy()
                )
                rows.append(
                    {
                        "replicate": replicate,
                        "candidate": candidate,
                        "metric": metric,
                        "difference": float(difference.mean()),
                    }
                )
    return pd.DataFrame(rows)


def run(
    repository: Path,
    site_master_path: Path,
    geometry_path: Path,
    prediction_paths: dict[str, Path],
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    if "pair" not in prediction_paths or len(prediction_paths) < 2:
        raise ValueError("pair and at least one candidate prediction are required")
    tables = load_predictions(prediction_paths)
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("interpretation table contains a sealed split")
    reference = tables["pair"]
    context = str(reference["context"].iloc[0])
    master = master[master["context"] == context].copy()
    selected_master_columns = [
        "example_id",
        "editing_index",
        "sequence",
        "structure",
        "local_position",
        "site_id",
        "chromosome",
        "genomic_position_1based",
        "strand",
    ]
    missing_master = set(selected_master_columns) - set(master)
    if missing_master:
        raise RuntimeError(f"site master lacks columns: {sorted(missing_master)}")
    table = reference[PREDICTION_COLUMNS].rename(
        columns={"probability": "pair_probability", "prediction": "pair_prediction"}
    )
    for name, candidate in tables.items():
        if name == "pair":
            continue
        table[f"{name}_probability"] = candidate["probability"]
        table[f"{name}_prediction"] = candidate["prediction"]
    table = table.merge(
        master[selected_master_columns], on="example_id", validate="one_to_one"
    )
    geometry = pd.read_parquet(geometry_path)
    geometry_columns = [column for column in geometry if column.startswith("target_geometry__")]
    table = table.merge(
        geometry[["example_id", *geometry_columns]],
        on="example_id",
        validate="one_to_one",
    )
    table = loss_contrasts(add_strata(table), tuple(name for name in tables if name != "pair"))

    labels = table["label"].to_numpy()
    transitions = []
    for candidate in (name for name in tables if name != "pair"):
        transitions.append(
            transition_row(
                labels,
                table["pair_prediction"].to_numpy(),
                table[f"{candidate}_prediction"].to_numpy(),
                "pair",
                candidate,
            )
        )
    if "baseline_oof_prediction" in reference:
        table["development_reference_probability"] = reference[
            "baseline_oof_probability"
        ]
        table["development_reference_prediction"] = reference[
            "baseline_oof_prediction"
        ]
        for candidate in (name for name in tables if name != "pair"):
            transitions.append(
                transition_row(
                    labels,
                    table["development_reference_prediction"].to_numpy(),
                    table[f"{candidate}_prediction"].to_numpy(),
                    "development_reference",
                    candidate,
                )
            )

    candidates = tuple(name for name in tables if name != "pair")
    strata = stratum_summary(table, candidates)
    bootstrap = pair_bootstrap_all(table, candidates, bootstrap_replicates, seed)
    confidence = (
        bootstrap.groupby(["candidate", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    case_rows = []
    for candidate in candidates:
        reference_fp = (table["label"] == 0) & (table["pair_prediction"] == 1)
        rescued = table[reference_fp & (table[f"{candidate}_prediction"] == 0)].copy()
        rescued["case_type"] = "pair_fp_rescued"
        damaged = table[
            (table["label"] == 1)
            & (table["pair_prediction"] == 1)
            & (table[f"{candidate}_prediction"] == 0)
        ].copy()
        damaged["case_type"] = "pair_tp_damaged"
        selected = pd.concat([rescued, damaged], ignore_index=True)
        selected["candidate"] = candidate
        selected["probability_change"] = (
            selected[f"{candidate}_probability"] - selected["pair_probability"]
        )
        case_rows.append(selected)
    cases = pd.concat(case_rows, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(transitions).to_csv(output_dir / "error_transitions.csv", index=False)
    strata.to_csv(output_dir / "geometry_strata.csv", index=False)
    bootstrap.to_parquet(output_dir / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(output_dir / "pair_bootstrap_ci.csv", index=False)
    cases.sort_values(["candidate", "case_type", "probability_change"]).to_parquet(
        output_dir / "case_studies.parquet", index=False
    )
    summary = {
        "scope": "canonical train+valid grouped OOF descriptive interpretation",
        "context": context,
        "rows": len(table),
        "pairs": int(table["pair_id"].nunique()),
        "candidates": list(candidates),
        "transitions": transitions,
        "canonical_test_access": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"target_local_interpretation_{context}_seed{seed}",
            "experiment_type": "development_oof_descriptive_interpretation",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "target_geometry": sha256_file(geometry_path),
                **{
                    f"{name}_predictions": sha256_file(path)
                    for name, path in prediction_paths.items()
                },
            },
            "fold_definition": "five global pair OOF folds; descriptive only",
            "seed": seed,
            "features": "prespecified target pairing, partner base, stem extent and local helix continuity strata",
            "hyperparameters": {"bootstrap_replicates": bootstrap_replicates},
            "artifact": str((output_dir / "summary.json").relative_to(repository)),
            "metrics": summary,
            "canonical_test_access": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--site-master", type=Path, default=Path("data/derived/site_master.parquet")
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=Path("data/derived/features/target_geometry_MuscleSkeletal.parquet"),
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="NAME=path/to/oof_predictions.parquet; include pair and candidates",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/target_local_interpretation/MuscleSkeletal"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    paths = {}
    for value in args.prediction:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or name in paths:
            parser.error(f"invalid --prediction: {value}")
        paths[name] = resolve(Path(raw_path))
    print(
        json.dumps(
            run(
                root,
                resolve(args.site_master),
                resolve(args.geometry),
                paths,
                resolve(args.output_dir),
                args.bootstrap_replicates,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
