"""Outer-isolated gate adding within-substrate editability rank to a GAT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from adarpro.features.substrate_profile import summarize_profile
from adarpro.models.baseline_oof import build_graphs, torch_load, upstream_modules
from adarpro.models.hierarchical_gate import fit_gate
from adarpro.models.multitask_oof import MultiTaskGAT, predict_binary
from adarpro.models.residual import bootstrap_differences, metrics, rescue_metrics
from adarpro.quarantine import sha256_file
from adarpro.registry import register


@torch.no_grad()
def predict_all_a(
    model: MultiTaskGAT,
    graphs: list,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Predict binary editability for graphs whose labels are placeholders."""

    model.eval()
    probabilities = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        binary_logit, _, _ = model(batch.to(device))
        probabilities.extend(torch.sigmoid(binary_logit).cpu().tolist())
    return np.asarray(probabilities, dtype=float)


def build_all_a_graphs(canonical: pd.DataFrame, baseline) -> tuple[list, pd.DataFrame]:
    """Build one graph for every adenosine of every development substrate."""

    pairs = canonical.drop_duplicates("pair_id")
    graphs, rows = [], []
    for row in pairs.itertuples(index=False):
        for local_position, base in enumerate(row.sequence):
            if base.upper() != "A":
                continue
            graph = baseline.create_rna_graph(
                row.sequence, row.structure, local_position, 0
            )
            if graph is None:
                raise RuntimeError(
                    f"unbuildable all-A graph: pair={row.pair_id} "
                    f"position={local_position}"
                )
            graphs.append(graph)
            rows.append(
                {"pair_id": row.pair_id, "local_position": local_position}
            )
    return graphs, pd.DataFrame(rows)


def relative_features(
    canonical: pd.DataFrame,
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, float]:
    """Calculate focal percentile/z-score and audit focal recomputation."""

    if len(metadata) != len(probabilities):
        raise RuntimeError("all-A metadata/prediction length mismatch")
    profiles = {}
    for pair_id, indices in metadata.groupby("pair_id", sort=False).groups.items():
        selected = np.asarray(list(indices), dtype=int)
        positions = metadata.loc[selected, "local_position"].to_numpy(dtype=int)
        order = np.argsort(positions)
        profiles[pair_id] = (
            positions[order],
            probabilities[selected][order],
        )
    rows = []
    maximum_difference = 0.0
    for row in canonical.itertuples(index=False):
        positions, values = profiles[row.pair_id]
        matches = np.flatnonzero(positions == int(row.local_position))
        if len(matches) != 1:
            raise RuntimeError(
                f"canonical focal A absent from profile: {row.example_id}"
            )
        focal_index = int(matches[0])
        focal_probability = float(values[focal_index])
        maximum_difference = max(
            maximum_difference,
            abs(focal_probability - float(row.expanded_probability)),
        )
        summary = summarize_profile(values, focal_index, threshold)
        rows.append(
            {
                "example_id": row.example_id,
                "relative_percentile": summary[
                    "substrate_profile__focal_percentile"
                ],
                "relative_zscore": summary["substrate_profile__focal_zscore"],
            }
        )
    return pd.DataFrame(rows), maximum_difference


def permute_within_pair(
    table: pd.DataFrame, columns: list[str], seed: int
) -> pd.DataFrame:
    """Break focal alignment while preserving each pair's feature multiset."""

    result = table.copy()
    rng = np.random.default_rng(seed)
    for indices in result.groupby("pair_id", sort=False).groups.values():
        selected = np.asarray(list(indices), dtype=int)
        permutation = rng.permutation(len(selected))
        result.loc[selected, columns] = result.loc[
            selected[permutation], columns
        ].to_numpy()
    return result


def run(
    repository: Path,
    site_master_path: Path,
    base_run_dir: Path,
    context: str,
    mode: str,
    output_dir: Path,
    batch_size: int,
    bootstrap_replicates: int,
    seed: int,
    device: torch.device,
) -> dict:
    master = pd.read_parquet(site_master_path)
    if set(master["source_split"].unique()) - {"train", "valid"}:
        raise RuntimeError("substrate-profile gate received a sealed split")
    canonical = (
        master[master["context"] == context].copy().reset_index(drop=True)
    )
    baseline = upstream_modules(repository)
    canonical_graphs = build_graphs(canonical, baseline)
    all_a_graphs, all_a_metadata = build_all_a_graphs(canonical, baseline)
    print(
        f"[substrate-profile gate] built {len(canonical_graphs)} canonical and "
        f"{len(all_a_graphs)} all-A graphs across "
        f"{canonical['pair_id'].nunique()} pairs",
        flush=True,
    )
    outputs, selections, checkpoint_hashes = [], [], {}
    maximum_outer_difference = 0.0
    maximum_focal_difference = 0.0
    relative_columns = ["relative_percentile", "relative_zscore"]
    for outer_fold in range(5):
        print(
            f"[substrate-profile gate] outer={outer_fold} predicting all sites",
            flush=True,
        )
        checkpoint = base_run_dir / f"outer_{outer_fold}" / "best.pth"
        checkpoint_hashes[f"outer_{outer_fold}"] = sha256_file(checkpoint)
        saved = torch_load(checkpoint, device)
        saved_mode = saved.get("config", {}).get("mode")
        if saved_mode != mode:
            raise RuntimeError(
                f"base checkpoint mode mismatch: expected {mode}, found {saved_mode}"
            )
        if "_context" in mode or mode.endswith("_local"):
            raise ValueError(
                "this first profile-gate implementation requires a graph-only "
                "MultiTaskGAT checkpoint"
            )
        model = MultiTaskGAT().to(device)
        model.load_state_dict(saved["model_state"])
        probability, labels = predict_binary(
            model,
            DataLoader(canonical_graphs, batch_size=batch_size, shuffle=False),
            device,
        )
        if not np.array_equal(labels, canonical["label"].to_numpy()):
            raise RuntimeError("base-model canonical prediction order mismatch")
        table = canonical[
            [
                "example_id",
                "context",
                "pair_id",
                "pair_local_key",
                "local_position",
                "label",
                "development_fold",
            ]
        ].copy()
        clipped = np.clip(probability, 1e-7, 1 - 1e-7)
        table["expanded_logit"] = np.log(clipped / (1 - clipped))
        table["expanded_probability"] = probability
        table["expanded_reference_prediction"] = (
            probability >= float(saved["best"]["threshold"])
        ).astype(int)
        saved_outer = pd.read_parquet(
            base_run_dir / f"outer_{outer_fold}" / "outer_predictions.parquet"
        ).sort_values("example_id")
        recomputed_outer = table[
            table["development_fold"] == outer_fold
        ].sort_values("example_id")
        maximum_outer_difference = max(
            maximum_outer_difference,
            float(
                np.max(
                    np.abs(
                        saved_outer["probability"].to_numpy()
                        - recomputed_outer["expanded_probability"].to_numpy()
                    )
                )
            ),
        )
        all_a_probability = predict_all_a(
            model, all_a_graphs, device, batch_size
        )
        profile, focal_difference = relative_features(
            table,
            all_a_metadata,
            all_a_probability,
            float(saved["best"]["threshold"]),
        )
        maximum_focal_difference = max(maximum_focal_difference, focal_difference)
        table = table.merge(profile, on="example_id", validate="one_to_one")
        shuffled = permute_within_pair(
            table[["example_id", "pair_id", *relative_columns]],
            relative_columns,
            seed + 100_000 * outer_fold,
        ).rename(
            columns={column: f"shuffled_{column}" for column in relative_columns}
        )
        table = table.merge(
            shuffled[["example_id", *[f"shuffled_{c}" for c in relative_columns]]],
            on="example_id",
            validate="one_to_one",
        )
        gate_sets = {
            "expanded_recalibrated": ["expanded_logit"],
            "expanded_plus_relative_rank": ["expanded_logit", *relative_columns],
            "expanded_plus_shuffled_rank": [
                "expanded_logit",
                *[f"shuffled_{column}" for column in relative_columns],
            ],
        }
        for name, columns in gate_sets.items():
            outer, selection = fit_gate(table, columns, outer_fold, seed)
            outer["model"] = name
            outputs.append(outer)
            selection["model"] = name
            selections.append(selection)
        reference = table[table["development_fold"] == outer_fold].copy()
        reference["model"] = "expanded_reference"
        reference["probability"] = reference["expanded_probability"]
        reference["prediction"] = reference["expanded_reference_prediction"]
        reference["threshold"] = float(saved["best"]["threshold"])
        reference["matched_recall_threshold"] = np.nan
        reference["matched_recall_prediction"] = reference["prediction"]
        outputs.append(reference)
        print(
            f"[substrate-profile gate] outer={outer_fold} complete; "
            f"focal_delta={focal_difference:.3g}",
            flush=True,
        )
    tolerance = 5e-5
    if maximum_outer_difference > tolerance:
        raise RuntimeError(
            f"recomputed expanded outer probability mismatch: "
            f"{maximum_outer_difference}"
        )
    if maximum_focal_difference > tolerance:
        raise RuntimeError(
            f"all-A/focal probability mismatch: {maximum_focal_difference}"
        )
    predictions = pd.concat(outputs, ignore_index=True)
    reference = predictions[predictions["model"] == "expanded_reference"].copy()
    metric_rows, bootstraps = [], []
    for name, group in predictions.groupby("model", sort=False):
        row = metrics(group["label"], group["probability"], group["prediction"])
        reference_view = reference if name != "expanded_reference" else group
        rescue_table = group.merge(
            reference_view[["example_id", "prediction"]].rename(
                columns={"prediction": "baseline_oof_prediction"}
            ),
            on="example_id",
            validate="one_to_one",
        )
        row.update(rescue_metrics(rescue_table, "prediction"))
        row["model"] = name
        metric_rows.append(row)
        values = bootstrap_differences(
            group, reference, bootstrap_replicates, seed + 4000
        )
        values["model"] = name
        bootstraps.append(values)
    metrics_table = pd.DataFrame(metric_rows)
    bootstrap = pd.concat(bootstraps, ignore_index=True)
    confidence = (
        bootstrap.groupby(["model", "metric"])["difference"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
    )
    destination = output_dir / mode / context
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(destination / "oof_predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(destination / "fold_selections.csv", index=False)
    metrics_table.to_csv(destination / "metrics.csv", index=False)
    bootstrap.to_parquet(destination / "pair_bootstrap.parquet", index=False)
    confidence.to_csv(destination / "pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "outer-isolated development gate; no outer pair enters either stage",
        "context": context,
        "base_mode": mode,
        "rows": len(reference),
        "pairs": int(reference["pair_id"].nunique()),
        "all_a_graphs": len(all_a_graphs),
        "maximum_recomputed_outer_difference": maximum_outer_difference,
        "maximum_all_a_focal_difference": maximum_focal_difference,
        "canonical_test_access": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    register(
        repository,
        {
            "run_id": f"substrate_profile_gate_{mode}_{context}_seed{seed}",
            "experiment_type": "outer_isolated_expanded_plus_relative_rank_gate",
            "context": context,
            "input_hashes": {
                "site_master": sha256_file(site_master_path),
                "base_checkpoints": checkpoint_hashes,
            },
            "fold_definition": "five global pair folds; outer k; inner (k+1)%5",
            "seed": seed,
            "features": "outer-model focal rank among every adenosine on its substrate",
            "hyperparameters": {
                "gate": "L2 logistic; C/threshold selected on inner fold",
                "negative_control": "relative features permuted within pair",
                "bootstrap_replicates": bootstrap_replicates,
            },
            "artifact": str(destination / "oof_predictions.parquet"),
            "metrics": metrics_table.to_dict(orient="records"),
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
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--context", default="MuscleSkeletal")
    parser.add_argument("--mode", default="binary_pair_balanced")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/runs/substrate_profile_gate")
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(
        json.dumps(
            run(
                root,
                resolve(args.site_master),
                resolve(args.base_run_dir),
                args.context,
                args.mode,
                resolve(args.output_dir),
                args.batch_size,
                args.bootstrap_replicates,
                args.seed,
                device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
