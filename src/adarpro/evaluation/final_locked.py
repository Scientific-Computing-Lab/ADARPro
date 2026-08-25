"""Execute the frozen ADARpro pipeline on the canonical test exactly once."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from adarpro.models.baseline_oof import predict, torch_load, upstream_modules
from adarpro.models.final_ensemble import validate_checkpoint
from adarpro.models.final_training import FINAL_MODES
from adarpro.models.fixed_checkpoint_ensemble import (
    equal_logit_average_many,
    model_for_mode,
)
from adarpro.models.multitask_oof import attach_local_windows, predict_binary
from adarpro.models.residual import metrics
from adarpro.quarantine import (
    CONTEXTS,
    QuarantineViolation,
    consume_final_authorization,
    load_policy,
    sha256_file,
)
from adarpro.registry import register

MANIFEST_SCHEMA = 1
BOOTSTRAP_METRICS = (
    "auroc",
    "auprc",
    "f1",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "brier",
)


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError("unsupported final manifest schema")
    if not manifest.get("candidate_name") or not manifest.get("code_revision"):
        raise RuntimeError("final manifest lacks candidate name or code revision")
    if not manifest.get("selection_policy_sha256"):
        raise RuntimeError("final manifest lacks the model-selection policy hash")
    protocol = manifest.get("development_protocol", {})
    if protocol.get("grouping_unit") != "pair_id":
        raise RuntimeError("final manifest lacks the pair-grouped development protocol")
    if protocol.get("canonical_test_access_during_selection") is not False:
        raise RuntimeError("final manifest does not attest sealed model selection")
    candidate = manifest.get("candidate_specification", {})
    if (
        not candidate.get("family")
        or not candidate.get("components")
        or not candidate.get("weights")
    ):
        raise RuntimeError("final manifest lacks the selected candidate specification")
    frozen_inputs = manifest.get("frozen_inputs", {})
    if not frozen_inputs:
        raise RuntimeError("final manifest lacks a frozen input inventory")
    for name, artifact in frozen_inputs.items():
        if (
            not name
            or not artifact.get("path")
            or not artifact.get("sha256")
            or int(artifact.get("bytes", -1)) < 0
        ):
            raise RuntimeError(f"invalid frozen input record: {name}")
    if set(manifest.get("contexts", {})) != set(CONTEXTS):
        raise RuntimeError("final manifest must freeze all five benchmark contexts")
    for context, specification in manifest["contexts"].items():
        if not specification.get("validation_summary") or not specification.get(
            "validation_summary_sha256"
        ):
            raise RuntimeError(f"missing frozen validation summary for {context}")
        components = specification.get("components", [])
        weights = np.asarray(specification.get("weights", []), dtype=float)
        if not components or len(weights) != len(components):
            raise RuntimeError(f"invalid component/weight list for {context}")
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
            raise RuntimeError(f"invalid frozen weights for {context}")
        threshold = float(specification.get("threshold", np.nan))
        if not 0 <= threshold <= 1:
            raise RuntimeError(f"invalid frozen threshold for {context}")
        names = [component.get("name") for component in components]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise RuntimeError(f"invalid component names for {context}")
        for component in components:
            mode = component.get("mode", "")
            if mode not in FINAL_MODES:
                raise RuntimeError(f"unsupported final component mode: {mode}")
            if not component.get("checkpoint") or not component.get("sha256"):
                raise RuntimeError(f"incomplete frozen component for {context}")
        if {component["name"]: component["mode"] for component in components} != (
            candidate["components"]
        ):
            raise RuntimeError(f"candidate-family mismatch for {context}")
        if not np.allclose(weights, np.asarray(candidate["weights"], dtype=float)):
            raise RuntimeError(f"candidate-weight mismatch for {context}")


def weighted_logit_average(
    probabilities: list[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    if len(probabilities) == 1:
        return np.asarray(probabilities[0], dtype=float)
    if np.allclose(weights, np.repeat(1 / len(weights), len(weights))):
        return equal_logit_average_many(probabilities)
    epsilon = np.finfo(np.float64).eps
    logits = [
        np.log(np.clip(values, epsilon, 1 - epsilon) / np.clip(1 - values, epsilon, 1))
        for values in probabilities
    ]
    return 1 / (1 + np.exp(-np.average(logits, axis=0, weights=weights)))


def load_canonical_table(repository: Path, context: str, baseline) -> pd.DataFrame:
    """Read an authorized canonical split and preserve JSONL/metadata order."""

    jsonl_path = repository / "data" / "human" / context / "test.jsonl"
    metadata_path = repository / "data" / "human" / context / "test.metadata.csv"
    parsed = []
    with jsonl_path.open() as handle:
        for index, line in enumerate(handle):
            sequence, structure, target, label = baseline.parse_openai_json(
                json.loads(line)
            )
            if not sequence or not structure or len(sequence) != len(structure):
                raise RuntimeError(f"invalid canonical row {context}/{index}")
            parsed.append(
                {
                    "example_id": f"{context}:canonical:{index}",
                    "context": context,
                    "sequence": sequence,
                    "structure": structure,
                    "local_position": target,
                    "label": label,
                }
            )
    table = pd.DataFrame(parsed)
    metadata = pd.read_csv(metadata_path)
    if len(table) != len(metadata):
        raise RuntimeError(f"canonical JSONL/metadata length mismatch for {context}")
    expected = metadata["yes_no"].map({"no": 0, "yes": 1}).to_numpy()
    if not np.array_equal(table["label"].to_numpy(), expected):
        raise RuntimeError(f"canonical label/metadata mismatch for {context}")
    table["pair_id"] = metadata["pair_id"].to_numpy()
    table["row_index"] = np.arange(len(table))
    return table


def verify_artifact_inventory(policy: dict, repository: Path) -> None:
    for artifact in policy["test_artifacts"]:
        path = repository / artifact["path"]
        if path.stat().st_size != artifact["bytes"]:
            raise QuarantineViolation(f"canonical artifact size changed: {path}")
        if sha256_file(path) != artifact["sha256"]:
            raise QuarantineViolation(f"canonical artifact hash changed: {path}")


def verify_frozen_inputs(manifest: dict, repository: Path) -> None:
    """Verify every environment, data and implementation input bound at freeze."""

    for name, artifact in manifest["frozen_inputs"].items():
        path = repository / artifact["path"]
        if not path.is_file():
            raise RuntimeError(f"missing frozen input {name}: {path}")
        if path.stat().st_size != int(artifact["bytes"]):
            raise RuntimeError(f"frozen input size mismatch for {name}: {path}")
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"frozen input hash mismatch for {name}: {path}")
    selection = manifest["frozen_inputs"].get("model_selection_policy")
    if selection is None or selection["sha256"] != manifest["selection_policy_sha256"]:
        raise RuntimeError("selection-policy hashes are inconsistent")


def preflight(
    repository: Path, policy_path: Path, manifest_path: Path, output_dir: Path
) -> dict:
    """Validate every non-test dependency before spending the one-use token."""

    policy = load_policy(policy_path)
    if policy["state"] != "final_frozen":
        raise QuarantineViolation("final pipeline is not frozen")
    frozen = policy["final_locked_test"]["freeze_manifest"]
    if frozen is None or sha256_file(manifest_path) != frozen["sha256"]:
        raise QuarantineViolation("provided manifest is not the frozen manifest")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    if manifest.get("code_revision") != revision:
        raise RuntimeError("working revision differs from frozen final manifest")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to overwrite an existing final-test result")
    # Artifact identity is non-outcome-bearing and can be checked before the
    # one-use authorization is spent.  Keep every deterministic failure that
    # does not require parsing canonical rows on this side of the boundary.
    verify_artifact_inventory(policy, repository)
    verify_frozen_inputs(manifest, repository)
    for context, specification in manifest["contexts"].items():
        summary_path = repository / specification["validation_summary"]
        if not summary_path.is_file():
            raise RuntimeError(f"missing frozen validation summary: {summary_path}")
        if sha256_file(summary_path) != specification["validation_summary_sha256"]:
            raise RuntimeError(f"frozen validation summary changed: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if float(summary["threshold"]) != float(specification["threshold"]):
            raise RuntimeError(f"frozen threshold mismatch for {context}")
        if summary.get("components") != {
            component["name"]: component["mode"]
            for component in specification["components"]
        }:
            raise RuntimeError(f"frozen component summary mismatch for {context}")
        if not np.allclose(summary.get("weights", []), specification["weights"]):
            raise RuntimeError(f"frozen ensemble-weight mismatch for {context}")
        for component in specification["components"]:
            checkpoint = repository / component["checkpoint"]
            if not checkpoint.is_file():
                raise RuntimeError(f"missing frozen checkpoint: {checkpoint}")
            if sha256_file(checkpoint) != component["sha256"]:
                raise RuntimeError(f"frozen checkpoint hash mismatch: {checkpoint}")
            saved = torch_load(checkpoint, torch.device("cpu"))
            validate_checkpoint(saved, context, component["mode"])
    return manifest


def baseline_predictions(
    repository: Path,
    context: str,
    graphs: list,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float, dict]:
    baseline = upstream_modules(repository)
    model = baseline.RNAEditingGNN(8, 32, 1, 4).to(device)
    checkpoint = repository / "checkpoints" / f"baseline_{context}" / "best.pth"
    saved = torch_load(checkpoint, device)
    model.load_state_dict(saved["model_state"])
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    probability, labels = predict(model, loader, device)
    threshold = float(saved["best"]["metrics"]["threshold"])
    return probability, threshold, {
        "labels": labels,
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def assert_published_baseline(
    context: str, observed: dict, published: pd.DataFrame
) -> None:
    row = published[
        (published["context"] == context) & (published["variant"] == "baseline")
    ]
    if len(row) != 1:
        raise RuntimeError(f"missing published baseline aggregate for {context}")
    expected = row.iloc[0]
    for metric in (
        "auroc",
        "auprc",
        "f1",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "mcc",
    ):
        if abs(float(observed[metric]) - float(expected[metric])) > 2e-5:
            raise RuntimeError(
                f"published baseline mismatch for {context}/{metric}: "
                f"{observed[metric]} versus {expected[metric]}"
            )


def candidate_predictions(
    repository: Path,
    specification: dict,
    table: pd.DataFrame,
    graphs: list,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if any(component["mode"].endswith("_local") for component in specification["components"]):
        attach_local_windows(graphs, table)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    outputs = {}
    probabilities = []
    for component in specification["components"]:
        checkpoint = repository / component["checkpoint"]
        if sha256_file(checkpoint) != component["sha256"]:
            raise RuntimeError(f"frozen checkpoint hash mismatch: {checkpoint}")
        saved = torch_load(checkpoint, device)
        config = saved.get("config", {})
        if config.get("context") != table["context"].iloc[0]:
            raise RuntimeError("frozen checkpoint context mismatch")
        if config.get("mode") != component["mode"]:
            raise RuntimeError("frozen checkpoint mode mismatch")
        model = model_for_mode(component["mode"], device)
        model.load_state_dict(saved["model_state"])
        probability, labels = predict_binary(model, loader, device)
        if not np.array_equal(labels, table["label"].to_numpy()):
            raise RuntimeError("candidate canonical row mismatch")
        probabilities.append(probability)
        outputs[component["name"]] = probability
    weights = np.asarray(specification["weights"], dtype=float)
    return weighted_logit_average(probabilities, weights), outputs


def pair_bootstrap_delta(
    predictions: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    pairs = predictions["pair_id"].unique()
    pair_values = predictions["pair_id"].to_numpy()
    indices = {pair: np.flatnonzero(pair_values == pair) for pair in pairs}
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        subset = predictions.iloc[np.concatenate([indices[pair] for pair in sampled])]
        candidate = metrics(
            subset["label"], subset["probability"], subset["prediction"]
        )
        baseline = metrics(
            subset["label"],
            subset["baseline_probability"],
            subset["baseline_prediction"],
        )
        for metric in BOOTSTRAP_METRICS:
            rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "difference": candidate[metric] - baseline[metric],
                }
            )
    return pd.DataFrame(rows)


def run(
    repository: Path,
    policy_path: Path,
    manifest_path: Path,
    authorization: str,
    output_dir: Path,
    batch_size: int,
    bootstrap_replicates: int,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    manifest = preflight(repository, policy_path, manifest_path, output_dir)
    # Only after every non-test dependency passes preflight do we consume the
    # single-use authorization and permit canonical row access.
    consume_final_authorization(policy_path, authorization)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_module = upstream_modules(repository)
    published = pd.read_csv(repository / "results/adarpro/baseline_reproduction.csv")
    metric_rows = []
    for context_index, context in enumerate(CONTEXTS):
        table = load_canonical_table(repository, context, baseline_module)
        graphs = [
            baseline_module.create_rna_graph(
                row.sequence,
                row.structure,
                int(row.local_position),
                int(row.label),
            )
            for row in table.itertuples(index=False)
        ]
        if any(graph is None for graph in graphs):
            raise RuntimeError(f"unbuildable canonical graph for {context}")
        baseline_probability, baseline_threshold, baseline_info = baseline_predictions(
            repository, context, graphs, device, batch_size
        )
        if not np.array_equal(baseline_info["labels"], table["label"].to_numpy()):
            raise RuntimeError("published baseline canonical row mismatch")
        probability, components = candidate_predictions(
            repository,
            manifest["contexts"][context],
            table,
            graphs,
            device,
            batch_size,
        )
        threshold = float(manifest["contexts"][context]["threshold"])
        table["probability"] = probability
        table["prediction"] = (probability >= threshold).astype(int)
        table["baseline_probability"] = baseline_probability
        table["baseline_prediction"] = (
            baseline_probability >= baseline_threshold
        ).astype(int)
        for name, values in components.items():
            table[f"{name}_probability"] = values
        table.to_parquet(output_dir / f"{context}_predictions.parquet", index=False)

        candidate_score = metrics(table["label"], probability, table["prediction"])
        baseline_score = metrics(
            table["label"], baseline_probability, table["baseline_prediction"]
        )
        assert_published_baseline(context, baseline_score, published)
        metric_rows.extend(
            [
                {
                    "context": context,
                    "model": "published_baseline_reproduced",
                    "threshold": baseline_threshold,
                    "pair_count": int(table["pair_id"].nunique()),
                    **baseline_score,
                },
                {
                    "context": context,
                    "model": manifest["candidate_name"],
                    "threshold": threshold,
                    "pair_count": int(table["pair_id"].nunique()),
                    **candidate_score,
                },
            ]
        )
        bootstrap = pair_bootstrap_delta(
            table, bootstrap_replicates, seed + context_index
        )
        bootstrap.to_parquet(
            output_dir / f"{context}_pair_bootstrap.parquet", index=False
        )
        confidence = (
            bootstrap.groupby("metric")["difference"]
            .quantile([0.025, 0.5, 0.975])
            .unstack()
            .reset_index()
            .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"})
        )
        confidence.to_csv(
            output_dir / f"{context}_pair_bootstrap_ci.csv", index=False
        )

    metric_table = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "metrics.csv"
    metric_table.to_csv(metrics_path, index=False)
    summary = {
        "scope": "single consumed canonical-test evaluation",
        "candidate": manifest["candidate_name"],
        "manifest_sha256": sha256_file(manifest_path),
        "bootstrap_replicates": bootstrap_replicates,
        "metrics_sha256": sha256_file(metrics_path),
        "canonical_test_access": True,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    register(
        repository,
        {
            "run_id": f"final_locked_test_{manifest['candidate_name']}",
            "experiment_type": "single_consumed_canonical_test_evaluation",
            "context": "all five canonical contexts",
            "input_hashes": {"frozen_manifest": sha256_file(manifest_path)},
            "fold_definition": "immutable published 64:16:20 pair split; test only",
            "seed": seed,
            "features": manifest["candidate_name"],
            "hyperparameters": {"bootstrap_replicates": bootstrap_replicates},
            "artifact": str(metrics_path.relative_to(repository)),
            "metrics": summary,
            "canonical_test_access": True,
        },
    )
    return metric_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--policy", type=Path, default=Path("provenance/test_quarantine.json")
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/adarpro/final_locked_test")
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    repository = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repository / path

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    table = run(
        repository,
        resolve(args.policy),
        resolve(args.manifest),
        args.authorization,
        resolve(args.output_dir),
        args.batch_size,
        args.bootstrap_replicates,
        args.seed,
        device,
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
