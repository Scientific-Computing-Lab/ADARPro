"""Build a hash-complete final manifest before the one-use canonical test."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from adarpro.evaluation.final_locked import validate_manifest
from adarpro.quarantine import CONTEXTS, load_policy, sha256_file

FROZEN_INPUT_PATHS = {
    "environment_yml": "environment.yml",
    "conda_packages": "provenance/conda_packages_2026-08-21.json",
    "pip_freeze": "provenance/pip_freeze_2026-08-21.txt",
    "environment_resolution": "provenance/environment_resolution.md",
    "upstream_provenance": "provenance/upstream.json",
    "model_selection_policy": "provenance/model_selection_policy.json",
    "unified_model_selection_policy": (
        "provenance/unified_model_selection_policy.json"
    ),
    "unified_development_decision": (
        "provenance/unified_development_decision.json"
    ),
    "global_pair_split": "data_construction/split/global_pair_split.json",
    "site_master": "data/derived/site_master.parquet",
    "expanded_development_sites": (
        "data/derived/cross_tissue/development_raw_combined_sites.parquet"
    ),
    "published_baseline_metrics": "results/adarpro/baseline_reproduction.csv",
    "upstream_baseline_model": "code/gnnadar_verb_compact.py",
    "baseline_adapter": "src/adarpro/models/baseline_oof.py",
    "model_implementation": "src/adarpro/models/multitask_oof.py",
    "final_training_implementation": "src/adarpro/models/final_training.py",
    "ensemble_implementation": "src/adarpro/models/final_ensemble.py",
    "ensemble_math": "src/adarpro/models/fixed_checkpoint_ensemble.py",
    "metric_implementation": "src/adarpro/models/residual.py",
    "quarantine_implementation": "src/adarpro/quarantine.py",
    "locked_evaluation_implementation": "src/adarpro/evaluation/final_locked.py",
    **{
        f"published_baseline_checkpoint_{context}": (
            f"checkpoints/baseline_{context}/best.pth"
        )
        for context in CONTEXTS
    },
}


def frozen_input_inventory(repository: Path) -> dict[str, dict[str, str | int]]:
    """Hash every non-test file that can affect the locked prediction run."""

    inventory = {}
    for name, relative_path in FROZEN_INPUT_PATHS.items():
        path = repository / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        inventory[name] = {
            "path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return inventory


def context_specification(repository: Path, summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    context = summary.get("context")
    if context not in CONTEXTS:
        raise RuntimeError(f"invalid frozen context summary: {summary_path}")
    if summary.get("canonical_test_access") is not False:
        raise RuntimeError("final component summary is not development-only")
    if "components" in summary:
        component_modes = summary["components"]
        checkpoint_paths = summary.get("checkpoints", {})
        checkpoint_hashes = summary.get("checkpoint_sha256", {})
        if set(component_modes) != set(checkpoint_paths) or set(component_modes) != set(
            checkpoint_hashes
        ):
            raise RuntimeError("ensemble summary component metadata is incomplete")
        weights = summary.get("weights", [])
        components = [
            {
                "name": name,
                "mode": mode,
                "checkpoint": checkpoint_paths[name],
                "sha256": checkpoint_hashes[name],
            }
            for name, mode in component_modes.items()
        ]
    else:
        components = [
            {
                "name": "model",
                "mode": summary["mode"],
                "checkpoint": summary["checkpoint"],
                "sha256": summary["checkpoint_sha256"],
            }
        ]
        weights = [1.0]
    for component in components:
        checkpoint = repository / component["checkpoint"]
        if not checkpoint.is_file() or sha256_file(checkpoint) != component["sha256"]:
            raise RuntimeError(f"checkpoint identity mismatch: {checkpoint}")
    return {
        "context": context,
        "components": components,
        "weights": weights,
        "threshold": float(summary["threshold"]),
        "validation_summary": str(summary_path.relative_to(repository)),
        "validation_summary_sha256": sha256_file(summary_path),
    }


def build_manifest(
    repository: Path,
    policy_path: Path,
    summary_paths: dict[str, Path],
    candidate_name: str,
    output: Path,
) -> dict:
    policy = load_policy(policy_path)
    if policy["state"] != "development_sealed":
        raise RuntimeError("a final manifest can only be built while development is sealed")
    if set(summary_paths) != set(CONTEXTS):
        raise RuntimeError("one frozen validation summary is required for every context")
    selection_policy_path = repository / "provenance/model_selection_policy.json"
    selection_policy = json.loads(selection_policy_path.read_text())
    selection = selection_policy.get("development_selection_result", {})
    expected_components = selection.get("selected_components")
    expected_weights = selection.get("selected_weights")
    if not isinstance(expected_components, dict) or not isinstance(
        expected_weights, list
    ):
        raise TypeError("model-selection policy lacks the frozen component family")
    specifications = {}
    for expected_context, path in summary_paths.items():
        specification = context_specification(repository, path)
        if specification.pop("context") != expected_context:
            raise RuntimeError("summary context does not match command-line context")
        observed_components = {
            component["name"]: component["mode"]
            for component in specification["components"]
        }
        if observed_components != expected_components:
            raise RuntimeError(
                f"{expected_context} components differ from the selected global family"
            )
        if sorted(float(value) for value in specification["weights"]) != sorted(
            float(value) for value in expected_weights
        ):
            raise RuntimeError(
                f"{expected_context} weights differ from the selected global weights"
            )
        specifications[expected_context] = specification
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    manifest = {
        "schema_version": 1,
        "candidate_name": candidate_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": revision,
        "upstream_revision": policy["upstream_revision"],
        "benchmark": (
            "published AdarEdit human binary task; immutable global pair split; "
            "published per-context test protocol"
        ),
        "selection_policy": "provenance/model_selection_policy.json",
        "selection_policy_sha256": sha256_file(selection_policy_path),
        "development_protocol": {
            "grouping_unit": "pair_id",
            "development_folds": "five global pair-disjoint folds",
            "training_split": "published train pairs; expanded raw high/low sites",
            "selection_split": "published validation canonical rows",
            "checkpoint_selection": "maximum validation F1 over 1000 epochs",
            "threshold_selection": "published validation only",
            "random_seed": 42,
            "canonical_test_access_during_selection": False,
        },
        "candidate_specification": {
            "family": selection["selected_family"],
            "components": expected_components,
            "weights": expected_weights,
            "fusion": "equal-weight logit average",
        },
        "frozen_inputs": frozen_input_inventory(repository),
        "contexts": specifications,
    }
    validate_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite a final manifest: {output}")
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--policy", type=Path, default=Path("provenance/test_quarantine.json")
    )
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="CONTEXT=path/to/frozen/summary.json; provide all five contexts",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("provenance/final_pipeline_manifest.json")
    )
    args = parser.parse_args()
    repository = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repository / path

    summaries = {}
    for value in args.summary:
        context, separator, raw_path = value.partition("=")
        if not separator or context not in CONTEXTS or context in summaries:
            parser.error(f"invalid --summary value: {value}")
        summaries[context] = resolve(Path(raw_path))
    manifest = build_manifest(
        repository,
        resolve(args.policy),
        summaries,
        args.candidate_name,
        resolve(args.output),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
