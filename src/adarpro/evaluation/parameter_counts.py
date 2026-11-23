"""Count the trainable parameters of saved grouped-OOF fold checkpoints.

Each component is a completed run folder (``outer_0`` .. ``outer_4``, each with ``best.pth``). Every fold's
checkpoint is rebuilt exactly as ``expanded_natural_prevalence`` rebuilds it, loaded strictly (so the layout must
match), and counted with ``unified.parameter_count`` (trainable parameters only). The five folds must agree.
Writes one JSON; nothing is registered and no test file is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from adarpro.models.baseline_oof import torch_load
from adarpro.models.multitask_oof import (
    LocalCNN,
    MultiTaskGAT,
    check_checkpoint_mode,
    mode_is_graph_free,
)
from adarpro.models.unified import parameter_count
from adarpro.quarantine import sha256_file


def build_model(mode: str) -> torch.nn.Module:
    """The network that ``expanded_natural_prevalence`` builds for a checkpoint of ``mode``."""

    if mode_is_graph_free(mode):
        return LocalCNN()
    return MultiTaskGAT(
        use_local=mode.endswith("_local"),
        use_target_pool=mode.endswith("_target_pool"),
        use_anchor_pool=mode.endswith("_anchor_pool"),
        use_duplicate_pool=mode.endswith("_duplicate_pool"),
        use_pair_context="_context" in mode,
    )


def count_run(run_dir: Path, mode: str) -> dict:
    per_fold, hashes = {}, {}
    for fold in range(5):
        path = run_dir / f"outer_{fold}" / "best.pth"
        saved = torch_load(path, torch.device("cpu"))
        check_checkpoint_mode(saved, mode, path)
        model = build_model(mode)
        model.load_state_dict(saved["model_state"])  # strict: the layout must match
        per_fold[str(fold)] = parameter_count(model)
        hashes[str(fold)] = sha256_file(path)
    if len(set(per_fold.values())) != 1:
        raise RuntimeError(f"{run_dir}: folds disagree on the parameter count: {per_fold}")
    return {"mode": mode, "parameters": next(iter(per_fold.values())), "per_fold": per_fold,
            "best_checkpoint_sha256_by_outer_fold": hashes}


def run(repository: Path, components: dict[str, tuple[str, Path]], output_path: Path) -> dict:
    if not components:
        raise ValueError("at least one component is required")
    counted = {}
    for name, (mode, run_dir) in components.items():
        counted[name] = count_run(run_dir, mode)
        counted[name]["run_dir"] = str(run_dir.relative_to(repository)) if run_dir.is_relative_to(repository) else str(run_dir)
    summary = {
        "scope": "trainable parameter counts of completed grouped-OOF fold checkpoints",
        "counting": "adarpro.models.unified.parameter_count (requires_grad parameters) on strictly loaded best.pth",
        "components": counted,
        "total_parameters": sum(c["parameters"] for c in counted.values()),
        "canonical_test_access": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--component", action="append", required=True,
                        help="NAME=MODE=path/to/run_dir; repeat for each component")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    components = {}
    for specification in args.component:
        parts = specification.split("=", 2)
        if len(parts) != 3 or not all(parts):
            parser.error(f"invalid --component value: {specification}")
        name, mode, raw_path = parts
        if name in components:
            parser.error(f"duplicate component name: {name}")
        components[name] = (mode, resolve(Path(raw_path)))
    print(json.dumps(run(root, components, resolve(args.output)), indent=2))


if __name__ == "__main__":
    main()
