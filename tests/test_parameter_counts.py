import json

import pytest
import torch

from adarpro.evaluation.parameter_counts import build_model, run
from adarpro.models.multitask_oof import LocalCNN
from adarpro.models.unified import parameter_count


def _fake_run(tmp_path, mode, model, folds=range(5)):
    run_dir = tmp_path / mode
    for fold in folds:
        (run_dir / f"outer_{fold}").mkdir(parents=True)
        torch.save({"config": {"mode": mode}, "model_state": model.state_dict()}, run_dir / f"outer_{fold}" / "best.pth")
    return run_dir


def test_counts_every_fold_and_sums_components(tmp_path):
    cnn = _fake_run(tmp_path, "binary_pair_balanced_cnn", LocalCNN())
    out = tmp_path / "counts.json"
    summary = run(tmp_path, {"cnn": ("binary_pair_balanced_cnn", cnn)}, out)
    expected = parameter_count(LocalCNN())
    assert summary["components"]["cnn"]["parameters"] == expected
    assert set(summary["components"]["cnn"]["per_fold"].values()) == {expected}
    assert summary["total_parameters"] == expected
    assert json.loads(out.read_text())["canonical_test_access"] is False


def test_refuses_a_checkpoint_of_another_mode(tmp_path):
    cnn = _fake_run(tmp_path, "binary_pair_balanced_cnn", LocalCNN())
    with pytest.raises(ValueError, match="trained as mode"):
        run(tmp_path, {"cnn": ("binary_pair_balanced_shifted_cnn", cnn)}, tmp_path / "x.json")


def test_strict_load_rejects_a_mismatched_layout(tmp_path):
    graph_mode = "binary_pair_balanced"
    run_dir = _fake_run(tmp_path, graph_mode, LocalCNN())  # CNN weights labelled as the graph
    assert type(build_model(graph_mode)).__name__ == "MultiTaskGAT"
    with pytest.raises(RuntimeError):
        run(tmp_path, {"graph": (graph_mode, run_dir)}, tmp_path / "x.json")
