"""The graph-free CNN (``LocalCNN``, mode ``binary_pair_balanced_cnn``).

The reference for "same CNN as the hybrid" is the hybrid's own, untouched ``MultiTaskGAT(use_local=True)`` code
(test (d)); parameter counts are worked by hand from the layer shapes (test (b)), not read from the code.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from adarpro.models.final_training import FINAL_MODES
from adarpro.models.fixed_checkpoint_ensemble import model_for_mode
from adarpro.models.multitask_oof import (
    CNN_MODES,
    MODES,
    LocalCNN,
    MultiTaskGAT,
    canonical_window_examples,
    create_model,
    local_window,
    make_training_graphs,
    mode_is_graph_free,
    mode_uses_local_window,
    partner_lane_run_suffix,
    saved_partner_lane,
    train_epoch,
)

CNN = "binary_pair_balanced_cnn"


def baseline_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    import gnnadar_verb_compact as baseline

    return baseline


def window_example(sequence: str, structure: str, target: int, label: float) -> Data:
    return Data(
        local_window=local_window(sequence, structure, target),
        y=torch.tensor([label], dtype=torch.float32),
    )


# (a) no trace of the graph
def test_state_dict_has_only_the_cnn_and_three_heads():
    keys = set(LocalCNN().state_dict())
    assert keys == {
        "local_convolution.0.weight",
        "local_convolution.0.bias",
        "local_convolution.2.weight",
        "local_convolution.2.bias",
        "binary_head.weight",
        "binary_head.bias",
        "ordinal_head.weight",
        "ordinal_head.bias",
        "continuous_head.weight",
        "continuous_head.bias",
    }
    assert not any("gat" in key or "batch_norm" in key for key in keys)


# (b) hand count: conv1 12*32*5 + 32 = 1,952; conv2 32*32*5 + 32 = 5,152 (CNN branch 7,104);
#     heads on 64 dims: binary 64 + 1 = 65, ordinal 64*3 + 3 = 195, continuous 65 (325); total 7,429
def test_parameter_count_matches_the_hand_count():
    model = LocalCNN()
    assert sum(p.numel() for p in model.local_convolution.parameters()) == 7_104
    assert sum(p.numel() for p in model.parameters()) == 7_429
    hybrid = MultiTaskGAT(use_local=True)
    assert sum(p.numel() for p in hybrid.local_convolution.parameters()) == 7_104


# (c) runs on window-only examples; the hybrid cannot, so the test tells them apart
def test_forward_needs_only_the_window_and_the_hybrid_cannot():
    batch = Batch.from_data_list(
        [
            window_example("AACGUU", "((..))", 1, 1.0),
            window_example("GAGAAAAACAC", "(.(.....).)", 1, 0.0),
        ]
    )
    assert batch.local_window.shape == (2, 12, 65)
    assert getattr(batch, "x", None) is None and getattr(batch, "edge_index", None) is None
    model = LocalCNN().eval()
    with torch.no_grad():
        binary, ordinal, continuous = model(batch)
    assert binary.shape == (2,) and ordinal.shape == (2, 3) and continuous.shape == (2,)
    assert ((continuous >= 0) & (continuous <= 1)).all()
    with pytest.raises((AttributeError, TypeError)):
        MultiTaskGAT(use_local=True).eval()(batch)


# (d) the same CNN as the hybrid's branch: with the hybrid's convolution weights copied in, the 64-d readout
#     equals the last 64 dims of the hybrid's own embedding on the same windows
def test_readout_equals_the_hybrid_local_branch():
    baseline = baseline_module()
    torch.manual_seed(17)
    hybrid = MultiTaskGAT(use_local=True).eval()
    cnn = LocalCNN().eval()
    cnn.local_convolution.load_state_dict(hybrid.local_convolution.state_dict())
    cases = [("AACGUU", "((..))", 1), ("GAGAAAAACAC", "(.(.....).)", 1), ("GAGAAAAACAC", "(.(.....).)", 4)]
    graphs, windows = [], []
    for sequence, structure, target in cases:
        graph = baseline.create_rna_graph(sequence, structure, target, 1)
        graph.local_window = local_window(sequence, structure, target)
        graphs.append(graph)
        windows.append(window_example(sequence, structure, target, 1.0))
    with torch.no_grad():
        hybrid_embedding, _ = hybrid.encode(Batch.from_data_list(graphs))
        cnn_embedding = cnn.encode(Batch.from_data_list(windows))
    assert hybrid_embedding.shape == (3, 128 + 64)
    assert torch.equal(hybrid_embedding[:, -64:], cnn_embedding)


# (e) mode routing
def test_the_cnn_mode_routes_to_the_graph_free_network():
    assert CNN in MODES and CNN in FINAL_MODES and CNN_MODES == (CNN,)
    assert not CNN.endswith("_local") and "_context" not in CNN
    assert mode_is_graph_free(CNN) and mode_uses_local_window(CNN)
    assert mode_uses_local_window("binary_pair_balanced_local")
    assert not mode_is_graph_free("binary_pair_balanced_local")
    assert not mode_uses_local_window("binary_pair_balanced")
    model, optimizer = create_model(torch.device("cpu"), CNN)
    assert isinstance(model, LocalCNN) and not hasattr(model, "gat_layers")
    assert optimizer.defaults["lr"] == 1e-3 and optimizer.defaults["weight_decay"] == 1e-4
    hybrid, _ = create_model(torch.device("cpu"), "binary_pair_balanced_local")
    assert isinstance(hybrid, MultiTaskGAT) and hybrid.use_local
    assert isinstance(model_for_mode(CNN, torch.device("cpu")), LocalCNN)
    assert isinstance(model_for_mode("binary_pair_balanced_local", torch.device("cpu")), MultiTaskGAT)


# (f) the partner lane of a CNN checkpoint is always recorded; a CNN config without it is refused
def test_cnn_checkpoints_must_record_their_partner_lane():
    with pytest.raises(RuntimeError):
        saved_partner_lane({"config": {"mode": CNN}})
    assert saved_partner_lane({"config": {"mode": CNN, "partner_lane": "opposite"}}) == "opposite"
    assert saved_partner_lane({"config": {"mode": CNN, "partner_lane": "legacy"}}) == "legacy"
    assert saved_partner_lane({"config": {"mode": "binary_pair_balanced_local"}}) == "legacy"
    assert partner_lane_run_suffix("opposite", mode_uses_local_window(CNN)) == "_partner_opposite"


# graph-free example builders: same labels, same windows, same order, no graph
def test_graph_free_examples_carry_labels_and_windows_only():
    table = pd.DataFrame(
        {
            "example_id": ["a", "b", "c"],
            "sequence": ["AACGUU", "GAGAAAAACAC", "GAGAAAAACAC"],
            "structure": ["((..))", "(.(.....).)", "(.(.....).)"],
            "local_position": [1, 1, 4],
            "label": [1, 0, 1],
        }
    )
    examples = canonical_window_examples(table)
    assert [float(example.y) for example in examples] == [1.0, 0.0, 1.0]
    for example, row in zip(examples, table.itertuples(index=False)):
        assert set(example.keys()) == {"y", "local_window"}
        assert torch.equal(
            example.local_window, local_window(row.sequence, row.structure, row.local_position)
        )
    with pytest.raises(RuntimeError):
        canonical_window_examples(table.assign(structure=["((..))", "(.(.....).)", "(.(....)"]))


def test_graph_free_raw_examples_match_the_graph_builder_except_the_graph():
    raw = pd.DataFrame(
        {
            "example_id": ["r1", "r2", "r3"],
            "sequence": ["AACGUU", "GAGAAAAACAC", "GAGAAAAACAC"],
            "structure": ["((..))", "(.(.....).)", "(.(.....).)"],
            "local_position": [1, 1, 4],
            "editing_index": [0.0, 7.0, 40.0],
        }
    )
    with_graph = make_training_graphs(raw, baseline_module())
    graph_free = make_training_graphs(raw, None, include_graph=False)
    label_fields = ("y", "ordinal_label", "editing_fraction", "binary_mask", "binary_label", "local_window")
    for full, bare in zip(with_graph, graph_free):
        assert getattr(bare, "x", None) is None and getattr(bare, "edge_index", None) is None
        assert set(bare.keys()) == set(label_fields)
        for field in label_fields:
            assert torch.equal(getattr(full, field), getattr(bare, field)), field


def test_one_training_step_on_graph_free_examples():
    examples = []
    for label in (0.0, 1.0, 1.0, 0.0):
        example = window_example("GAGAAAAACAC", "(.(.....).)", 1, label)
        example.binary_mask = torch.tensor([True])
        example.binary_label = torch.tensor([label])
        examples.append(example)
    model, optimizer = create_model(torch.device("cpu"), CNN)
    losses = train_epoch(model, DataLoader(examples, batch_size=2), optimizer, CNN, torch.device("cpu"))
    assert losses["loss"] > 0


# the published folders are protected from a new component or ensemble
def test_final_training_refuses_the_published_frozen_folder(tmp_path):
    from adarpro.models.final_training import run

    with pytest.raises(RuntimeError, match="models/frozen"):
        run(tmp_path, tmp_path / "none.parquet", tmp_path / "none.parquet", "MuscleSkeletal", CNN,
            tmp_path / "models" / "frozen", 1, 2, 1, 42, torch.device("cpu"))


def test_final_ensemble_refuses_the_published_ensemble_folder(tmp_path):
    from adarpro.models.final_ensemble import run

    pd.DataFrame(
        {"context": ["MuscleSkeletal"], "source_split": ["valid"], "label": [1]}
    ).to_parquet(tmp_path / "master.parquet")
    common = {
        "context": "MuscleSkeletal",
        "canonical_test_access": False,
        "training_split": "published train pairs; expanded raw high/low sites",
        "selection_split": "published validation canonical rows",
    }
    paths = []
    for name, mode, extra in (("pair", "binary_pair_balanced", {}), ("cnn", CNN, {"partner_lane": "opposite"})):
        path = tmp_path / f"{name}.pth"
        torch.save({"config": {**common, "mode": mode, **extra}}, path)
        paths.append((name, mode, path))
    with pytest.raises(RuntimeError, match="frozen_ensemble"):
        run(tmp_path, tmp_path / "master.parquet", "MuscleSkeletal", paths,
            tmp_path / "models" / "frozen_ensemble", 2, 42, torch.device("cpu"))
    destination = tmp_path / "elsewhere" / "MuscleSkeletal"
    destination.mkdir(parents=True)
    (destination / "summary.json").write_text(json.dumps({"components": {"pair": "binary_pair_balanced"}}))
    with pytest.raises(RuntimeError, match="different ensemble"):
        run(tmp_path, tmp_path / "master.parquet", "MuscleSkeletal", paths,
            tmp_path / "elsewhere", 2, 42, torch.device("cpu"))
