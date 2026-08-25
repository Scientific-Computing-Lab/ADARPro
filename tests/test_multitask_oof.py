from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from adarpro.models.multitask_oof import (
    MultiTaskGAT,
    attach_pair_context,
    local_window,
    pair_context_table,
    predict_binary_and_engagement,
    sample_epoch_indices,
    shifted_local_position,
    train_epoch,
)


def test_balanced_epoch_sampler_uses_requested_states():
    states = {
        0: np.arange(0, 10),
        1: np.arange(10, 20),
        2: np.arange(20, 30),
    }
    selected = sample_epoch_indices(states, 12, "multitask", np.random.default_rng(7))
    assert len(selected) == 12
    assert sum(value < 10 for value in selected) == 4
    assert sum(10 <= value < 20 for value in selected) == 4
    assert sum(value >= 20 for value in selected) == 4

    binary_local = sample_epoch_indices(
        states, 10, "binary_natural_local", np.random.default_rng(8)
    )
    assert not any(10 <= value < 20 for value in binary_local)


def test_pair_balanced_sampler_weights_pairs_not_rows():
    states = {0: np.arange(0, 10), 1: np.array([], dtype=int), 2: np.arange(10, 14)}
    pairs = np.array([1] * 9 + [2] + [3] * 2 + [4] * 2)
    selected = sample_epoch_indices(
        states,
        2000,
        "binary_pair_balanced",
        np.random.default_rng(9),
        pair_values=pairs,
    )
    low_pairs = pairs[selected[selected < 10]]
    assert 0.4 < np.mean(low_pairs == 1) < 0.6


def test_pair_balanced_multitask_sampler_uses_all_three_states():
    states = {
        0: np.arange(0, 10),
        1: np.arange(10, 20),
        2: np.arange(20, 30),
    }
    pairs = np.repeat(np.arange(6), 5)
    selected = sample_epoch_indices(
        states,
        30,
        "multitask_pair_balanced",
        np.random.default_rng(10),
        pair_values=pairs,
    )
    assert len(selected) == 30
    assert sum(selected < 10) == 10
    assert sum((selected >= 10) & (selected < 20)) == 10
    assert sum(selected >= 20) == 10


def test_multitask_gat_emits_three_heads():
    # A minimal local stand-in for the upstream graph constructor is enough to
    # validate tensor/head dimensions without touching repository data.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    import gnnadar_verb_compact as baseline

    graph = baseline.create_rna_graph("AACGUU", "((..))", 1, 1)
    batch = Batch.from_data_list([graph, graph])
    model = MultiTaskGAT()
    model.eval()
    with torch.no_grad():
        binary, ordinal, continuous = model(batch)
    assert binary.shape == (2,)
    assert ordinal.shape == (2, 3)
    assert continuous.shape == (2,)
    assert torch.all((continuous >= 0) & (continuous <= 1))


def test_local_window_is_target_centered_and_padded():
    window = local_window("AGC", ".()", 0)
    assert window.shape == (1, 12, 65)
    assert window[0, 0, 32] == 1
    assert window[0, 1, 33] == 1
    assert window[0, 5, 33] == 1
    assert window[0, 4, 0] == 1
    # An unpaired target has no helix-aligned partner and keeps the partner
    # view padded as N.
    assert window[0, 10, 32] == 1


def test_local_window_aligns_the_partner_arm_in_reverse():
    window = local_window("AACGUU", "((..))", 1)
    assert window[0, 0, 32] == 1
    assert window[0, 0, 31] == 1
    assert window[0, 2, 33] == 1
    # Target 1 pairs with 4; moving +1 on the target arm aligns to 3.
    assert window[0, 9, 32] == 1
    assert window[0, 7, 33] == 1


def test_shifted_local_control_uses_distant_deterministic_anchor():
    assert shifted_local_position(300, 50) == 146
    assert shifted_local_position(300, 250) == 154
    assert shifted_local_position(80, 10) == 79
    assert shifted_local_position(80, 70) == 0


def test_local_multitask_gat_batches_windows():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    import gnnadar_verb_compact as baseline

    graphs = [baseline.create_rna_graph("AACGUU", "((..))", 1, 1) for _ in range(2)]
    for graph in graphs:
        graph.local_window = local_window("AACGUU", "((..))", 1)
    batch = Batch.from_data_list(graphs)
    model = MultiTaskGAT(use_local=True)
    model.eval()
    with torch.no_grad():
        binary, ordinal, continuous = model(batch)
    assert binary.shape == (2,)
    assert ordinal.shape == (2, 3)
    assert continuous.shape == (2,)


def test_target_pool_multitask_gat_batches_focal_nodes():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    import gnnadar_verb_compact as baseline

    graphs = [baseline.create_rna_graph("AACGUU", "((..))", 1, 1) for _ in range(2)]
    batch = Batch.from_data_list(graphs)
    model = MultiTaskGAT(use_target_pool=True)
    model.eval()
    with torch.no_grad():
        binary, ordinal, continuous = model(batch)
    assert binary.shape == (2,)
    assert ordinal.shape == (2, 3)
    assert continuous.shape == (2,)


def test_anchor_pool_is_a_shape_matched_nonfocal_control():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    import gnnadar_verb_compact as baseline

    graphs = [baseline.create_rna_graph("AACGUU", "((..))", 1, 1) for _ in range(2)]
    batch = Batch.from_data_list(graphs)
    target_model = MultiTaskGAT(use_target_pool=True)
    anchor_model = MultiTaskGAT(use_anchor_pool=True)
    assert sum(parameter.numel() for parameter in target_model.parameters()) == sum(
        parameter.numel() for parameter in anchor_model.parameters()
    )
    with torch.no_grad():
        binary, ordinal, continuous = anchor_model(batch)
    assert binary.shape == (2,)
    assert ordinal.shape == (2, 3)
    assert continuous.shape == (2,)


def test_duplicate_pool_is_parameter_matched_without_new_node_information():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    import gnnadar_verb_compact as baseline

    graph = baseline.create_rna_graph("AACGUU", "((..))", 1, 1)
    batch = Batch.from_data_list([graph, graph])
    target_model = MultiTaskGAT(use_target_pool=True)
    control_model = MultiTaskGAT(use_duplicate_pool=True)
    assert sum(parameter.numel() for parameter in target_model.parameters()) == sum(
        parameter.numel() for parameter in control_model.parameters()
    )
    with torch.no_grad():
        binary, ordinal, continuous = control_model(batch)
    assert binary.shape == (2,)
    assert ordinal.shape == (2, 3)
    assert continuous.shape == (2,)


def test_pooling_variants_have_identical_seeded_initial_parameters():
    models = []
    for options in (
        {"use_target_pool": True},
        {"use_anchor_pool": True},
        {"use_duplicate_pool": True},
    ):
        torch.manual_seed(17)
        models.append(MultiTaskGAT(**options))
    reference = models[0].state_dict()
    for model in models[1:]:
        assert reference.keys() == model.state_dict().keys()
        for name, value in reference.items():
            assert torch.equal(value, model.state_dict()[name]), name


def test_pair_context_model_emits_engagement_and_site_outputs():
    import sys

    import pandas as pd
    from torch_geometric.loader import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    import gnnadar_verb_compact as baseline

    sequence = "AANNNNNNNNNNGU"
    structure = "((..........))"
    raw = pd.DataFrame(
        {
            "pair_id": [1, 1],
            "editing_index": [0.0, 20.0],
            "sequence": [sequence, sequence],
            "structure": [structure, structure],
            "start1": [0, 0],
            "end1": [1, 1],
            "start2": [12, 12],
            "end2": [13, 13],
            "strand": ["+", "+"],
        }
    )
    context = pair_context_table(raw)
    graphs = [baseline.create_rna_graph(sequence, structure, 1, 1) for _ in range(2)]
    attach_pair_context(graphs, raw, context)
    batch = next(iter(DataLoader(graphs, batch_size=2)))
    model = MultiTaskGAT(use_pair_context=True)
    model.eval()
    with torch.no_grad():
        binary, ordinal, continuous, engagement = model.forward_with_engagement(batch)
    assert binary.shape == (2,)
    assert ordinal.shape == (2, 3)
    assert continuous.shape == engagement.shape == (2,)
    assert torch.all((engagement >= 0) & (engagement <= 1))


def test_pair_context_engagement_loss_trains():
    import sys

    import pandas as pd
    from torch_geometric.loader import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    import gnnadar_verb_compact as baseline

    sequence = "AANNNNNNNNNNGU"
    structure = "((..........))"
    raw = pd.DataFrame(
        {
            "pair_id": [1, 1],
            "editing_index": [0.0, 20.0],
            "sequence": [sequence, sequence],
            "structure": [structure, structure],
            "start1": [0, 0],
            "end1": [1, 1],
            "start2": [12, 12],
            "end2": [13, 13],
            "strand": ["+", "+"],
        }
    )
    context = pair_context_table(raw)
    graphs = [
        baseline.create_rna_graph(sequence, structure, 1, label) for label in (0, 1)
    ]
    for graph, label in zip(graphs, (0.0, 1.0)):
        graph.binary_mask = torch.tensor([True])
        graph.binary_label = torch.tensor([label])
    attach_pair_context(graphs, raw, context)
    model = MultiTaskGAT(use_pair_context=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = train_epoch(
        model,
        DataLoader(graphs, batch_size=2),
        optimizer,
        "binary_pair_balanced_context_engagement",
        torch.device("cpu"),
    )
    assert losses["loss"] > 0
    assert losses["engagement"] > 0


def test_context_mode_name_constructs_context_model():
    from adarpro.models.multitask_oof import create_model

    model, _ = create_model(
        torch.device("cpu"), "binary_pair_balanced_context_engagement"
    )
    assert model.use_pair_context
    assert not model.use_local


def test_context_prediction_returns_engagement_values():
    import sys

    import pandas as pd
    from torch_geometric.loader import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    import gnnadar_verb_compact as baseline

    sequence = "AANNNNNNNNNNGU"
    structure = "((..........))"
    raw = pd.DataFrame(
        {
            "pair_id": [1],
            "editing_index": [20.0],
            "sequence": [sequence],
            "structure": [structure],
            "start1": [0],
            "end1": [1],
            "start2": [12],
            "end2": [13],
            "strand": ["+"],
        }
    )
    graph = baseline.create_rna_graph(sequence, structure, 1, 1)
    attach_pair_context([graph], raw, pair_context_table(raw))
    model = MultiTaskGAT(use_pair_context=True)
    probability, label, engagement = predict_binary_and_engagement(
        model, DataLoader([graph], batch_size=1), torch.device("cpu")
    )
    assert probability.shape == label.shape == engagement.shape == (1,)
