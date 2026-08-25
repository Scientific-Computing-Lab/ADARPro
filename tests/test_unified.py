from pathlib import Path

import torch
from torch_geometric.data import Batch

from adarpro.models.multitask_oof import local_window
from adarpro.models.unified import VARIANTS, UnifiedADARPro, parameter_count
from adarpro.models.unified_oof import train_unified_epoch


def example_batch() -> Batch:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    import gnnadar_verb_compact as baseline

    graphs = []
    for label in (0, 1):
        graph = baseline.create_rna_graph("AACGUU", "((..))", 1, label)
        graph.local_window = local_window("AACGUU", "((..))", 1)
        graph.binary_label = torch.tensor([float(label)])
        graphs.append(graph)
    return Batch.from_data_list(graphs)


def test_unified_model_emits_one_final_and_one_auxiliary_logit():
    batch = example_batch()
    model = UnifiedADARPro()
    model.eval()
    with torch.no_grad():
        final, auxiliary = model(batch)
    assert final.shape == (2,)
    assert auxiliary.shape == (2,)
    assert parameter_count(model) < 100_000


def test_unified_training_backpropagates_through_all_views():
    batch = example_batch()
    model = UnifiedADARPro()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_unified_epoch(model, [batch], optimizer, 0.25, torch.device("cpu"))
    required = (
        model.gat_layers[0].lin.weight,
        model.local_convolution[0].weight,
        model.global_projection[0].weight,
        model.focal_projection[0].weight,
        model.local_projection[0].weight,
        model.fusion[0].weight,
        model.final_head.weight,
        model.global_auxiliary_head.weight,
    )
    assert all(parameter.grad is not None for parameter in required)
    assert all(torch.isfinite(parameter.grad).all() for parameter in required)


def test_anchor_control_is_parameter_matched():
    target = UnifiedADARPro(anchor="target")
    anchor = UnifiedADARPro(anchor="first")
    assert parameter_count(target) == parameter_count(anchor)


def test_preregistered_variants_are_fully_specified():
    assert VARIANTS["unified_target_local_interaction_aux_v1"].auxiliary_weight == 0.25
    assert VARIANTS["unified_target_local_interaction_no_aux_v1"].auxiliary_weight == 0.0
    assert VARIANTS["unified_anchor_local_interaction_aux_v1"].anchor == "first"
    assert VARIANTS[
        "unified_target_shifted_local_interaction_aux_v1"
    ].shifted_local

