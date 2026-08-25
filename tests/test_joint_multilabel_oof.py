import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

from adarpro.models.joint_multilabel_oof import (
    TISSUES,
    JointMultiLabelGAT,
    WeightedSiteDataset,
    aggregate_sample_weights,
    build_multilabel_training_table,
    train_epoch,
)


def toy_raw() -> pd.DataFrame:
    rows = []
    for context, level in zip(TISSUES, (0.0, 20.0, 8.0, 30.0)):
        rows.append(
            {
                "pair_local_key": "p:1",
                "pair_id": "p",
                "context": context,
                "editing_index": level,
                "sequence": "AACGUU",
                "structure": "((..))",
                "local_position": 1,
            }
        )
    return pd.DataFrame(rows)


def test_multilabel_table_masks_intermediate_context():
    sites, binary, labels, graph_indices, context_indices = (
        build_multilabel_training_table(toy_raw())
    )
    assert len(sites) == 1
    assert len(binary) == 3
    assert labels.tolist() == [[0.0, 1.0, 0.0, 1.0]]
    assert graph_indices.tolist() == [0, 0, 0]
    assert context_indices.tolist() == [0, 1, 3]


def test_aggregate_sample_weights_preserves_repeated_draws():
    graph_indices, weights = aggregate_sample_weights(
        np.array([0, 1, 1, 2]),
        np.array([3, 3, 4]),
        np.array([0, 1, 2]),
    )
    assert graph_indices.tolist() == [3, 4]
    assert weights.sum() == 4
    assert weights[0].tolist() == [1.0, 2.0, 0.0, 0.0]
    assert weights[1].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_multilabel_table_rejects_conflicting_site_identity():
    raw = pd.concat([toy_raw(), toy_raw().assign(sequence="AACGUA")])
    with pytest.raises(RuntimeError, match="inconsistent intrinsic identity"):
        build_multilabel_training_table(raw)


def test_weighted_multilabel_batch_trains():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    import gnnadar_verb_compact as baseline

    graphs = [baseline.create_rna_graph("AACGUU", "((..))", 1, 0) for _ in range(2)]
    labels = np.asarray([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.float32)
    weights = np.asarray([[1, 2, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
    dataset = WeightedSiteDataset(graphs, np.asarray([0, 1]), labels, weights)
    model = JointMultiLabelGAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_epoch(
        model, DataLoader(dataset, batch_size=2), optimizer, torch.device("cpu")
    )
    assert loss > 0
