"""Low-capacity, jointly trained multi-view ADARPro architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool

from adarpro.models.multitask_oof import LOCAL_RADIUS


@dataclass(frozen=True)
class UnifiedVariant:
    """Immutable architectural choices for a preregistered unified variant."""

    name: str
    anchor: str = "target"
    shifted_local: bool = False
    auxiliary_weight: float = 0.25


VARIANTS = {
    "unified_target_local_interaction_aux_v1": UnifiedVariant(
        name="unified_target_local_interaction_aux_v1"
    ),
    "unified_target_local_interaction_no_aux_v1": UnifiedVariant(
        name="unified_target_local_interaction_no_aux_v1",
        auxiliary_weight=0.0,
    ),
    "unified_anchor_local_interaction_aux_v1": UnifiedVariant(
        name="unified_anchor_local_interaction_aux_v1",
        anchor="first",
    ),
    "unified_target_shifted_local_interaction_aux_v1": UnifiedVariant(
        name="unified_target_shifted_local_interaction_aux_v1",
        shifted_local=True,
    ),
}


class UnifiedADARPro(nn.Module):
    """Joint global, focal and helix-aligned local representation model.

    The auxiliary global head is used only as a training objective. Inference
    always uses ``final_head`` after representation-level fusion; no component
    probability or logit is averaged into the final prediction.
    """

    def __init__(
        self,
        *,
        input_dim: int = 8,
        hidden_dim: int = 32,
        heads: int = 4,
        view_dim: int = 32,
        fusion_dim: int = 64,
        dropout: float = 0.2,
        anchor: str = "target",
    ) -> None:
        super().__init__()
        if anchor not in {"target", "first"}:
            raise ValueError(f"unsupported focal anchor: {anchor}")
        self.anchor = anchor
        self.gat_layers = nn.ModuleList(
            [
                GATConv(input_dim, hidden_dim, heads=heads, concat=True),
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True),
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True),
            ]
        )
        graph_dim = hidden_dim * heads
        self.batch_norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(graph_dim) for _ in range(3)]
        )
        self.dropout = nn.Dropout(dropout)
        self.local_convolution = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(graph_dim, view_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.focal_projection = nn.Sequential(
            nn.Linear(graph_dim, view_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.local_projection = nn.Sequential(
            nn.Linear(64, view_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        # Three views plus all pairwise, elementwise representation interactions.
        self.fusion = nn.Sequential(
            nn.Linear(6 * view_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.final_head = nn.Linear(fusion_dim, 1)
        self.global_auxiliary_head = nn.Linear(graph_dim, 1)

    def _graph_views(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        node_state = data.x
        for layer, normalization in zip(self.gat_layers, self.batch_norm_layers):
            node_state = layer(node_state, data.edge_index)
            node_state = self.dropout(normalization(F.relu(node_state)))
        global_state = global_mean_pool(node_state, data.batch)
        if self.anchor == "target":
            target_mask = data.x[:, 7] > 0.5
            target_batch = data.batch[target_mask]
            expected = torch.arange(
                global_state.shape[0],
                device=target_batch.device,
                dtype=target_batch.dtype,
            )
            if target_batch.shape != expected.shape or not torch.equal(
                target_batch, expected
            ):
                raise RuntimeError(
                    "each graph must contain exactly one ordered target node"
                )
            focal_state = node_state[target_mask]
        else:
            focal_state = node_state[data.ptr[:-1]]
        return global_state, focal_state

    def _local_view(self, data) -> torch.Tensor:
        local_map = self.local_convolution(data.local_window)
        return torch.cat(
            [
                local_map[:, :, LOCAL_RADIUS],
                F.adaptive_max_pool1d(local_map, 1).squeeze(-1),
            ],
            dim=1,
        )

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        global_state, focal_state = self._graph_views(data)
        local_state = self._local_view(data)
        global_view = self.global_projection(global_state)
        focal_view = self.focal_projection(focal_state)
        local_view = self.local_projection(local_state)
        fused = torch.cat(
            [
                global_view,
                focal_view,
                local_view,
                global_view * focal_view,
                global_view * local_view,
                focal_view * local_view,
            ],
            dim=1,
        )
        final_logit = self.final_head(self.fusion(fused)).view(-1)
        auxiliary_global_logit = self.global_auxiliary_head(global_state).view(-1)
        return final_logit, auxiliary_global_logit


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable parameters for provenance."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

