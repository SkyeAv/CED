"""Frozen pre-optimization reference encoder (do not edit).

Verbatim copy of RelationalAttentionEncoder as of commit 7deca03. The
production encoder groups edges by relation once per forward; this reference
keeps the historical per-layer boolean-mask loop so
tests/test_encoder_equivalence.py can assert bitwise-identical outputs.
"""

from __future__ import annotations

import torch  # pyright: ignore[reportMissingImports]
from torch import nn  # pyright: ignore[reportMissingImports]


class RelationalAttentionEncoderReference(nn.Module):
    """Edge-conditioned heterogeneous attention over arbitrary Biolink/Amalgam relations."""

    def __init__(self, dimension: int, statement_dimension: int, edge_types: int, layers: int = 2) -> None:
        super().__init__()
        if dimension < 2 or layers < 1 or edge_types < 1:
            raise ValueError("encoder dimensions/layers/edge types must be positive")
        self.layers_count = layers
        self.input = nn.Linear(dimension, dimension)
        self.queries = nn.ModuleList([nn.Linear(dimension, dimension) for _ in range(edge_types)])
        self.keys = nn.ModuleList([nn.Linear(dimension, dimension) for _ in range(edge_types)])
        self.values = nn.ModuleList([nn.Linear(dimension, dimension) for _ in range(edge_types)])
        self.outputs = nn.ModuleList([nn.Linear(dimension, dimension) for _ in range(edge_types)])
        self.gates = nn.Linear(statement_dimension, dimension)
        self.norms = nn.ModuleList([nn.LayerNorm(dimension) for _ in range(layers)])
        self.output = nn.LayerNorm(dimension)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_id: torch.Tensor,
        statement_features: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.dtype != torch.long or edge_type_id.dtype != torch.long:
            raise ValueError("edge indices and edge-type IDs must be int64 tensors")
        if edge_index.shape[1] != edge_type_id.numel() or statement_features.shape[0] != edge_type_id.numel():
            raise ValueError("unaligned message tensors")
        state = self.input(nodes)
        for layer_index in range(self.layers_count):
            messages = state.new_zeros(state.shape)
            for parameter_index in torch.unique(edge_type_id).tolist():
                selected = edge_type_id == parameter_index
                sources = edge_index[0, selected]
                targets = edge_index[1, selected]
                relation = int(parameter_index)
                gate = torch.sigmoid(self.gates(statement_features[selected]))
                values = self.values[relation](state[sources]) * gate
                scores = (self.queries[relation](state[targets]) * self.keys[relation](state[sources])).sum(-1)
                # Stable per-target softmax without materializing a dense target-source matrix.
                maximum = torch.full((state.shape[0],), -torch.inf, device=state.device, dtype=scores.dtype)
                maximum.scatter_reduce_(0, targets, scores, reduce="amax", include_self=True)
                weights = torch.exp(scores - maximum[targets])
                denominators = state.new_zeros(state.shape[0])
                denominators.index_add_(0, targets, weights)
                messages.index_add_(
                    0, targets, values * (weights / denominators[targets].clamp_min(1e-9)).unsqueeze(-1)
                )
            state = self.norms[layer_index](state + self.outputs[layer_index](messages))
        return self.output(state)
