from __future__ import annotations

from typing import TYPE_CHECKING

import torch  # pyright: ignore[reportMissingImports]
from torch import nn  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    pass


class RelationalAttentionEncoder(nn.Module):
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
        self.outputs = nn.ModuleList([nn.Linear(dimension, dimension) for _ in range(layers)])
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
        # Group edges by relation once per forward: a stable sort keeps the
        # original edge order inside each type, so every per-type computation
        # below sees the identical row sequence (and floating-point accumulation
        # order, forward and backward) as the historical boolean-mask loop.
        unique_types, inverse = torch.unique(edge_type_id, return_inverse=True)
        inverse = inverse.reshape(-1)
        order = torch.argsort(inverse, stable=True)
        counts = torch.bincount(inverse, minlength=unique_types.numel()).tolist()
        segments: list[tuple[int, slice]] = []
        start = 0
        for raw_type, count in zip(unique_types.tolist(), counts, strict=True):
            segments.append((int(raw_type), slice(start, start + int(count))))
            start += int(count)
        sources_sorted = edge_index[0, order]
        targets_sorted = edge_index[1, order]
        for layer_index in range(self.layers_count):
            messages = state.new_zeros(state.shape)
            for relation, segment in segments:
                # Per-type gathers mirror the reference autograd graph exactly:
                # same index vectors, same separate state/statement gathers.
                sources = sources_sorted[segment]
                targets = targets_sorted[segment]
                gate = torch.sigmoid(self.gates(statement_features[order[segment]]))
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
