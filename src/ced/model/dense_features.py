from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from ced.model.schemas import DenseFeatureStore, TensorRef
from ced.model.tensors import read_tensor

_AUTOENCODER_WEIGHT = 0.1
_OWNER_CHUNK = 4096


class FeatureProvider(Protocol):
    def node_features(self, global_node_id: torch.Tensor) -> torch.Tensor: ...

    def statement_features(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor: ...

    def reconstruction_loss(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor | None: ...


class RandomProjectionProvider(nn.Module):
    """Deterministic temporary provider; semantic providers remain swappable."""

    node_vectors: torch.Tensor
    statement_vectors: torch.Tensor

    def __init__(self, dimension: int, statement_dimension: int, relations: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("node_vectors", torch.randn(65_536, dimension, generator=generator))
        self.register_buffer(
            "statement_vectors", torch.randn(max(1, 2 * relations), statement_dimension, generator=generator)
        )

    def node_features(self, global_node_id: torch.Tensor) -> torch.Tensor:
        return self.node_vectors[global_node_id % self.node_vectors.shape[0]]

    def statement_features(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor:
        del statement_id
        return self.statement_vectors[relation_id.clamp(0, self.statement_vectors.shape[0] - 1)]

    def reconstruction_loss(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor | None:
        del relation_id, statement_id
        return None


def _ranges(starts: torch.Tensor, ends: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return concatenated half-open ranges and their original owner positions."""
    counts = ends - starts
    owners = torch.nonzero(counts > 0, as_tuple=False).squeeze(1)
    starts = starts[owners]
    counts = counts[owners]
    total = int(counts.sum().item())
    if total == 0:
        empty = starts.new_empty(0)
        return empty, empty
    offsets = torch.cumsum(counts, dim=0) - counts
    repeated_offsets = torch.repeat_interleave(offsets, counts)
    rows = torch.arange(total, device=starts.device) - repeated_offsets + torch.repeat_interleave(starts, counts)
    positions = owners[torch.repeat_interleave(torch.arange(len(owners), device=starts.device), counts)]
    return rows, positions


class SemanticNodeHead(nn.Module):
    """Attention over semantic tokens, followed by a residual bottleneck autoencoder."""

    def __init__(self, text_dimension: int, prefix_count: int, category_count: int, hidden: int = 64) -> None:
        super().__init__()
        self.prefix_embedding = nn.Embedding(prefix_count, 16, padding_idx=0)
        self.category_embedding = nn.Embedding(category_count, 16, padding_idx=0)
        self.text_projection = nn.Linear(text_dimension, hidden, bias=False)
        self.class_text_projection = nn.Linear(text_dimension, hidden, bias=False)
        self.prefix_projection = nn.Linear(16, hidden)
        self.category_projection = nn.Linear(16, hidden)
        self.topology_projection = nn.Linear(32, hidden)
        self.degree_projection = nn.Linear(1, hidden)
        self.attention = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden)
        self.bottleneck = nn.Linear(hidden, 32)
        self.residual_norm = nn.LayerNorm(32)
        self.residual = nn.Sequential(nn.Linear(32, hidden), nn.GELU(), nn.Linear(hidden, 32))
        self.output_norm = nn.LayerNorm(32)
        self.reconstruction = nn.Sequential(nn.Linear(32, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(
        self,
        text: torch.Tensor,
        class_text: torch.Tensor,
        prefix_id: torch.Tensor,
        categories: torch.Tensor,
        topology: torch.Tensor,
        degree: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack(
            (
                self.text_projection(text),
                self.class_text_projection(class_text),
                self.prefix_projection(self.prefix_embedding(prefix_id)),
                self.category_projection(categories),
                self.topology_projection(topology),
                self.degree_projection(torch.log1p(degree).unsqueeze(-1)),
            ),
            dim=1,
        )
        attended = self.attention(tokens, tokens, tokens, need_weights=False)[0]
        summary = self.attention_norm(tokens + attended).mean(dim=1)
        embedded = self.bottleneck(summary)
        embedded = self.output_norm(embedded + self.residual(self.residual_norm(embedded)))
        if not self.training:
            return embedded, embedded.new_zeros(())
        reconstructed = self.reconstruction(embedded)
        return embedded, nn.functional.mse_loss(reconstructed, summary.detach())


class SemanticFeatureProvider(nn.Module):
    """Disk-backed text lookup with learned prefix/category and statement feature heads."""

    text_owners: torch.Tensor
    text_roles: torch.Tensor
    text_parts: torch.Tensor
    text_offsets: torch.Tensor
    prefix_ids: torch.Tensor
    category_ids: torch.Tensor
    category_indptr: torch.Tensor
    node_degrees: torch.Tensor

    def __init__(
        self, document: dict[str, Any], root: Path, seed: int, *, fallback: FeatureProvider | None = None
    ) -> None:
        super().__init__()
        dense_value = document["features"]["dense"]
        self.store = DenseFeatureStore.model_validate(dense_value)
        self.root = root
        self.node_count = int(document["entities"])
        self.fallback = (
            fallback if fallback is not None else RandomProjectionProvider(32, 16, int(document["relations"]), seed)
        )
        self.node_head = SemanticNodeHead(
            self.store.dimension,
            self.store.node.prefix_vocabulary_size,
            self.store.node.category_vocabulary_size,
        )
        self.qualifier_projection = nn.Linear(self.store.dimension, 16, bias=False)
        table = pq.read_table(root / self.store.index.path, columns=["owner_index", "row", "role"])
        owners = np.array(table["owner_index"].to_numpy(zero_copy_only=False), dtype=np.int64, copy=True, order="C")
        rows = np.array(table["row"].to_numpy(zero_copy_only=False), dtype=np.int64, copy=True, order="C")
        role_ids = {
            "node": 0,
            "node_name": 0,
            "node_class": 1,
            "statement": 2,
            "qualifier": 2,
        }
        try:
            roles = np.array([role_ids[role] for role in table["role"].to_pylist()], dtype=np.int8)
        except KeyError as exception:
            raise ValueError(f"unsupported dense text role: {exception.args[0]}") from exception
        starts = [part.row_start for part in self.store.parts]
        part_ids = np.searchsorted(np.asarray(starts, dtype=np.int64), rows, side="right") - 1
        if rows.size and (part_ids.min() < 0 or part_ids.max() >= len(self.store.parts)):
            raise ValueError("dense text index references a missing cache part")
        offsets = rows - np.asarray(starts, dtype=np.int64)[part_ids] if rows.size else rows
        self.register_buffer("text_owners", torch.from_numpy(np.ascontiguousarray(owners)), persistent=False)
        self.register_buffer("text_roles", torch.from_numpy(roles), persistent=False)
        self.register_buffer("text_parts", torch.from_numpy(np.ascontiguousarray(part_ids)), persistent=False)
        self.register_buffer("text_offsets", torch.from_numpy(np.ascontiguousarray(offsets)), persistent=False)
        node = self.store.node
        self.register_buffer("prefix_ids", torch.from_numpy(node.prefix_id.load(root)), persistent=False)
        self.register_buffer("category_ids", torch.from_numpy(node.category_id.load(root)), persistent=False)
        self.register_buffer("category_indptr", torch.from_numpy(node.category_indptr.load(root)), persistent=False)
        self._part_index = -1
        self._part_vectors: torch.Tensor | None = None
        self._verified_parts: set[int] = set()
        self._autoencoder_losses: list[tuple[torch.Tensor, int]] = []
        self._head = TensorRef.model_validate(document["tensors"]["head"])
        self._tail = TensorRef.model_validate(document["tensors"]["tail"])
        self._split = TensorRef.model_validate(document["tensors"]["split"])
        self.register_buffer("node_degrees", torch.empty(0, dtype=torch.float32), persistent=False)

    def _degrees(self, device: torch.device) -> torch.Tensor:
        if self.node_degrees.numel() != self.node_count:
            head = torch.from_numpy(self._head.load(self.root)).to(device)
            tail = torch.from_numpy(self._tail.load(self.root)).to(device)
            split = torch.from_numpy(self._split.load(self.root)).to(device)
            train = split == 0
            degrees = torch.bincount(head[train], minlength=self.node_count)
            degrees += torch.bincount(tail[train], minlength=self.node_count)
            self.node_degrees = degrees.to(torch.float32)
        return self.node_degrees

    def _load_part(self, index: int) -> torch.Tensor:
        if self._part_index != index or self._part_vectors is None:
            reference = self.store.parts[index]
            if index in self._verified_parts:
                values = read_tensor(
                    self.root / reference.path, column=reference.column, dtype=reference.dtype, shape=reference.shape
                )
            else:
                values = reference.load(self.root)
                self._verified_parts.add(index)
            self._part_vectors = torch.from_numpy(values)
            self._part_index = index
        return self._part_vectors

    def _mean_text(self, owner_ids: torch.Tensor, role: int) -> torch.Tensor:
        owner_ids = owner_ids.contiguous()
        unique, inverse = torch.unique(owner_ids, return_inverse=True)
        means = unique.new_zeros((len(unique), self.store.dimension), dtype=torch.float32)
        if self.text_owners.numel() and len(unique):
            left = torch.searchsorted(self.text_owners, unique)
            right = torch.searchsorted(self.text_owners, unique, right=True)
            selected, positions = _ranges(left, right)
            matching = self.text_roles[selected] == role
            selected = selected[matching]
            positions = positions[matching]
            if selected.numel():
                part_ids = self.text_parts[selected]
                offsets = self.text_offsets[selected]
                vectors = means.new_zeros((len(selected), self.store.dimension))
                for part_index in torch.unique(part_ids).tolist():
                    mask = part_ids == part_index
                    part = self._load_part(part_index)
                    values = part[offsets[mask].cpu()].to(unique.device)
                    vectors[mask] = values
                means.index_add_(0, positions, vectors)
                counts = torch.bincount(positions, minlength=len(unique)).clamp_min(1).unsqueeze(-1)
                means = means / counts
        return means[inverse]

    def _node_features_chunk(self, global_node_id: torch.Tensor) -> torch.Tensor:
        text = self._mean_text(global_node_id, 0)
        class_text = self._mean_text(global_node_id, 1)
        prefix = self.prefix_ids[global_node_id]
        starts = self.category_indptr[global_node_id]
        ends = self.category_indptr[global_node_id + 1]
        selected, positions = _ranges(starts, ends)
        categories = self.node_head.category_embedding.weight.new_zeros((len(global_node_id), 16))
        if selected.numel():
            embedded = self.node_head.category_embedding(self.category_ids[selected])
            categories.index_add_(0, positions, embedded)
            counts = torch.bincount(positions, minlength=len(global_node_id)).clamp_min(1).unsqueeze(-1)
            categories = categories / counts
        degree = self._degrees(global_node_id.device)[global_node_id]
        embedded, loss = self.node_head(
            text, class_text, prefix, categories, self.fallback.node_features(global_node_id), degree
        )
        self._autoencoder_losses.append((loss * _AUTOENCODER_WEIGHT, len(global_node_id)))
        return embedded

    def node_features(self, global_node_id: torch.Tensor) -> torch.Tensor:
        if not self.training:
            self.reset_autoencoder_loss()
        chunks = [
            self._node_features_chunk(global_node_id[start : start + _OWNER_CHUNK])
            for start in range(0, len(global_node_id), _OWNER_CHUNK)
        ]
        if not chunks:
            return self.qualifier_projection.weight.new_empty((0, 32))
        return torch.cat(chunks)

    def statement_features(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor:
        chunks = []
        for start in range(0, len(statement_id), _OWNER_CHUNK):
            chunk = statement_id[start : start + _OWNER_CHUNK]
            text = self._mean_text(chunk + self.node_count, 2)
            fallback = self.fallback.statement_features(relation_id[start : start + _OWNER_CHUNK], chunk)
            chunks.append(fallback + self.qualifier_projection(text))
        if not chunks:
            return self.qualifier_projection.weight.new_empty((0, 16))
        return torch.cat(chunks)

    def reconstruction_loss(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor:
        loss = self.autoencoder_loss()
        statement_loss = self.fallback.reconstruction_loss(relation_id, statement_id)
        return loss if statement_loss is None else loss + statement_loss

    def reset_autoencoder_loss(self) -> None:
        self._autoencoder_losses.clear()

    def autoencoder_loss(self) -> torch.Tensor:
        if not self._autoencoder_losses:
            return self.qualifier_projection.weight.new_zeros(())
        total = torch.stack([loss * count for loss, count in self._autoencoder_losses]).sum()
        count = sum(count for _, count in self._autoencoder_losses)
        self.reset_autoencoder_loss()
        return total / max(1, count)
