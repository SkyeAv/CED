from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ced.model.schemas import SemanticFeatureStore, TensorRef


class SemanticAutoencoderHead(nn.Module):
    """Compress frozen text embeddings and reconstruct them through a bottleneck."""

    def __init__(self, input_dimension: int, latent_dimension: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dimension, latent_dimension)
        self.decoder = nn.Linear(latent_dimension, input_dimension)

    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.encoder(vectors)

    def reconstruction_loss(self, vectors: torch.Tensor) -> torch.Tensor:
        codes = self.encode(vectors)
        return F.mse_loss(self.decoder(codes), vectors)


def _csr_rows(indptr: torch.Tensor, owners: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    starts = indptr[owners]
    lengths = indptr[owners + 1] - starts
    repeated_starts = torch.repeat_interleave(starts, lengths)
    offsets = torch.arange(repeated_starts.numel(), device=owners.device) - torch.repeat_interleave(
        torch.cumsum(lengths, 0) - lengths, lengths
    )
    return repeated_starts + offsets, lengths


class SemanticFeatureProvider(nn.Module):
    """Frozen predicate/qualifier text vectors with trainable autoencoder heads."""

    node_vectors: torch.Tensor
    predicate_vectors: torch.Tensor
    qualifier_vectors: torch.Tensor
    qualifier_id: torch.Tensor
    qualifier_indptr: torch.Tensor

    def __init__(
        self,
        dimension: int,
        statement_dimension: int,
        relations: int,
        seed: int,
        store: SemanticFeatureStore,
        root: Path,
    ) -> None:
        super().__init__()
        if store.relations != relations:
            raise ValueError("semantic relation count does not match the model")
        self.relations = relations
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("node_vectors", torch.randn(65_536, dimension, generator=generator), persistent=False)
        self.register_buffer("predicate_vectors", self._load(store.predicates, root), persistent=False)
        self.register_buffer("qualifier_vectors", self._load(store.qualifiers, root), persistent=False)
        links = store.statement_qualifiers
        self.register_buffer("qualifier_id", self._load(links.qualifier_id, root), persistent=False)
        self.register_buffer("qualifier_indptr", self._load(links.indptr, root), persistent=False)
        if self.qualifier_id.numel() and (
            int(self.qualifier_id.min()) < 0 or int(self.qualifier_id.max()) >= store.qualifier_count
        ):
            raise ValueError("statement qualifier ID is outside the qualifier vocabulary")
        if (
            int(self.qualifier_indptr[0]) != 0
            or int(self.qualifier_indptr[-1]) != self.qualifier_id.numel()
            or bool((self.qualifier_indptr < 0).any())
            or bool((torch.diff(self.qualifier_indptr) < 0).any())
        ):
            raise ValueError("statement qualifier CSR offsets are invalid")
        self.predicate_head = SemanticAutoencoderHead(store.dimension, statement_dimension)
        self.qualifier_head = SemanticAutoencoderHead(store.dimension, statement_dimension)
        self.direction = nn.Embedding(2, statement_dimension)
        nn.init.zeros_(self.direction.weight)

    @staticmethod
    def _load(reference: TensorRef, root: Path) -> torch.Tensor:
        return torch.from_numpy(reference.load(root))

    def node_features(self, global_node_id: torch.Tensor) -> torch.Tensor:
        return self.node_vectors[global_node_id % self.node_vectors.shape[0]]

    def _qualifier_rows(self, statement_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows, lengths = _csr_rows(self.qualifier_indptr, statement_id)
        return self.qualifier_id[rows], lengths

    def statement_features(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor:
        if relation_id.numel() != statement_id.numel():
            raise ValueError("relation and statement IDs must be aligned")
        predicate_id = relation_id.remainder(self.relations)
        predicate = self.predicate_head.encode(self.predicate_vectors[predicate_id])
        direction = self.direction((relation_id >= self.relations).long())
        qualifier_ids, lengths = self._qualifier_rows(statement_id)
        qualifier_codes = self.qualifier_head.encode(self.qualifier_vectors[qualifier_ids])
        combined = predicate.new_zeros((statement_id.numel(), predicate.shape[-1]))
        messages = torch.arange(statement_id.numel(), device=statement_id.device).repeat_interleave(lengths)
        combined.index_add_(0, messages, qualifier_codes)
        qualifiers = combined / lengths.clamp_min(1).to(combined.dtype).unsqueeze(-1)
        return predicate + direction + qualifiers

    def reconstruction_loss(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor:
        if relation_id.numel() != statement_id.numel():
            raise ValueError("relation and statement IDs must be aligned")
        if not relation_id.numel():
            return torch.stack([parameter.sum() for parameter in self.parameters()]).sum() * 0
        predicate_ids = torch.unique(relation_id.remainder(self.relations))
        predicate_loss = self.predicate_head.reconstruction_loss(self.predicate_vectors[predicate_ids])
        qualifier_ids, _ = self._qualifier_rows(statement_id)
        qualifier_ids = torch.unique(qualifier_ids)
        if not qualifier_ids.numel():
            return predicate_loss
        qualifier_loss = self.qualifier_head.reconstruction_loss(self.qualifier_vectors[qualifier_ids])
        return predicate_loss + qualifier_loss
