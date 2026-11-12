from __future__ import annotations

from pathlib import Path

import torch  # pyright: ignore[reportMissingImports]
from torch import nn  # pyright: ignore[reportMissingImports]
from torch.nn import functional as F  # pyright: ignore[reportMissingImports]

from ced.model.schemas import StatementAttributeStore, TensorRef


class AttributeAutoencoderHead(nn.Module):
    """Compress frozen statement-attribute text embeddings through a trainable bottleneck."""

    def __init__(self, input_dimension: int, latent_dimension: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dimension, latent_dimension)
        self.decoder = nn.Linear(latent_dimension, input_dimension)

    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.encoder(vectors)

    def reconstruction_loss(self, vectors: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.decoder(self.encode(vectors)), vectors)


def _csr_rows(indptr: torch.Tensor, owners: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather CSR data rows for owner ids without a host synchronization.

    Returns the flat data-row indices for every owner and the per-owner row count,
    so callers can pool or scatter with one ``repeat_interleave`` and one
    ``index_add_`` regardless of how sparse the attributes are.
    """
    starts = indptr[owners]
    lengths = indptr[owners + 1] - starts
    repeated_starts = torch.repeat_interleave(starts, lengths)
    offsets = torch.arange(repeated_starts.numel(), device=owners.device) - torch.repeat_interleave(
        torch.cumsum(lengths, 0) - lengths, lengths
    )
    return repeated_starts + offsets, lengths


def _zero_loss(model: nn.Module) -> torch.Tensor:
    """A zero loss still tied to every parameter, so empty inputs keep gradients defined."""
    return torch.stack([parameter.sum() for parameter in model.parameters()]).sum() * 0


class StatementAttributeProvider(nn.Module):
    """Per-edge gate inputs from relation identity, category texts, and numeric attributes.

    Categorical attribute values are embedded once at encode time and stay frozen;
    a trainable head compresses each statement's mean-pooled category vector to the
    statement dimension. Numeric attributes enter as z-scored sparse rows gathered
    into a dense field vector and passed through a linear projection, so statements
    without a value contribute an exact zero for that field.
    """

    node_vectors: torch.Tensor
    relation_vectors: torch.Tensor
    categorical_vectors: torch.Tensor
    categorical_id: torch.Tensor
    categorical_indptr: torch.Tensor
    numeric_value: torch.Tensor
    numeric_field_id: torch.Tensor
    numeric_valid: torch.Tensor
    numeric_indptr: torch.Tensor

    def __init__(
        self,
        dimension: int,
        statement_dimension: int,
        relations: int,
        seed: int,
        store: StatementAttributeStore,
        root: Path,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("node_vectors", torch.randn(65_536, dimension, generator=generator), persistent=False)
        self.register_buffer(
            "relation_vectors",
            torch.randn(max(1, 2 * relations), statement_dimension, generator=generator),
            persistent=False,
        )
        self.register_buffer("categorical_vectors", self._load(store.categorical_vectors, root), persistent=False)
        self.register_buffer("categorical_id", self._load(store.categorical_id, root), persistent=False)
        self.register_buffer("categorical_indptr", self._load(store.categorical_indptr, root), persistent=False)
        self.register_buffer("numeric_value", self._load(store.numeric_value, root), persistent=False)
        self.register_buffer("numeric_field_id", self._load(store.numeric_field_id, root), persistent=False)
        self.register_buffer("numeric_valid", self._load(store.numeric_valid, root), persistent=False)
        self.register_buffer("numeric_indptr", self._load(store.numeric_indptr, root), persistent=False)
        if self.categorical_id.numel() and (
            int(self.categorical_id.min()) < 0 or int(self.categorical_id.max()) >= store.categorical_count
        ):
            raise ValueError("statement category ID is outside the category vocabulary")
        for name, data, fields in (
            ("category", self.categorical_id, self.categorical_indptr),
            ("numeric", self.numeric_field_id, self.numeric_indptr),
        ):
            if (
                int(fields[0]) != 0
                or int(fields[-1]) != data.numel()
                or bool((fields < 0).any())
                or bool((torch.diff(fields) < 0).any())
            ):
                raise ValueError(f"statement {name} CSR offsets are invalid")
        if self.numeric_field_id.numel() and (
            int(self.numeric_field_id.min()) < 0 or int(self.numeric_field_id.max()) >= store.numeric_fields
        ):
            raise ValueError("statement numeric field ID is outside the fitted numeric fields")
        self.categorical_head = AttributeAutoencoderHead(store.dimension, statement_dimension)
        self.numeric_encoder = nn.Linear(store.numeric_fields, statement_dimension)
        self.numeric_decoder = nn.Linear(statement_dimension, store.numeric_fields)

    @staticmethod
    def _load(reference: TensorRef, root: Path) -> torch.Tensor:
        return torch.from_numpy(reference.load(root).copy())

    def node_features(self, global_node_id: torch.Tensor) -> torch.Tensor:
        return self.node_vectors[global_node_id % self.node_vectors.shape[0]]

    def _numeric_dense(
        self, statement_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather numeric attributes into a dense (rows, fields) matrix plus COO coordinates.

        Returns the dense matrix, the per-entry owner positions and field columns,
        the masked entry values, and the entry validity mask.
        """
        rows, lengths = _csr_rows(self.numeric_indptr, statement_id)
        positions = torch.repeat_interleave(torch.arange(statement_id.numel(), device=statement_id.device), lengths)
        columns = self.numeric_field_id[rows]
        dense = torch.zeros(
            (statement_id.numel(), self.numeric_encoder.in_features),
            dtype=self.numeric_value.dtype,
            device=statement_id.device,
        )
        values = torch.zeros(rows.numel(), dtype=self.numeric_value.dtype, device=statement_id.device)
        valid = torch.zeros(rows.numel(), dtype=torch.bool, device=statement_id.device)
        if rows.numel():
            valid = self.numeric_valid[rows]
            values = self.numeric_value[rows] * valid.to(self.numeric_value.dtype)
            dense.view(-1).index_add_(0, positions * dense.shape[-1] + columns, values.to(dense.dtype))
        return dense, positions, columns, values, valid

    def statement_features(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor:
        if relation_id.numel() != statement_id.numel():
            raise ValueError("relation and statement IDs must be aligned")
        base = self.relation_vectors[relation_id.clamp(0, self.relation_vectors.shape[0] - 1)]
        rows, lengths = _csr_rows(self.categorical_indptr, statement_id)
        pooled = torch.zeros(
            (statement_id.numel(), self.categorical_vectors.shape[-1]),
            dtype=self.categorical_vectors.dtype,
            device=statement_id.device,
        )
        if rows.numel():
            owners = torch.repeat_interleave(torch.arange(statement_id.numel(), device=statement_id.device), lengths)
            pooled.index_add_(0, owners, self.categorical_vectors[self.categorical_id[rows]])
        categorical = self.categorical_head.encode(pooled / lengths.clamp_min(1).unsqueeze(-1).to(pooled.dtype))
        dense, _, _, _, _ = self._numeric_dense(statement_id)
        return base + categorical + self.numeric_encoder(dense.to(self.numeric_encoder.weight.dtype))

    def reconstruction_loss(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor | None:
        """Reconstruct the frozen category vectors and the sparse numeric rows of a batch."""
        if relation_id.numel() != statement_id.numel():
            raise ValueError("relation and statement IDs must be aligned")
        if not relation_id.numel():
            return _zero_loss(self)
        rows, _ = _csr_rows(self.categorical_indptr, statement_id)
        used = torch.unique(self.categorical_id[rows])
        loss = (
            self.categorical_head.reconstruction_loss(self.categorical_vectors[used])
            if used.numel()
            else _zero_loss(self.categorical_head)
        )
        dense, positions, columns, values, valid = self._numeric_dense(statement_id)
        if bool(valid.any()):
            predicted = self.numeric_decoder(self.numeric_encoder(dense.to(self.numeric_encoder.weight.dtype)))
            loss = loss + F.mse_loss(predicted[positions[valid], columns[valid]], values[valid].to(predicted.dtype))
        return loss
