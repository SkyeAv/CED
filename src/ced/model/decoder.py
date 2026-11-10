from __future__ import annotations

import torch  # pyright: ignore[reportMissingImports]
from torch import nn  # pyright: ignore[reportMissingImports]


class ComplexDecoder(nn.Module):
    """Directed ComplEx scoring over encoder states and arbitrary relation IDs."""

    def __init__(self, dimension: int, relations: int) -> None:
        super().__init__()
        if dimension < 2 or relations < 1:
            raise ValueError("decoder dimensions/relations must be positive")
        self.projection = nn.Linear(dimension, 2 * dimension)
        self.relations = nn.Embedding(2 * relations, 2 * dimension)
        nn.init.xavier_uniform_(self.relations.weight)

    def complex_states(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projection(nodes)
        return projected[..., : projected.shape[-1] // 2], projected[..., projected.shape[-1] // 2 :]

    def score(self, subjects: torch.Tensor, relations: torch.Tensor, objects: torch.Tensor) -> torch.Tensor:
        if relations.dtype != torch.long:
            raise ValueError("relation IDs must be int64 tensors")
        subject_real, subject_imaginary = self.complex_states(subjects)
        object_real, object_imaginary = self.complex_states(objects)
        relation_real, relation_imaginary = self.relations(relations).chunk(2, dim=-1)
        return (subject_real * relation_real - subject_imaginary * relation_imaginary).mul(object_real).sum(-1) + (
            subject_real * relation_imaginary + subject_imaginary * relation_real
        ).mul(object_imaginary).sum(-1)
