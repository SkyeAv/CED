from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def link_loss(scores: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    if valid is None:
        valid = torch.ones_like(positive, dtype=torch.bool)
    assert valid is not None
    if scores.shape != positive.shape or scores.shape != valid.shape:
        raise ValueError("loss tensors must be aligned")
    selected = scores[valid]
    targets = positive[valid].to(scores.dtype)
    if not selected.numel():
        return scores.sum() * 0
    return F.binary_cross_entropy_with_logits(selected, targets)


class AuxiliaryAttributeDecoder(nn.Module):
    """Field-specific reconstruction over arbitrary fitted categorical and numeric attributes."""

    def __init__(self, dimension: int, categorical_counts: Mapping[int, int], numeric_fields: Sequence[int]) -> None:
        super().__init__()
        self.categorical_fields = sorted(categorical_counts)
        self.categorical_index = {field: index for index, field in enumerate(self.categorical_fields)}
        self.categorical_heads = nn.ModuleList(
            [nn.Linear(dimension, categorical_counts[field]) for field in self.categorical_fields]
        )
        self.numeric_fields = sorted(numeric_fields)
        self.numeric_index = {field: index for index, field in enumerate(self.numeric_fields)}
        self.numeric_heads = nn.ModuleList([nn.Linear(dimension, 1) for _ in self.numeric_fields])

    def _group_loss(
        self,
        states: torch.Tensor,
        fields: torch.Tensor,
        targets: torch.Tensor,
        index: dict[int, int],
        heads: nn.ModuleList,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = states.sum() * 0
        used = torch.zeros_like(targets, dtype=torch.bool) if valid is None else valid.clone()
        # One host fetch for the present fields; for the masked numeric path one
        # bincount replaces the historical per-field .any() host synchronization.
        # Iteration order stays ascending by field id, preserving the loss
        # accumulation order (and therefore CPU bitwise results).
        present = torch.unique(fields).tolist()
        counts: list[int] | None = None
        if valid is not None:
            masked = fields[valid]
            counts = torch.bincount(masked).tolist() if masked.numel() else []
        for field in present:
            head_index = index.get(int(field))
            if head_index is None:
                continue
            if counts is not None and (field >= len(counts) or counts[int(field)] == 0):
                continue
            selected = fields == field
            if valid is not None:
                selected &= valid
            used |= selected
            prediction = heads[head_index](states[selected])
            if valid is None:
                loss = loss + F.cross_entropy(prediction, targets[selected])
            else:
                loss = loss + F.mse_loss(prediction.squeeze(-1), targets[selected])
        used_total = int(used.sum())
        if used_total == 0:
            return states.sum() * 0
        return loss / max(used_total, 1)

    def forward(
        self,
        states: torch.Tensor,
        categorical_field: torch.Tensor,
        categorical_target: torch.Tensor,
        numeric_field: torch.Tensor,
        numeric_target: torch.Tensor,
        numeric_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ``_auxiliary_tensors`` concatenates categorical states before numeric
        # states; each group loss must see only its own slice or a batch that
        # mixes both feature kinds misaligns masks against states.
        categorical_count = categorical_field.shape[0]
        if states.shape[0] != categorical_count + numeric_field.shape[0]:
            raise ValueError("auxiliary states and fields are misaligned")
        category = self._group_loss(
            states[:categorical_count],
            categorical_field,
            categorical_target,
            self.categorical_index,
            self.categorical_heads,
        )
        number = self._group_loss(
            states[categorical_count:],
            numeric_field,
            numeric_target,
            self.numeric_index,
            self.numeric_heads,
            numeric_valid,
        )
        return category, number


def auxiliary_loss(
    decoder: AuxiliaryAttributeDecoder,
    states: torch.Tensor,
    categorical_field: torch.Tensor,
    categorical_target: torch.Tensor,
    numeric_field: torch.Tensor,
    numeric_target: torch.Tensor,
    numeric_valid: torch.Tensor,
    *,
    weight: float = 0.1,
) -> torch.Tensor:
    if weight < 0:
        raise ValueError("auxiliary weight must be nonnegative")
    category, number = decoder(
        states, categorical_field, categorical_target, numeric_field, numeric_target, numeric_valid
    )
    return weight * (category + number)
