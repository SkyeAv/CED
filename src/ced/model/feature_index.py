from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch

from ced.model.schemas import TensorRef


def _tensor(reference: TensorRef | dict[str, object], root: Path) -> torch.Tensor:
    ref: TensorRef = reference if isinstance(reference, TensorRef) else TensorRef.model_validate(reference)
    return torch.from_numpy(ref.load(root).copy())


def _ranges(sorted_owner: torch.Tensor, owners: torch.Tensor) -> torch.Tensor:
    left = torch.searchsorted(sorted_owner, owners, right=False)
    right = torch.searchsorted(sorted_owner, owners, right=True)
    lengths = right - left
    total = int(lengths.sum())
    if total == 0:
        return torch.empty(0, dtype=torch.long)
    repeated_left = torch.repeat_interleave(left, lengths)
    offsets = torch.arange(total) - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
    return repeated_left + offsets


class FeatureIndex:
    """Disk-backed CSR/COO lookup with Torch range selection."""

    def __init__(self, root: Path, document: dict) -> None:
        features = document["features"]
        self.node_count = document["entities"]
        self.categorical = features.get("categorical")
        self.numeric = features.get("numeric")
        if self.categorical:
            self.cat_owner = _tensor(self.categorical["owner_id"], root)
            self.cat_field = _tensor(self.categorical["field_id"], root)
            self.cat_value = _tensor(self.categorical["value_id"], root)
        if self.numeric:
            self.num_owner = _tensor(self.numeric["owner_id"], root)
            self.num_field = _tensor(self.numeric["field_id"], root)
            self.num_value = _tensor(self.numeric["value"], root)
            self.num_valid = _tensor(self.numeric["valid"], root)

    def categorical_rows(self, owners: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.categorical or owners.numel() == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty.clone(), empty.clone()
        rows = _ranges(self.cat_owner, owners)
        return rows, self.cat_field[rows], self.cat_value[rows]

    def numeric_rows(self, owners: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.numeric or owners.numel() == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty.clone(), torch.empty(0), torch.empty(0, dtype=torch.bool)
        rows = _ranges(self.num_owner, owners)
        return rows, self.num_field[rows], self.num_value[rows], self.num_valid[rows]

    def statement_owner(self, statement_ids: torch.Tensor) -> torch.Tensor:
        return statement_ids + self.node_count

    def metadata(self, root: Path) -> tuple[dict[int, int], list[int]]:
        return load_feature_metadata(root)


def load_feature_metadata(root: Path) -> tuple[dict[int, int], list[int]]:
    if not (root / "fitted_fields.parquet").exists():
        return {}, []
    categories = pq.read_table(root / "fitted_categories.parquet", columns=["field_id", "value_id"])
    fields = categories["field_id"].to_numpy(zero_copy_only=False)
    values = categories["value_id"].to_numpy(zero_copy_only=False)
    counts: dict[int, int] = {}
    for field, value in zip(fields.tolist(), values.tolist(), strict=True):
        counts[field] = max(counts.get(field, 0), int(value) + 1)
    fitted = pq.read_table(root / "fitted_fields.parquet", columns=["idx", "kind"])
    numeric = [
        int(idx)
        for idx, kind in zip(fitted["idx"].to_pylist(), fitted["kind"].to_pylist(), strict=True)
        if kind == "numeric"
    ]
    return counts, sorted(numeric)
