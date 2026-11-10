from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from ced.config.models import ConfigBase
from ced.kg.audit import validate_checksums
from ced.model.tensors import DEFAULT_COLUMN, read_tensor


class VocabularyArtifact(ConfigBase):
    path: str
    rows: int = Field(ge=0)
    sha256: str


class EntityVocabulary(ConfigBase):
    id: str
    index: int = Field(ge=0)
    type_set_id: int = Field(ge=0)
    mapping_status: str


class RelationVocabulary(ConfigBase):
    id: str
    index: int = Field(ge=0)
    inverse_of: int | None = None


class NodeTypeSetVocabulary(ConfigBase):
    index: int = Field(ge=0)
    categories: list[str]


class FeatureFieldSpec(ConfigBase):
    index: int = Field(ge=0)
    path: str
    owner_kind: Literal["node", "statement"]
    kind: Literal["categorical", "numeric", "text"]
    count: int = Field(ge=0)
    mean: float = 0
    scale: float = Field(default=1, gt=0)
    unknown_id: int = 0
    missing_id: int = 1


class SplitRecord(ConfigBase):
    statement_id: str
    group_id: str
    split: Literal["train", "validation", "test", "schema"]


TensorFormat = Literal["npy", "parquet"]


class TensorRef(ConfigBase):
    """One tensor inside a dataset artifact.

    Several refs may address different `column` values of the same Parquet file;
    the file checksum is then repeated across them, which is harmless because
    manifest checksums and Modal uploads are both keyed by path.
    """

    path: str
    column: str = DEFAULT_COLUMN
    format: TensorFormat = "npy"
    dtype: str = Field(pattern=r"^(int64|float32|bool|int8)$")
    shape: list[int]
    sha256: str

    def load(self, root: Path) -> np.ndarray:
        """Return the tensor as an owned C-contiguous array, validating the file."""
        validate_checksums(root, {self.path: self.sha256})
        values = self._read(root / self.path)
        if values.dtype != np.dtype(self.dtype) or list(values.shape) != self.shape:
            raise ValueError(f"tensor shape/dtype mismatch: {self.path}")
        if not values.flags.c_contiguous:
            raise ValueError(f"noncontiguous tensor: {self.path}")
        return values

    def _read(self, path: Path) -> np.ndarray:
        if self.format == "parquet":
            return read_tensor(path, column=self.column, dtype=self.dtype, shape=self.shape)
        if self.format == "npy":
            if not path.is_file():
                raise FileNotFoundError(f"dataset tensor is missing: {self.path}")
            return np.load(path, allow_pickle=False)
        raise ValueError(f"unsupported tensor format: {self.format}")


class TextRecord(ConfigBase):
    owner_id: int
    field_id: int
    ordinal: int = 0
    role: str = "attribute"
    language: str | None = None
    text: str


class DenseFeatureRef(ConfigBase):
    owner_id: int
    field_id: int
    row: int
    dimension: int = Field(gt=0)
    dtype: str
    encoder_key: str
    tensor: TensorRef


class DenseFeatureStore(ConfigBase):
    provider: str
    model: str
    encoder_key: str
    dimension: int = Field(gt=0)
    rows: int = Field(ge=0)
    index_path: str
    tensor: TensorRef
    checksums: dict[str, str] = Field(default_factory=dict)


class CategoricalCsr(ConfigBase):
    owner_id: TensorRef
    field_id: TensorRef
    value_id: TensorRef
    ordinal: TensorRef
    indptr: TensorRef


class NumericCoo(ConfigBase):
    owner_id: TensorRef
    field_id: TensorRef
    value: TensorRef
    valid: TensorRef


class FeatureBatch(ConfigBase):
    categorical: CategoricalCsr | None = None
    numeric: NumericCoo | None = None
    text: list[TextRecord] = Field(default_factory=list)
    dense: list[DenseFeatureRef] = Field(default_factory=list)


class NegativeBatch(ConfigBase):
    subject: TensorRef
    object: TensorRef
    valid: TensorRef


class GraphBatch(ConfigBase):
    global_node_id: TensorRef
    seed_id: TensorRef
    type_set_id: TensorRef
    edge_index: TensorRef
    relation_id: TensorRef
    edge_type_id: TensorRef
    statement_id: TensorRef
    positive_statement_id: TensorRef
    negative: NegativeBatch
    node_features: FeatureBatch = Field(default_factory=FeatureBatch)
    statement_features: FeatureBatch = Field(default_factory=FeatureBatch)


class CandidatePools(ConfigBase):
    relation_id: int
    subject_type_sets: list[int]
    object_type_sets: list[int]
    subjects: TensorRef
    objects: TensorRef


class EvaluationBatch(ConfigBase):
    split: Literal["validation", "test"]
    graph: GraphBatch
    ranking: Literal["exact", "sampled"]


class TrainingExample(ConfigBase):
    graph: GraphBatch
    split: Literal["train", "validation", "test"]


class FeatureStoreArtifact(ConfigBase):
    categorical: CategoricalCsr | None = None
    numeric: NumericCoo | None = None
    text: list[TextRecord] = Field(default_factory=list)


class DatasetManifest(ConfigBase):
    schema_version: int = 1
    dataset_id: str
    snapshot_id: str
    seed: int
    fit_split: Literal["train", "all"] = "train"
    entities: int = Field(gt=0)
    relations: int = Field(gt=0)
    statements: int = Field(ge=0)
    tensors: dict[str, TensorRef]
    feature_fields: list[FeatureFieldSpec] = Field(default_factory=list)
    features: FeatureStoreArtifact = Field(default_factory=FeatureStoreArtifact)
    batches: list[TrainingExample] = Field(default_factory=list)
    vocabularies: dict[str, VocabularyArtifact] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def topology_shapes(self) -> DatasetManifest:
        expected = {
            "head": [self.statements],
            "tail": [self.statements],
            "relation_id": [self.statements],
            "split": [self.statements],
            "type_set_id": [self.entities],
        }
        for key, shape in expected.items():
            if key not in self.tensors or self.tensors[key].shape != shape:
                raise ValueError(f"unaligned topology tensor: {key}")
        return self
