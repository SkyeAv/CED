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


class TensorRef(ConfigBase):
    """One tensor inside a dataset artifact.

    Several refs may address different `column` entries of one safetensors file;
    the file checksum is then repeated across them, which is harmless because
    manifest checksums and Modal uploads are both keyed by path.
    """

    path: str
    column: str = DEFAULT_COLUMN
    dtype: str = Field(pattern=r"^(int64|float32|bool|int8)$")
    shape: list[int]
    sha256: str

    def load(self, root: Path) -> np.ndarray:
        """Return the tensor as an owned C-contiguous array.

        `read_tensor` enforces the checksum, dtype, shape and contiguity, so a
        mismatched or truncated artifact raises instead of being silently reshaped
        into wrong indices.
        """
        validate_checksums(root, {self.path: self.sha256})
        return read_tensor(root / self.path, column=self.column, dtype=self.dtype, shape=self.shape)


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
    parts: list[TensorRef] = Field(default_factory=list)
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


class IsolationTreePruningReport(ConfigBase):
    max_component_nodes: int = Field(ge=0)
    input_nodes: int = Field(ge=0)
    input_statements: int = Field(ge=0)
    pruned_components: int = Field(ge=0)
    pruned_nodes: int = Field(ge=0)
    pruned_statements: int = Field(ge=0)


class QualifierStatementCsr(ConfigBase):
    statement_id: TensorRef
    qualifier_id: TensorRef
    indptr: TensorRef


class SemanticFeatureStore(ConfigBase):
    provider: str
    model: str
    revision: str | None = None
    encoder_key: str
    dimension: int = Field(gt=0)
    relations: int = Field(gt=0)
    qualifier_count: int = Field(ge=0)
    predicate_text_path: str
    qualifier_text_path: str
    predicates: TensorRef
    qualifiers: TensorRef
    statement_qualifiers: QualifierStatementCsr

    @model_validator(mode="after")
    def semantic_shapes(self) -> SemanticFeatureStore:
        if self.predicates.shape != [self.relations, self.dimension]:
            raise ValueError("predicate embedding tensor does not match the relation vocabulary")
        if self.qualifiers.shape != [self.qualifier_count, self.dimension]:
            raise ValueError("qualifier embedding tensor does not match the qualifier vocabulary")
        links = self.statement_qualifiers
        if links.statement_id.shape != links.qualifier_id.shape or len(links.statement_id.shape) != 1:
            raise ValueError("statement qualifier tensors must be aligned vectors")
        return self


class DatasetManifest(ConfigBase):
    schema_version: int = 2
    dataset_id: str
    snapshot_id: str
    seed: int
    pruning: IsolationTreePruningReport | None = None
    fit_split: Literal["train", "all"] = "train"
    entities: int = Field(gt=0)
    relations: int = Field(gt=0)
    statements: int = Field(ge=0)
    tensors: dict[str, TensorRef]
    feature_fields: list[FeatureFieldSpec] = Field(default_factory=list)
    features: FeatureStoreArtifact = Field(default_factory=FeatureStoreArtifact)
    semantic: SemanticFeatureStore | None = None
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
        if self.semantic is not None:
            if self.semantic.relations != self.relations:
                raise ValueError("semantic relation count does not match the dataset")
            if self.semantic.statement_qualifiers.indptr.shape != [self.statements + 1]:
                raise ValueError("statement qualifier CSR does not match the dataset")
        return self
