"""Statement attribute edge features: encoding, provider math, and training wiring.

Why these exist: effect size and effect type live on MOKG statements as edge
attributes. They were preserved through preparation but never reached the encoder
gate, because the temporary feature provider discarded statement ids. These tests
pin the encode step, the per-edge gather math and the provider swap, so a silent
fallback to relation-only gating fails here instead of degrading training unnoticed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ced.acquisition.base import sha256_file
from ced.config.models import RuntimeConfig
from ced.kg.writer import ATTRIBUTE_SCHEMA, NODE_SCHEMA, STATEMENT_SCHEMA
from ced.model.prepare import prepare_dataset, save_tensor
from ced.model.schemas import StatementAttributeStore, TrainingExample
from ced.model.split import group_split, statement_group
from ced.model.statement_encoding import encode_statement_attributes
from ced.model.statement_features import StatementAttributeProvider
from ced.model.training import build_model, config_hash, train_step

NODES = {
    "id": ["N0", "N1", "N2", "N3"],
    "names": [[], [], [], []],
    "categories": [["biolink:Gene"], ["biolink:Gene"], ["biolink:Disease"], ["biolink:Disease"]],
    "in_taxon": [[], [], [], []],
    "provided_by": [[], [], [], []],
    "source_ids": [[], [], [], []],
    "mapping_status": ["mapped", "mapped", "mapped", "mapped"],
    "original_label": ["", "", "", ""],
    "native_ids_json": ["{}", "{}", "{}", "{}"],
    "attributes_json": ["{}", "{}", "{}", "{}"],
}
STATEMENTS = {
    "id": ["s0", "s1", "s2"],
    "source_id": ["bench"] * 3,
    "native_id": ["0", "1", "2"],
    "subject": ["N0", "N0", "N1"],
    "predicate": ["p0", "p1", "p2"],
    "object": ["N1", "N2", "N3"],
    "relation_role": ["asserted"] * 3,
    "target_eligible": [True] * 3,
    "categories": [[]] * 3,
    "qualifiers_json": ["{}"] * 3,
    "attributes_json": ["{}"] * 3,
    "publications": [[]] * 3,
    "sources_json": ["[]"] * 3,
    "original_record_json": ["{}"] * 3,
}
ATTRIBUTES = {
    "owner_kind": ["statement", "statement", "statement", "statement"],
    "owner_id": ["s0", "s0", "s1", "s2"],
    "source_id": ["bench"] * 4,
    "path": ["effect_type", "effect_size", "effect_type", "effect_size"],
    "value_type": ["string", "numeric", "string", "numeric"],
    "ordinal": [0, 0, 0, 0],
    "raw_json": ['"increases"', "2.0", '"decreases"', "4.0"],
    "text": [None, None, None, None],
    "numeric": [None, 2.0, None, 4.0],
    "provenance_json": ["{}"] * 4,
}


def write_snapshot(root: Path) -> None:
    """Emit a canonical snapshot whose statements carry effect type and size."""
    shutil.rmtree(root, ignore_errors=True)
    for name, schema, table in (
        ("nodes", NODE_SCHEMA, NODES),
        ("statements", STATEMENT_SCHEMA, STATEMENTS),
        ("attributes", ATTRIBUTE_SCHEMA, ATTRIBUTES),
    ):
        (root / name).mkdir(parents=True)
        columns = {field.name: table[field.name] for field in schema}
        pq.write_table(pa.table(columns, schema=schema), root / name / "data_0.parquet")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_hash": "bench",
                "counts": {"nodes": 4, "statements": 3, "attributes": 4},
            }
        ),
        encoding="utf-8",
    )
    checksums = {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*.parquet"))}
    (root / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")


@pytest.fixture
def attribute_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, RuntimeConfig]:
    """Prepare a dataset whose seed makes s0 a training statement."""
    monkeypatch.setattr("ced.model.prepare.require_cpu_worker", lambda runtime: None)
    runtime = RuntimeConfig.model_validate(
        {
            "remote_project": str(tmp_path),
            "artifact_root": str(tmp_path / "artifacts"),
            "cpu_workers": 4,
            "memory_limit": "4GB",
            "disk_reserve_gb": 0.5,
            "text_encoding": {"provider": "random", "dimension": 8, "batch_size": 2},
        }
    )
    group = statement_group("N0", "p0", "N1")
    settings = runtime.split
    split_settings = [settings.model_copy(update={"seed": candidate}) for candidate in range(100)]
    split = next(item for item in split_settings if group_split(group, item) == 0)
    runtime = runtime.model_copy(update={"split": split})
    write_snapshot(runtime.artifact_path("snapshots", "snapshot"))
    dataset = prepare_dataset(runtime, "snapshot")
    return dataset, runtime


def encode(runtime: RuntimeConfig, dataset: Path, monkeypatch: pytest.MonkeyPatch) -> StatementAttributeStore:
    monkeypatch.setattr("ced.model.statement_encoding.require_cpu_worker", lambda config: None)
    return encode_statement_attributes(runtime, dataset.name)


def test_encode_statement_attributes_builds_csr_stores(
    attribute_dataset: tuple[Path, RuntimeConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, runtime = attribute_dataset
    store = encode(runtime, dataset, monkeypatch)
    assert store.dimension == 8
    assert store.statement_count == 3
    assert store.categorical_count == 2  # "effect_type: decreases" sorts before "increases"
    assert store.numeric_fields == 1
    assert store.categorical_id.load(dataset).tolist() == [1, 0]
    assert store.categorical_indptr.load(dataset).tolist() == [0, 1, 2, 2]
    vectors = store.categorical_vectors.load(dataset)
    assert vectors.shape == (2, 8)
    assert np.isfinite(vectors).all()
    numeric = store.numeric_value.load(dataset)
    assert store.numeric_field_id.load(dataset).tolist()[0] == 0
    assert bool(store.numeric_valid.load(dataset)[0])
    assert int(store.numeric_indptr.load(dataset)[-1]) == int(numeric.shape[0])
    manifest = json.loads((dataset / "manifest.json").read_text())
    field = next(item for item in manifest["feature_fields"] if item["path"] == "effect_size")
    assert numeric[0] == pytest.approx((2.0 - field["mean"]) / field["scale"])

    # A second run republishes byte-identical artifacts through the shared cache.
    artifacts = [path.name for path in sorted(dataset.glob("statement-attribute-*.safetensors"))]
    before = {name: sha256_file(dataset / name) for name in artifacts}
    again = encode(runtime, dataset, monkeypatch)
    assert again.model_dump(mode="json") == store.model_dump(mode="json")
    assert {name: sha256_file(dataset / name) for name in artifacts} == before


def hand_store(root: Path) -> StatementAttributeStore:
    """A small store with known vectors for exact gather math assertions."""
    vectors = save_tensor(root, "cat-vectors", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    category_id = save_tensor(root, "cat-id", np.array([1, 0], dtype=np.int64))
    category_indptr = save_tensor(root, "cat-indptr", np.array([0, 1, 2, 2], dtype=np.int64))
    value = save_tensor(root, "num-value", np.array([2.0, -1.0], dtype=np.float32))
    field = save_tensor(root, "num-field", np.array([0, 1], dtype=np.int64))
    valid = save_tensor(root, "num-valid", np.array([True, False]))
    indptr = save_tensor(root, "num-indptr", np.array([0, 1, 1, 2], dtype=np.int64))
    return StatementAttributeStore(
        provider="random",
        model="test-model",
        encoder_key="test-key",
        dimension=2,
        statement_count=3,
        categorical_count=2,
        numeric_fields=2,
        categorical_vectors=vectors,
        categorical_id=category_id,
        categorical_indptr=category_indptr,
        numeric_value=value,
        numeric_field_id=field,
        numeric_valid=valid,
        numeric_indptr=indptr,
    )


def test_provider_gathers_pooling_and_numerics(tmp_path: Path) -> None:
    provider = StatementAttributeProvider(4, 3, 1, seed=1, store=hand_store(tmp_path), root=tmp_path)
    relation_id = torch.tensor([0, 0, 0], dtype=torch.long)
    statement_id = torch.tensor([0, 1, 2], dtype=torch.long)
    features = provider.statement_features(relation_id, statement_id)
    pooled = torch.stack([provider.categorical_vectors[1], provider.categorical_vectors[0], torch.zeros(2)])
    numeric = torch.tensor([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    expected = (
        provider.relation_vectors[relation_id]
        + provider.categorical_head.encode(pooled)
        + provider.numeric_encoder(numeric)
    )
    assert features.shape == (3, 3)
    assert torch.allclose(features, expected, atol=1e-6)
    with pytest.raises(ValueError, match="aligned"):
        provider.statement_features(torch.tensor([0]), torch.tensor([0, 1]))


def test_provider_reconstruction_loss_supervises_both_heads(tmp_path: Path) -> None:
    provider = StatementAttributeProvider(4, 3, 1, seed=1, store=hand_store(tmp_path), root=tmp_path)
    loss = provider.reconstruction_loss(torch.tensor([0, 0]), torch.tensor([0, 2], dtype=torch.long))
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    assert provider.categorical_head.encoder.weight.grad is not None
    assert provider.numeric_encoder.weight.grad is not None
    empty = provider.reconstruction_loss(torch.tensor([0]), torch.tensor([2], dtype=torch.long))
    assert empty is not None
    assert float(empty) == 0.0
    with pytest.raises(ValueError, match="aligned"):
        provider.reconstruction_loss(torch.tensor([0]), torch.tensor([0, 1], dtype=torch.long))


def test_provider_rejects_ids_outside_the_vocabulary(tmp_path: Path) -> None:
    store = hand_store(tmp_path)
    save_tensor(tmp_path, "bad-id", np.array([1, 0, 5], dtype=np.int64))
    save_tensor(tmp_path, "bad-indptr", np.array([0, 1, 2, 3], dtype=np.int64))
    document = store.model_dump(mode="json")
    document["categorical_id"] = {
        "path": "bad-id.safetensors",
        "dtype": "int64",
        "shape": [3],
        "sha256": sha256_file(tmp_path / "bad-id.safetensors"),
    }
    document["categorical_indptr"] = {
        "path": "bad-indptr.safetensors",
        "dtype": "int64",
        "shape": [4],
        "sha256": sha256_file(tmp_path / "bad-indptr.safetensors"),
    }
    validated = StatementAttributeStore.model_validate(document)
    with pytest.raises(ValueError, match="outside the category vocabulary"):
        StatementAttributeProvider(4, 3, 1, seed=1, store=validated, root=tmp_path)


def test_manifest_rejects_misaligned_statement_attributes(tmp_path: Path) -> None:
    payload = hand_store(tmp_path).model_dump(mode="json") | {"statement_count": 99}
    with pytest.raises(ValueError, match="does not match"):
        StatementAttributeStore.model_validate(payload)


def test_build_model_and_train_step_use_statement_attributes(
    attribute_dataset: tuple[Path, RuntimeConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, runtime = attribute_dataset
    encode(runtime, dataset, monkeypatch)
    document = json.loads((dataset / "manifest.json").read_text())
    example = next(item for item in document["batches"] if item["split"] == "train")
    graph = TrainingExample.model_validate(example).graph
    from ced.model.tensors import read_tensor

    reference = graph.edge_type_id
    edge_type_count = 1 + int(
        read_tensor(
            dataset / reference.path, column=reference.column, dtype=reference.dtype, shape=reference.shape
        ).max()
    )
    model = build_model(document, dataset, edge_type_count=edge_type_count, seed=7)
    assert isinstance(model.features, StatementAttributeProvider)
    identity = config_hash(document, learning_rate=1e-3, seed=7, auxiliary_weight=0.1)
    plain = config_hash({**document, "statement_attributes": None}, learning_rate=1e-3, seed=7, auxiliary_weight=0.1)
    assert identity != plain
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_step(
        model,
        optimizer,
        dataset,
        document,
        graph,
        None,
        device=torch.device("cpu"),
        auxiliary_weight=0.1,
        attribute_weight=0.1,
    )
    assert np.isfinite(loss)
