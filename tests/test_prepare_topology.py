"""End-to-end preparation tests on a synthetic canonical snapshot.

Why these exist: `prepare_dataset` had no test coverage at all before the tensor
storage rewrite. A regression in topology emission, split assignment or feature joins
would otherwise only surface during a multi-hour build or a paid Modal run, long
after the offending change landed.
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
from ced.config.models import RuntimeConfig, SplitSettings
from ced.kg.audit import validate_checksums
from ced.kg.writer import ATTRIBUTE_SCHEMA, NODE_SCHEMA, PREDICATE_SCHEMA, STATEMENT_SCHEMA
from ced.model.encoding import encode_dataset_text
from ced.model.feature_index import FeatureIndex
from ced.model.prepare import prepare_dataset
from ced.model.schemas import TensorRef
from ced.model.split import group_split, split_bounds, statement_group

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
    "id": ["s0", "s1", "s2", "s3", "s4", "s5"],
    "source_id": ["bench"] * 6,
    "native_id": ["0", "1", "2", "3", "4", "5"],
    "subject": ["N0", "N0", "N1", "N2", "N3", "N1"],
    "predicate": ["p0", "p1", "p0", "p2", "p0", "p1"],
    "object": ["N1", "N2", "N2", "N3", "N0", "N3"],
    "relation_role": ["asserted"] * 5 + ["schema"],
    "target_eligible": [True] * 6,
    "categories": [[]] * 6,
    "qualifiers_json": [
        '{"biolink:object_aspect_qualifier":"common","CED:rare_qualifier":{"value":1}}',
        '{"biolink:object_aspect_qualifier":"common"}',
        "{}",
        '{"CED:held_out_qualifier":"singleton"}',
        "{}",
        "{}",
    ],
    "attributes_json": ["{}"] * 6,
    "publications": [[]] * 6,
    "sources_json": ["[]"] * 6,
    "original_record_json": ["{}"] * 6,
}
PREDICATES = {
    "id": ["p0", "p1", "p2"],
    "source_id": ["bench"] * 3,
    "native_id": ["0", "1", "2"],
    "name": ["causes", "affects", "associated with"],
    "definition": ["causes", "affects", "associated with"],
    "attributes_json": ["{}", "{}", "{}"],
}
ATTRIBUTES = {
    "owner_kind": ["node", "node", "statement"],
    "owner_id": ["N0", "N1", "s0"],
    "source_id": ["bench"] * 3,
    "path": ["score", "status", "note"],
    "value_type": ["numeric", "string", "string"],
    "ordinal": [0, 0, 0],
    "raw_json": ["2.5", '"known"', '"unused"'],
    "text": [None, None, "payload " * 32],
    "numeric": [2.5, None, None],
    "provenance_json": ["{}"] * 3,
}


def write_snapshot(root: Path) -> None:
    """Emit a canonical snapshot with the real writer schemas and checksums."""
    shutil.rmtree(root, ignore_errors=True)
    for name, schema, table in (
        ("nodes", NODE_SCHEMA, NODES),
        ("statements", STATEMENT_SCHEMA, STATEMENTS),
        ("attributes", ATTRIBUTE_SCHEMA, ATTRIBUTES),
        ("predicates", PREDICATE_SCHEMA, PREDICATES),
    ):
        (root / name).mkdir(parents=True)
        columns = {field.name: table[field.name] for field in schema}
        pq.write_table(pa.table(columns, schema=schema), root / name / "data_0.parquet")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_hash": "bench",
                "counts": {
                    "nodes": len(NODES["id"]),
                    "statements": len(STATEMENTS["id"]),
                    "attributes": len(ATTRIBUTES["owner_kind"]),
                },
            }
        ),
        encoding="utf-8",
    )
    checksums = {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*.parquet"))}
    (root / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")


def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, RuntimeConfig]:
    monkeypatch.setattr("ced.model.prepare.require_cpu_worker", lambda runtime: None)
    runtime = RuntimeConfig.model_validate(
        {
            "remote_project": str(tmp_path),
            "artifact_root": str(tmp_path / "artifacts"),
            "cpu_workers": 4,
            "memory_limit": "4GB",
            "disk_reserve_gb": 0.5,
        }
    )
    snapshot = runtime.artifact_path("snapshots", "snapshot")
    write_snapshot(snapshot)
    dataset = prepare_dataset(runtime, "snapshot")
    return snapshot, dataset, runtime


def test_prepare_dataset_streams_topology_and_features(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Topology is one streamed Parquet COPY whose split matches the Python reference.

    Why: the SQL CASE replaced the per-row Python loop that also produced
    split-records.ndjson. If the SQL drifted from `group_split`, statements would
    move between train and held-out splits and evaluation targets would leak into
    training without any error anywhere in the pipeline.
    """
    _, dataset, runtime = prepare(tmp_path, monkeypatch)
    manifest = json.loads((dataset / "manifest.json").read_text())

    assert manifest["schema_version"] == 2
    assert set(manifest["tensors"]) == {"head", "tail", "relation_id", "split", "type_set_id"}
    for name, reference in manifest["tensors"].items():
        assert reference["path"].endswith(".safetensors"), name
    assert not (dataset / "split-records.ndjson").exists()
    assert not list(dataset.rglob("*.npy"))
    suffixes = {path.suffix for path in dataset.rglob("*") if path.is_file() and path.name != "manifest.json"}
    assert suffixes <= {".parquet", ".json", ".safetensors"}
    validate_checksums(dataset, manifest["checksums"])
    for name, reference in manifest["tensors"].items():
        loaded = TensorRef.model_validate(reference).load(dataset)
        assert loaded.shape == tuple(reference["shape"]), name

    topology = pq.read_table(dataset / "topology.parquet")
    assert topology.num_rows == 6
    assert topology.schema.names == [
        "index",
        "statement_id",
        "group_id",
        "split",
        "head",
        "relation_id",
        "tail",
    ]

    statement_ids = topology.column("statement_id").to_pylist()
    expected = []
    for name in statement_ids:
        position = STATEMENTS["id"].index(name)
        expected.append(
            group_split(
                statement_group(
                    STATEMENTS["subject"][position], STATEMENTS["predicate"][position], STATEMENTS["object"][position]
                ),
                runtime.split,
                schema=STATEMENTS["relation_role"][position] == "schema",
            )
        )
    assert topology.column("split").to_pylist() == expected
    assert expected[statement_ids.index("s5")] == 3

    # Entity indices are dense and match the head/tail endpoints of each statement.
    entities = pq.read_table(dataset / "entities.parquet", columns=["id", "type_set_id"])
    assert entities.column("id").to_pylist() == sorted(entities.column("id").to_pylist())
    assert set(topology.column("head").to_pylist()) <= set(range(len(NODES["id"])))
    assert entities.column("type_set_id").to_pylist() == [1, 1, 0, 0]

    # Shard sampling ran end to end from the loaded arrays.
    assert manifest["batches"], "no training batches were emitted"
    graph = manifest["batches"][0]["graph"]
    assert graph["edge_index"]["shape"][0] == 2
    assert graph["edge_index"]["path"].endswith(".safetensors")
    assert graph["negative"]["subject"]["shape"][1] == runtime.split.negatives


def test_prepare_dataset_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two preparations of the same snapshot produce identical artifacts.

    Why: dataset ids are content hashes, so a nondeterministic emission would
    either poison resume paths or silently produce different training data for
    the same declared snapshot.
    """
    _snapshot, dataset, _ = prepare(tmp_path, monkeypatch)
    first = {
        str(path.relative_to(dataset)): sha256_file(path)
        for path in sorted(dataset.rglob("*"))
        if path.is_file() and path.suffix in {".parquet", ".safetensors"} and "batches" not in path.parts
    }
    shutil.rmtree(dataset)
    _, again, _ = prepare(tmp_path, monkeypatch)
    second = {
        str(path.relative_to(again)): sha256_file(path)
        for path in sorted(again.rglob("*"))
        if path.is_file() and path.suffix in {".parquet", ".safetensors"} and "batches" not in path.parts
    }
    assert first == second


def test_sql_split_case_matches_python_reference() -> None:
    """The SQL split CASE and group_split agree for the default and shifted ratios.

    Why: this is the equivalence that makes the streaming rewrite safe. The float
    comparison was replaced by exact integer thresholds, and DuckDB compares the
    hex digest while Python compares bytes; both must land on the same split for
    every group or train/test isolation breaks.
    """
    import duckdb

    settings = SplitSettings(seed=20261110)
    train_bound, valid_bound = split_bounds(settings)
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            """
            SELECT g.group_id,
              CASE WHEN g.schema THEN 3
                WHEN ('0x' || substr(sha256(? || chr(0) || g.group_id), 1, 16))::UBIGINT < ('0x' || ?)::UBIGINT THEN 0
                WHEN ('0x' || substr(sha256(? || chr(0) || g.group_id), 1, 16))::UBIGINT < ('0x' || ?)::UBIGINT THEN 1
                ELSE 2 END AS split
            FROM (
              SELECT sha256(x || chr(0) || 'p' || chr(0) || y) AS group_id, b AS schema
              FROM (VALUES ('a', 'z', false), ('b', 'z', false), ('c', 'z', false), ('d', 'z', false), ('schema', 'z', true)) AS t(x, y, b)
            ) g
            """,
            [str(settings.seed), train_bound, str(settings.seed), valid_bound],
        ).fetchall()
    finally:
        connection.close()
    assert len(rows) >= 5
    for group_id, split in rows:
        assert group_split(group_id, settings, schema=split == 3) == split


def test_feature_tensor_conversion_rejects_nulls(tmp_path: Path) -> None:
    from ced.model.feature_store import _convert_parquet_tensor, _read_parquet_tensor

    source = tmp_path / "nulls.parquet"
    pq.write_table(pa.table({"value": pa.array([1, None], pa.int64())}), source)
    with pytest.raises(ValueError, match="contains nulls"):
        _read_parquet_tensor(source, "value", "int64")
    with pytest.raises(ValueError, match="contains nulls"):
        _convert_parquet_tensor(source, "value", "int64", tmp_path / "nulls.safetensors")
    assert not list(tmp_path.glob("*.tmp-*"))


def test_write_indptr_covers_owners_without_rows(tmp_path: Path) -> None:
    """CSR offsets must give every owner an explicit entry, including empty ones.

    Why: FeatureIndex selects rows with searchsorted over the owner column, and a
    sparse or shifted indptr would attribute one entity's attributes to another.
    Owners with no attributes must still advance the offset by zero.
    """
    import duckdb

    from ced.model.feature_store import write_indptr

    source = tmp_path / "categorical.parquet"
    pq.write_table(
        pa.table(
            {
                "owner_index": pa.array([0, 0, 2, 2, 5], pa.int64()),
                "field_id": pa.array([1, 2, 1, 3, 9], pa.int64()),
            }
        ),
        source,
        compression="zstd",
    )
    connection = duckdb.connect()
    try:
        reference = write_indptr(connection, source, 7, tmp_path / "categorical-indptr.safetensors")
    finally:
        connection.close()
    assert reference.shape == [8]
    assert reference.load(tmp_path).tolist() == [0, 2, 2, 4, 4, 4, 5, 5]


def test_write_indptr_rejects_owners_outside_range(tmp_path: Path) -> None:
    """An owner outside [0, owners] must fail loudly, never produce a short CSR.

    Why: that mismatch means the owner index and the entity vocabulary disagree.
    Silently dropping those rows would delete attributes from the feature store
    without any trace downstream.
    """
    import duckdb

    from ced.model.feature_store import write_indptr

    source = tmp_path / "categorical.parquet"
    pq.write_table(
        pa.table({"owner_index": pa.array([0, 9], pa.int64()), "field_id": pa.array([1, 2], pa.int64())}),
        source,
        compression="zstd",
    )
    connection = duckdb.connect()
    try:
        with pytest.raises(ValueError, match="CSR invariants"):
            write_indptr(connection, source, 5, tmp_path / "categorical-indptr.safetensors")
    finally:
        connection.close()


def test_prepare_dataset_emits_ordered_feature_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Feature columns are owner-ordered safetensors with a valid CSR index.

    Why: the ORDER BY in each COPY is load-bearing. FeatureIndex selects rows by
    binary search over the owner column, so an unordered file would silently return
    the wrong attributes for every owner. This pins the ordering, the dtype of each
    column and the row totals against the fixture.
    """
    _, dataset, _runtime = prepare(tmp_path, monkeypatch)
    manifest = json.loads((dataset / "manifest.json").read_text())
    features = manifest["features"]

    categorical = features["categorical"]
    for name, reference in categorical.items():
        assert reference["path"].endswith(".safetensors"), name
        assert reference["dtype"] == "int64", name
    assert categorical["owner_id"]["path"] == "categorical-owner_index.safetensors"
    assert categorical["indptr"]["path"] == "categorical-indptr.safetensors"
    assert not list(dataset.rglob("*.partial.parquet"))

    owners = manifest["entities"] + manifest["statements"]
    owner_column = TensorRef.model_validate(categorical["owner_id"]).load(dataset)
    indptr = TensorRef.model_validate(categorical["indptr"]).load(dataset)
    assert np.diff(owner_column).min(initial=0) >= 0
    assert indptr.shape == (owners + 1,)
    assert indptr[0] == 0
    assert indptr[-1] == owner_column.shape[0]
    assert np.all(np.diff(indptr) >= 0)

    # Fixture: one numeric event on N0 (entity idx 0), one categorical on N1 (idx 1),
    # one long text event on statement s0, so the CSR has a single step at owner 2.
    assert owner_column.tolist() == [1]
    assert indptr.tolist() == [0, 0, 1] + [1] * (owners - 2)

    numeric = features["numeric"]
    assert numeric["value"]["dtype"] == "float32"
    assert numeric["valid"]["dtype"] == "bool"
    assert TensorRef.model_validate(numeric["owner_id"]).load(dataset).tolist() == [0]
    assert TensorRef.model_validate(numeric["valid"]).load(dataset).tolist() == [True]
    value = TensorRef.model_validate(numeric["value"]).load(dataset)
    assert np.isfinite(value).all()

    text_rows = 0
    for owner in ("node", "statement"):
        table = pq.read_table(dataset / f"{owner}-features.parquet")
        text_rows += sum(1 for kind in table.column("kind").to_pylist() if kind == "text")
    assert text_rows == 1

    index = FeatureIndex(dataset, manifest)
    rows, fields, values = index.categorical_rows(torch.tensor([1], dtype=torch.long))
    assert rows.tolist() == [0]
    assert len(values) == 1
    assert fields.shape == rows.shape
    assert index.numeric_rows(torch.tensor([0], dtype=torch.long))[0].tolist() == [0]


def test_encode_dataset_text_resolves_every_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """encode-text resolves every text feature through the append-only cache.

    Why: the rewrite moved missing-text resolution from a Python dict into a DuckDB
    anti-join on sha256(text). If that digest disagreed with Python's text_key, or
    the join dropped rows, text features would vanish from the dataset silently and
    the model would train on zeros.
    """
    _, dataset, _runtime = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr("ced.model.encoding.require_cpu_worker", lambda config: None)
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

    store = encode_dataset_text(runtime, dataset.name)
    assert store.rows == 9
    assert store.dimension == 8
    index = pq.read_table(dataset / store.index_path)
    assert index.num_rows == 9
    assert len(set(index.column("row").to_pylist())) == 6
    assert index.column("role").to_pylist().count("node_class") == 4
    assert len(store.parts) == 1
    assert all(reference.path.endswith(".safetensors") for reference in store.parts)
    assert store.parts[0].load(dataset).shape == (12, 8)
    document = json.loads((dataset / "manifest.json").read_text())
    semantic = document["semantic"]
    assert semantic["model"] == _runtime.text_encoding.model
    assert semantic["relations"] == 3
    assert semantic["qualifier_count"] == 3
    assert TensorRef.model_validate(semantic["predicates"]).load(dataset).shape == (3, 8)
    assert TensorRef.model_validate(semantic["qualifiers"]).load(dataset).shape == (3, 8)
    qualifier_text = pq.read_table(dataset / "qualifier-text.parquet").column("text").to_pylist()
    assert any("CED:rare_qualifier" in text for text in qualifier_text)
    assert any("CED:held_out_qualifier" in text for text in qualifier_text)
    assert TensorRef.model_validate(semantic["statement_qualifiers"]["indptr"]).load(dataset).shape == (7,)
    parts_before = [reference.path for reference in store.parts]
    digest_before = sha256_file(dataset / store.index_path)
    semantic_digest_before = sha256_file(dataset / "qualifier-embeddings.safetensors")

    # A second run encodes nothing new: no extra parts and a byte-identical index.
    again = encode_dataset_text(runtime, dataset.name)
    assert again.rows == store.rows
    assert [reference.path for reference in again.parts] == parts_before
    assert sha256_file(dataset / again.index_path) == digest_before
    assert sha256_file(dataset / "qualifier-embeddings.safetensors") == semantic_digest_before
    assert not (dataset / "text-records.partial.parquet").exists()
    assert not (dataset / "text-missing.partial.parquet").exists()


def test_prepared_vocabulary_loads_training_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the real producer, not a hand-written vocabulary with a different schema."""
    from ced.model.feature_index import load_feature_metadata

    _, _, runtime = prepare(tmp_path, monkeypatch)
    # Ensure s0 trains, so both N0's numeric and N1's categorical fields are fitted.
    group = statement_group("N0", "p0", "N1")
    seed = next(seed for seed in range(100) if group_split(group, runtime.split.model_copy(update={"seed": seed})) == 0)
    runtime = runtime.model_copy(update={"split": runtime.split.model_copy(update={"seed": seed})})
    dataset = prepare_dataset(runtime, "snapshot")
    counts, numeric = load_feature_metadata(dataset)
    fields = pq.read_table(dataset / "fitted_fields.parquet").to_pylist()
    categories = pq.read_table(dataset / "fitted_categories.parquet").to_pylist()
    assert categories
    expected = {}
    for category in categories:
        field = next(field for field in fields if field["idx"] == category["field_id"])
        assert field["kind"] == "categorical"
        assert field["owner_kind"] == category["owner_kind"]
        assert field["path"] == category["path"]
        expected[field["idx"]] = max(expected.get(field["idx"], 0), category["value_id"] + 1)
    assert counts == expected
    assert numeric == sorted(field["idx"] for field in fields if field["kind"] == "numeric")


def test_semantic_encoding_features_and_transfer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(NODES, "id", ["CED:N0", "X:N1", "X:N2", "X:N3"])
    monkeypatch.setitem(NODES, "names", [["zero"], ["one"], ["two"], ["three"]])
    monkeypatch.setitem(STATEMENTS, "subject", ["CED:N0", "CED:N0", "X:N1", "X:N2", "X:N3", "X:N1"])
    monkeypatch.setitem(STATEMENTS, "object", ["X:N1", "X:N2", "X:N2", "X:N3", "CED:N0", "X:N3"])
    monkeypatch.setitem(ATTRIBUTES, "owner_id", ["CED:N0", "X:N1", "s0"])
    monkeypatch.setitem(
        STATEMENTS,
        "qualifiers_json",
        ['{"AMALGAM:population_context_qualifier":"CED:Family_1"}', "{}", "{}", "{}", "{}", "{}"],
    )
    _, dataset, runtime = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr("ced.model.encoding.require_cpu_worker", lambda config: None)
    runtime = runtime.model_copy(
        update={
            "text_encoding": runtime.text_encoding.model_copy(
                update={"provider": "random", "dimension": 8, "batch_size": 2}
            )
        }
    )
    store = encode_dataset_text(runtime, dataset.name)
    index = pq.read_table(dataset / store.index.path)
    roles = index["role"].to_pylist()
    assert roles.count("node_name") == 4
    assert roles.count("node_class") == 4
    assert roles.count("qualifier") == 1
    assert roles.count("statement") == 1
    qualifier = next(row for row in index.to_pylist() if row["role"] == "qualifier")
    assert qualifier["text"] == "AMALGAM:population_context_qualifier: CED:Family_1"

    artifact = (dataset / store.index.path).parent
    prefixes = pq.read_table(artifact / "prefix-vocabulary.parquet")["value"].to_pylist()
    categories = pq.read_table(artifact / "category-vocabulary.parquet")["value"].to_pylist()
    assert prefixes == ["CED", "X"]
    assert categories == ["biolink:Disease", "biolink:Gene"]
    assert store.node.prefix_id.load(dataset).shape == (4,)
    assert store.node.category_indptr.load(dataset).shape == (5,)

    from ced.model.dense_features import SemanticFeatureProvider
    from ced.model.training import build_model

    document = json.loads((dataset / "manifest.json").read_text())
    model = build_model(document, dataset, edge_type_count=4, seed=3)
    monkeypatch.setattr("ced.model.dense_features._OWNER_CHUNK", 2)
    provider = model.features
    assert isinstance(provider, SemanticFeatureProvider)
    nodes = torch.arange(4, dtype=torch.long)
    embedded = provider.node_features(nodes)
    statements = torch.arange(6, dtype=torch.long)
    statement_features = provider.statement_features(torch.zeros(6, dtype=torch.long), statements)
    assert embedded.shape == (4, 32)
    assert statement_features.shape == (6, 16)
    assert torch.isfinite(embedded).all()
    assert torch.isfinite(statement_features).all()
    class_text = provider._mean_text(nodes, 1)
    assert torch.equal(class_text[0], class_text[1])
    assert torch.equal(class_text[2], class_text[3])
    assert not torch.equal(class_text[0], class_text[2])
    loss = (
        embedded.square().mean()
        + statement_features.square().mean()
        + provider.reconstruction_loss(torch.zeros(6, dtype=torch.long), statements)
    )
    loss.backward()
    from ced.model.semantic import SemanticFeatureProvider as StatementSemanticFeatureProvider

    statement_provider = provider.fallback
    assert isinstance(statement_provider, StatementSemanticFeatureProvider)
    for parameter in (
        provider.node_head.text_projection.weight,
        provider.node_head.class_text_projection.weight,
        provider.node_head.prefix_embedding.weight,
        provider.node_head.category_embedding.weight,
        provider.node_head.degree_projection.weight,
        provider.qualifier_projection.weight,
        statement_provider.predicate_head.encoder.weight,
        statement_provider.qualifier_head.encoder.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    tensors = {name: TensorRef.model_validate(value).load(dataset) for name, value in document["tensors"].items()}
    train = tensors["split"] == 0
    expected_degree = np.bincount(np.concatenate((tensors["head"][train], tensors["tail"][train])), minlength=4)
    np.testing.assert_array_equal(provider._degrees(torch.device("cpu")).numpy(), expected_degree)

    from ced.modal_plan import dataset_transfer_plan

    for mode in ("train", "evaluate"):
        shared = dataset_transfer_plan(dataset, runtime, mode).shared
        assert store.index.path in shared
        assert store.node.prefix_id.path in shared
        assert all(part.path in shared for part in store.parts)
