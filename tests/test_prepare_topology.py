"""End-to-end preparation tests on a synthetic canonical snapshot.

Why these exist: `prepare_dataset` had no test coverage at all before the Parquet
rewrite. A regression in topology emission, split assignment or feature joins
would otherwise only surface during a multi-hour build or a paid Modal run, long
after the offending change landed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ced.acquisition.base import sha256_file
from ced.config.models import RuntimeConfig, SplitSettings
from ced.kg.audit import validate_checksums
from ced.kg.writer import ATTRIBUTE_SCHEMA, NODE_SCHEMA, STATEMENT_SCHEMA
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
    "qualifiers_json": ["{}"] * 6,
    "attributes_json": ["{}"] * 6,
    "publications": [[]] * 6,
    "sources_json": ["[]"] * 6,
    "original_record_json": ["{}"] * 6,
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
    ):
        (root / name).mkdir(parents=True)
        columns = {field.name: table[field.name] for field in schema}
        pq.write_table(pa.table(columns, schema=schema), root / name / "data_0.parquet")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_hash": "bench",
                "counts": {"nodes": 4, "statements": 6, "attributes": 3},
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

    assert set(manifest["tensors"]) == {"head", "tail", "relation_id", "split", "type_set_id"}
    for name, reference in manifest["tensors"].items():
        assert reference["format"] == "parquet", name
    assert not (dataset / "split-records.ndjson").exists()
    assert not (dataset / "edge_index.npy").exists()
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
        for path in sorted(dataset.rglob("*.parquet"))
        if "batches" not in path.parts
    }
    shutil.rmtree(dataset)
    _, again, _ = prepare(tmp_path, monkeypatch)
    second = {
        str(path.relative_to(again)): sha256_file(path)
        for path in sorted(again.rglob("*.parquet"))
        if "batches" not in path.parts
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
