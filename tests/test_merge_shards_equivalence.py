"""Byte-equivalence tests for US-008 merge and shard optimizations.

Why this exists: merged canonical JSON determines downstream scientific content,
and batch tensors determine seeded training trajectories. Reducing Python/JSON
overhead is valid only if rows, references, and every saved tensor stay exactly
identical to the frozen implementations, including multi-group shard boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from perf_support import synthetic_graph
from reference_merge import merge_graph_reference
from reference_shards import write_batch_shards_reference

from ced.config.models import BatchSettings, RuntimeConfig, SplitSettings
from ced.kg.merge import merge_graph
from ced.kg.records import CanonicalBatch, NodeRecord
from ced.kg.writer import CanonicalBatchWriter
from ced.model.shards import _edge_type_ids, write_batch_shards


def test_edge_type_ids_preserve_first_occurrence_and_empty_train() -> None:
    edges = np.array([[2, 1, 2, 0], [3, 2, 3, 1]], dtype=np.int64)
    relations = np.array([1, 0, 1, 0], dtype=np.int64)
    types = np.array([5, 4, 3, 2], dtype=np.int64)
    splits = np.array([0, 0, 0, 2], dtype=np.int8)
    assert np.array_equal(_edge_type_ids(edges, relations, splits, types), np.array([0, 1, 0, 0]))
    assert np.array_equal(
        _edge_type_ids(edges, relations, np.array([1, 2, 2, 2], dtype=np.int8), types), np.zeros(4, dtype=np.int64)
    )
    negative_relations = np.array([-1, 0, -1, 0], dtype=np.int64)
    assert np.array_equal(_edge_type_ids(edges, negative_relations, splits, types), np.array([0, 1, 0, 0]))
    huge_types = types.copy()
    huge_types[2] = np.iinfo(np.int64).max
    assert np.array_equal(_edge_type_ids(edges, relations, splits, huge_types), np.array([0, 1, 0, 0]))


def _merge_fixture(root: Path) -> None:
    writer = CanonicalBatchWriter(root / "raw", shard_rows=2)
    writer.write(
        CanonicalBatch(
            nodes=[
                NodeRecord(
                    id="X:1",
                    names=["One"],
                    categories=["biolink:Gene"],
                    in_taxon=["NCBITaxon:9606"],
                    provided_by=["source-b"],
                    original_label=None,
                    native_ids={"source-b": "B:1"},
                    attributes={"retained": "until-long-events"},
                ),
                NodeRecord(
                    id="X:2",
                    names=["Two"],
                    categories=["biolink:Disease"],
                    native_ids={"source-a": "A:2"},
                    attributes={"single-group": "must-survive"},
                ),
            ]
        )
    )
    writer.write(
        CanonicalBatch(
            nodes=[
                NodeRecord(
                    id="X:1",
                    names=["Uno"],
                    categories=["AMALGAM:Gene"],
                    in_taxon=["NCBITaxon:9606", "NCBITaxon:10090"],
                    provided_by=["source-a"],
                    mapping_status="local",
                    original_label="Primary label",
                    native_ids={"source-a": "A:1"},
                    attributes={"discarded-from-denormalized-node": True},
                ),
                NodeRecord(
                    id="X:1",
                    names=["One", "First"],
                    categories=["biolink:Gene", "biolink:NamedThing"],
                    mapping_status="unresolved_local",
                    original_label="Primary label",
                    native_ids={"source-a": "A:override", "source-c": "C:1"},
                ),
            ]
        )
    )
    writer.close()


def test_merge_rows_and_kgx_are_identical_to_reference(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    produced_root = tmp_path / "produced"
    _merge_fixture(reference_root)
    _merge_fixture(produced_root)
    reference_counts = merge_graph_reference(reference_root, memory="128MB")
    produced_counts = merge_graph(produced_root, memory="128MB")
    assert produced_counts == reference_counts
    expected = pq.read_table(reference_root / "nodes").to_pylist()
    produced = pq.read_table(produced_root / "nodes").to_pylist()
    assert produced == expected
    one = next(row for row in produced if row["id"] == "X:1")
    two = next(row for row in produced if row["id"] == "X:2")
    assert set(json.loads(one["native_ids_json"])) == {"source-a", "source-b", "source-c"}
    assert one["attributes_json"] == "{}"
    assert json.loads(two["attributes_json"]) == {"single-group": "must-survive"}
    reference_kgx = (reference_root / "kgx/nodes/part-00000.ndjson").read_bytes()
    produced_kgx = (produced_root / "kgx/nodes/part-00000.ndjson").read_bytes()
    assert produced_kgx == reference_kgx


def _runtime(root: Path) -> RuntimeConfig:
    return RuntimeConfig(
        remote_project=root,
        artifact_root=root / ".ced",
        split=SplitSettings(seed=19, negatives=2),
        batch=BatchSettings(node_limit=32, neighbor_fanout=[3, 2]),
    )


def test_batch_shards_and_tensor_bytes_are_identical_to_reference(tmp_path: Path) -> None:
    arrays = synthetic_graph(nodes=30, edges=90, relations=3, type_sets=3, seed=23)
    reference_root = tmp_path / "reference"
    produced_root = tmp_path / "produced"
    expected = write_batch_shards_reference(reference_root, arrays, _runtime(tmp_path))
    produced = write_batch_shards(produced_root, arrays, _runtime(tmp_path))
    assert [example.model_dump(mode="json") for example in produced] == [
        example.model_dump(mode="json") for example in expected
    ]
    reference_files = sorted(path.relative_to(reference_root) for path in reference_root.rglob("*.npy"))
    produced_files = sorted(path.relative_to(produced_root) for path in produced_root.rglob("*.npy"))
    assert produced_files == reference_files
    for relative in reference_files:
        assert (produced_root / relative).read_bytes() == (reference_root / relative).read_bytes(), relative


def test_batch_shards_disk_path_is_identical_to_reference(tmp_path: Path) -> None:
    # The same bulk triples feed DiskGraph, so its seeded SQL path must also
    # receive exactly the historical row order and Python integer values.
    arrays = synthetic_graph(nodes=20, edges=40, relations=2, type_sets=2, seed=29)
    reference_root = tmp_path / "reference-disk"
    produced_root = tmp_path / "produced-disk"
    reference_root.mkdir()
    produced_root.mkdir()
    expected = write_batch_shards_reference(reference_root, arrays, _runtime(tmp_path), in_memory=False)
    produced = write_batch_shards(produced_root, arrays, _runtime(tmp_path), in_memory=False)
    assert [example.model_dump(mode="json") for example in produced] == [
        example.model_dump(mode="json") for example in expected
    ]
    for relative in sorted(path.relative_to(reference_root) for path in reference_root.rglob("*.npy")):
        assert (produced_root / relative).read_bytes() == (reference_root / relative).read_bytes(), relative
