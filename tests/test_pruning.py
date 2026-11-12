"""Isolated-tree pruning contracts for dataset preparation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest
import test_prepare_topology as topology

from ced.config.models import RuntimeConfig
from ced.model.prepare import prepare_dataset
from ced.model.schemas import TensorRef


def _component_mask(
    node_count: int, edges: list[tuple[int, int]], max_nodes: int = 2, batch_size: int | None = None
) -> tuple[np.ndarray, int, int]:
    from ced.model.pruning import _isolated_tree_nodes

    def batches():
        if edges:
            array = np.asarray(edges, dtype=np.int64)
            for start in range(0, len(edges), batch_size or len(edges)):
                chunk = array[start : start + (batch_size or len(edges))]
                yield chunk[:, 0].copy(), chunk[:, 1].copy()

    return _isolated_tree_nodes(node_count, max_nodes, batches)


def test_isolated_tree_detection_is_component_aware() -> None:
    """Only small disconnected trees are removed, not leaves attached to a cyclic core."""
    edges = [
        (0, 1),
        (1, 2),
        (2, 0),
        (2, 3),
        (3, 4),  # Cyclic component with two attached leaves: retain every node.
        (6, 7),  # Isolated pair: prune.
        (8, 9),
        (9, 10),  # Three-node tree: above the default threshold.
        (11, 12),
        (12, 11),  # Parallel multigraph edges form a cycle: retain.
        (13, 13),  # A self-loop is not a tree: retain.
        (14, 15),
        (15, 16),
        (16, 14),  # Disconnected cyclic component: retain.
    ]
    mask, components, statements = _component_mask(17, edges, batch_size=2)

    assert np.flatnonzero(mask).tolist() == [5, 6, 7]
    assert components == 2
    assert statements == 1

    larger, larger_components, larger_statements = _component_mask(17, edges, max_nodes=3)
    assert np.flatnonzero(larger).tolist() == [5, 6, 7, 8, 9, 10]
    assert larger_components == 3
    assert larger_statements == 3


def test_isolated_tree_detection_handles_empty_and_disconnected_inputs() -> None:
    mask, components, statements = _component_mask(3, [])
    assert mask.tolist() == [True, True, True]
    assert (components, statements) == (3, 0)

    empty = _component_mask(0, [])
    assert empty[0].size == 0
    assert empty[1:] == (0, 0)


def test_isolated_tree_detection_rejects_invalid_endpoint_batches() -> None:
    from ced.model.pruning import _isolated_tree_nodes

    with pytest.raises(ValueError, match="matching shapes"):
        _isolated_tree_nodes(2, 2, lambda: [(np.array([0]), np.array([0, 1]))])
    with pytest.raises(ValueError, match="outside the node vocabulary"):
        _isolated_tree_nodes(2, 2, lambda: [(np.array([0]), np.array([2]))])


def _pruning_fixture() -> tuple[dict, dict, dict]:
    nodes = copy.deepcopy(topology.NODES)
    nodes["id"].extend(["N4", "N5", "N6", "N7", "N8", "N9"])
    for key in ("names", "in_taxon", "provided_by", "source_ids"):
        nodes[key].extend([[] for _ in range(6)])
    nodes["categories"].extend([["biolink:Gene"] for _ in range(6)])
    for key in ("mapping_status", "original_label", "native_ids_json", "attributes_json"):
        template = topology.NODES[key][0]
        nodes[key].extend([template for _ in range(6)])

    statements = copy.deepcopy(topology.STATEMENTS)
    additions = [
        ("s6", "N5", "N6"),  # Two-node tree: prune.
        ("s7", "N7", "N8"),
        ("s8", "N8", "N9"),  # Three-node tree: retain at the conservative default.
    ]
    for statement_id, subject, object_id in additions:
        for key, values in statements.items():
            values.append(copy.deepcopy(topology.STATEMENTS[key][0]))
        statements["id"][-1] = statement_id
        statements["native_id"][-1] = statement_id
        statements["subject"][-1] = subject
        statements["predicate"][-1] = "p0"
        statements["object"][-1] = object_id

    attributes = copy.deepcopy(topology.ATTRIBUTES)
    for template, owner_kind, owner_id in ((0, "node", "N4"), (2, "statement", "s6"), (0, "node", "N7")):
        for key, values in attributes.items():
            values.append(copy.deepcopy(topology.ATTRIBUTES[key][template]))
        attributes["owner_kind"][-1] = owner_kind
        attributes["owner_id"][-1] = owner_id
    return nodes, statements, attributes


def _prepare_pruning_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_nodes: int,
    *,
    drop_node: str | None = None,
    tamper_counts: bool = False,
    tree_only: bool = False,
) -> tuple[Path, dict]:
    monkeypatch.setattr("ced.model.prepare.require_cpu_worker", lambda runtime: None)
    runtime = RuntimeConfig.model_validate(
        {
            "remote_project": str(tmp_path),
            "artifact_root": str(tmp_path / "artifacts"),
            "cpu_workers": 4,
            "memory_limit": "4GB",
            "disk_reserve_gb": 0.5,
            "preprocessing": {"isolated_tree_max_nodes": max_nodes},
        }
    )
    snapshot = runtime.artifact_path("snapshots", "snapshot")
    nodes, statements, attributes = _pruning_fixture()
    if drop_node is not None:
        position = nodes["id"].index(drop_node)
        for values in nodes.values():
            values.pop(position)
    if tree_only:
        node_ids = {"N5", "N6"}
        node_positions = [index for index, node_id in enumerate(nodes["id"]) if node_id in node_ids]
        nodes = {key: [values[index] for index in node_positions] for key, values in nodes.items()}
        statement_positions = [index for index, value in enumerate(statements["id"]) if value == "s6"]
        statements = {key: [values[index] for index in statement_positions] for key, values in statements.items()}
        attribute_positions = [
            index for index, owner_id in enumerate(attributes["owner_id"]) if owner_id in node_ids | {"s6"}
        ]
        attributes = {key: [values[index] for index in attribute_positions] for key, values in attributes.items()}
    with (
        patch.object(topology, "NODES", nodes),
        patch.object(topology, "STATEMENTS", statements),
        patch.object(topology, "ATTRIBUTES", attributes),
    ):
        topology.write_snapshot(snapshot)
    if tamper_counts:
        path = snapshot / "manifest.json"
        snapshot_manifest = json.loads(path.read_text())
        snapshot_manifest["counts"]["nodes"] += 1
        path.write_text(json.dumps(snapshot_manifest))
    dataset = prepare_dataset(runtime, "snapshot")
    return dataset, json.loads((dataset / "manifest.json").read_text())


def test_prepare_prunes_isolated_trees_before_vocabularies_features_and_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, manifest = _prepare_pruning_fixture(tmp_path, monkeypatch, max_nodes=2)

    assert manifest["entities"] == 7
    assert manifest["statements"] == 8
    assert manifest["pruning"] == {
        "max_component_nodes": 2,
        "input_nodes": 10,
        "input_statements": 9,
        "pruned_components": 2,
        "pruned_nodes": 3,
        "pruned_statements": 1,
    }
    assert manifest["vocabularies"]["entities"]["rows"] == 7
    assert manifest["tensors"]["type_set_id"]["shape"] == [7]
    assert manifest["tensors"]["head"]["shape"] == [8]

    entities = pq.read_table(dataset / "entities.parquet", columns=["id", "idx"])
    assert entities.column("id").to_pylist() == ["N0", "N1", "N2", "N3", "N7", "N8", "N9"]
    assert entities.column("idx").to_pylist() == list(range(7))
    topology_table = pq.read_table(dataset / "topology.parquet")
    statement_ids = topology_table.column("statement_id").to_pylist()
    assert "s6" not in statement_ids
    assert {"s7", "s8"} <= set(statement_ids)
    assert set(topology_table.column("head").to_pylist()) <= set(range(7))
    assert set(topology_table.column("tail").to_pylist()) <= set(range(7))

    node_owners = pq.read_table(dataset / "node-features.parquet", columns=["owner_id"]).column(0).to_pylist()
    statement_owners = pq.read_table(dataset / "statement-features.parquet", columns=["owner_id"]).column(0).to_pylist()
    assert "N4" not in node_owners
    assert "N7" in node_owners
    assert "s6" not in statement_owners
    retained_index = entities.column("idx").to_pylist()[entities.column("id").to_pylist().index("N7")]
    numeric_owners = TensorRef.model_validate(manifest["features"]["numeric"]["owner_id"]).load(dataset).tolist()
    assert retained_index in numeric_owners


def test_prepare_can_disable_isolated_tree_pruning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, manifest = _prepare_pruning_fixture(tmp_path, monkeypatch, max_nodes=0)

    assert manifest["entities"] == 10
    assert manifest["statements"] == 9
    assert manifest["pruning"]["pruned_components"] == 0
    assert manifest["pruning"]["pruned_nodes"] == 0
    assert manifest["pruning"]["pruned_statements"] == 0
    entities = pq.read_table(dataset / "entities.parquet", columns=["id"])
    assert "N4" in entities.column("id").to_pylist()


def test_prepare_rejects_dangling_pruning_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="statement endpoint count mismatch"):
        _prepare_pruning_fixture(tmp_path, monkeypatch, max_nodes=2, drop_node="N9")


def test_prepare_rejects_tampered_snapshot_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="manifest counts do not match"):
        _prepare_pruning_fixture(tmp_path, monkeypatch, max_nodes=2, tamper_counts=True)


def test_prepare_rejects_pruning_every_statement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="must retain at least one node and one statement"):
        _prepare_pruning_fixture(tmp_path, monkeypatch, max_nodes=2, tree_only=True)
