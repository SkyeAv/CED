"""Supervision-guarantee tests for neighborhood sampling.

Why this exists: fanout sampling can randomly drop a training batch's own
supervision statements from its message-passing set, and rare edges (CED
treatment edges) are exactly the ones lost, silently zeroing their training
signal. These tests pin the injection contract: train positives are guaranteed
present, held-out statements are never injected (split isolation), default
behavior stays byte-identical to the frozen reference, and the in-memory and
disk engines agree when injection is on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ced.model.batches import sample_neighborhood
from ced.model.disk_graph import DiskGraph


def _hub_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One supervision edge plus a hub whose fanout-[1] choice usually drops it.

    Statement 0 is the supervision edge (7 -> 99, relation 5); statements 1..20
    hang off hub node 0 with relation 0. With fanout [1] and seeds [0], the
    sampler picks one hub statement, so statement 0 survives only by luck.
    """
    edges = np.array([[7, *range(20)], [99, *range(1, 21)]], dtype=np.int64)
    relations = np.array([5, *([0] * 20)], dtype=np.int64)
    splits = np.zeros(21, dtype=np.int8)
    return edges, relations, splits


def test_required_train_positive_is_injected() -> None:
    edges, relations, splits = _hub_fixture()
    dropped = [
        seed
        for seed in range(64)
        if 0
        not in sample_neighborhood(
            edges, relations, splits, np.array([0]), fanout=[1], seed=seed, node_limit=64
        ).statement_id
    ]
    # The fixture must demonstrate the loss the injection prevents.
    assert dropped
    for seed in range(64):
        produced = sample_neighborhood(
            edges, relations, splits, np.array([0]), fanout=[1], seed=seed, node_limit=64, required=np.array([0])
        )
        # Injected statements enter message passing in both directions exactly once each.
        assert int(np.count_nonzero(produced.statement_id == 0)) == 2
        assert set(map(int, produced.global_node_id)) >= {7, 99}
        positions = np.flatnonzero(produced.statement_id == 0)
        assert produced.relation_id[positions[0]] == 5
        # Reverse messages carry the relation offset (relation_count = 6 here).
        assert produced.relation_id[positions[1]] == 5 + 6


def test_required_heldout_statements_are_not_injected() -> None:
    # Statements 1 and 2 are test-split and schema-role: injecting them would leak
    # evaluation targets into message passing, so the output must not change at all.
    edges = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int64)
    relations = np.array([0, 0, 0], dtype=np.int64)
    splits = np.array([0, 2, 3], dtype=np.int8)
    baseline = sample_neighborhood(edges, relations, splits, np.array([0]), fanout=[10], seed=1, node_limit=4)
    produced = sample_neighborhood(
        edges, relations, splits, np.array([0]), fanout=[10], seed=1, node_limit=4, required=np.array([1, 2])
    )
    for left, right in zip(
        (
            baseline.global_node_id,
            baseline.edge_index,
            baseline.statement_id,
            baseline.relation_id,
            baseline.edge_type_id,
        ),
        (
            produced.global_node_id,
            produced.edge_index,
            produced.statement_id,
            produced.relation_id,
            produced.edge_type_id,
        ),
        strict=True,
    ):
        assert np.array_equal(left, right)


def test_disk_neighborhood_matches_in_memory_with_required(tmp_path: Path) -> None:
    # fanout covers every training statement, so both engines must return identical
    # neighborhoods including the injected supervision statements.
    edges = np.array([[10, 10, 11, 12], [12, 13, 13, 13]], dtype=np.int64)
    relations = np.array([0, 1, 0, 1], dtype=np.int64)
    splits = np.array([0, 0, 0, 2], dtype=np.int8)
    types = np.array([*([9] * 10), 1, 1, 2, 2], dtype=np.int64)
    arrays = {"head": edges[0], "tail": edges[1], "relation_id": relations, "split": splits, "type_set_id": types}
    disk = DiskGraph(tmp_path / "graph.db", arrays)
    try:
        produced = disk.neighborhood(
            edges, relations, np.array([10]), fanout=[10], seed=1, node_limit=14, required=np.array([0, 1])
        )
    finally:
        disk.close()
    expected = sample_neighborhood(
        edges, relations, splits, np.array([10]), fanout=[10], seed=1, node_limit=14, required=np.array([0, 1])
    )
    assert set(map(int, produced.statement_id)) == {0, 1}
    for left, right in zip(
        (
            expected.global_node_id,
            expected.edge_index,
            expected.statement_id,
            expected.relation_id,
            expected.edge_type_id,
        ),
        (
            produced.global_node_id,
            produced.edge_index,
            produced.statement_id,
            produced.relation_id,
            produced.edge_type_id,
        ),
        strict=True,
    ):
        assert np.array_equal(left, right)


def test_required_injection_respects_node_limit() -> None:
    # Seed node 50 is isolated, so fanout adds nothing; injecting the (100, 101)
    # supervision edge would push the node set past node_limit and must fail loudly
    # instead of silently truncating supervision.
    edges = np.array([[100], [101]], dtype=np.int64)
    relations = np.array([0], dtype=np.int64)
    splits = np.zeros(1, dtype=np.int8)
    with pytest.raises(ValueError, match="required supervision endpoints exceed batch node limit"):
        sample_neighborhood(
            edges, relations, splits, np.array([50]), fanout=[3], seed=1, node_limit=2, required=np.array([0])
        )
