"""Equivalence tests: CSR sample_neighborhood vs the frozen mask-scan reference.

Why this exists: neighbor sampling determinism (RNG draw order, candidate
order, node_limit skip order) decides which statements enter every training
batch. The CSR rewrite must be byte-identical per seed or batches silently
change. Each case compares all five output arrays.
"""

from __future__ import annotations

import numpy as np
from perf_support import synthetic_graph
from reference_batches import sample_neighborhood_reference

from ced.model.batches import TrainAdjacency, sample_neighborhood


def _check_case(
    edges: np.ndarray,
    relations: np.ndarray,
    splits: np.ndarray,
    seeds: np.ndarray,
    *,
    fanout: list[int],
    seed: int,
    node_limit: int,
    edge_types: np.ndarray | None = None,
    adjacency: TrainAdjacency | None = None,
) -> None:
    expected = sample_neighborhood_reference(
        edges, relations, splits, seeds, fanout=fanout, seed=seed, node_limit=node_limit, edge_types=edge_types
    )
    produced = sample_neighborhood(
        edges,
        relations,
        splits,
        seeds,
        fanout=fanout,
        seed=seed,
        node_limit=node_limit,
        edge_types=edge_types,
        adjacency=adjacency,
    )
    actual = (
        produced.global_node_id,
        produced.edge_index,
        produced.statement_id,
        produced.relation_id,
        produced.edge_type_id,
    )
    for left, right in zip(actual, expected, strict=True):
        assert np.array_equal(left, right)


def test_csr_matches_reference_on_contract_fixture() -> None:
    edges = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int64)
    relations = np.array([0, 0, 0], dtype=np.int64)
    splits = np.array([0, 2, 3], dtype=np.int8)
    _check_case(edges, relations, splits, np.array([0]), fanout=[10], seed=1, node_limit=4)


def test_csr_matches_reference_with_self_loops() -> None:
    # A self-loop must be listed once per node, exactly like the boolean mask.
    edges = np.array([[0, 1, 0, 2, 1], [0, 1, 2, 1, 0]], dtype=np.int64)
    relations = np.array([0, 0, 1, 1, 0], dtype=np.int64)
    splits = np.array([0, 0, 0, 0, 2], dtype=np.int8)
    _check_case(edges, relations, splits, np.array([0]), fanout=[5, 5], seed=2, node_limit=8)


def test_csr_matches_reference_under_node_limit_pressure() -> None:
    graph = synthetic_graph(nodes=40, edges=300, relations=2, type_sets=2, seed=11)
    seeds = np.array([0, 1, 2, 3], dtype=np.int64)
    for limit in (5, 12, 40):
        _check_case(
            graph["edge_index"],
            graph["relation_id"],
            graph["split"],
            seeds,
            fanout=[8, 6],
            seed=3,
            node_limit=limit,
            edge_types=graph["edge_type_id"],
        )


def test_csr_matches_reference_on_random_graphs() -> None:
    for case in range(3):
        graph = synthetic_graph(
            nodes=80 + 30 * case, edges=600 + 300 * case, relations=1 + case, type_sets=2 + case, seed=200 + case
        )
        train = np.flatnonzero(graph["split"] == 0)
        seeds = np.unique(graph["edge_index"][0, train[:25]])
        for fanout, seed in (([10], 5), ([6, 4], 6), ([3, 3, 3], 7)):
            _check_case(
                graph["edge_index"],
                graph["relation_id"],
                graph["split"],
                seeds,
                fanout=fanout,
                seed=seed,
                node_limit=512,
                edge_types=graph["edge_type_id"],
            )


def test_shared_adjacency_matches_per_call_build() -> None:
    graph = synthetic_graph(nodes=60, edges=400, relations=2, type_sets=2, seed=12)
    seeds = np.array([1, 4, 9], dtype=np.int64)
    shared = TrainAdjacency(graph["edge_index"], graph["split"])
    _check_case(
        graph["edge_index"],
        graph["relation_id"],
        graph["split"],
        seeds,
        fanout=[5, 5],
        seed=9,
        node_limit=256,
        adjacency=shared,
    )
