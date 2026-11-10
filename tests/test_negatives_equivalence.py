"""Equivalence tests: vectorized negatives/pools vs the frozen reference.

Why this exists: the vectorized implementation replaced a per-cell Python loop
that negative sampling determinism depends on (RNG draw order, wraparound scan
order, first-valid-candidate selection). Every optimization pass must keep
producing byte-identical arrays, or downstream batch tensors silently change.
"""

from __future__ import annotations

import numpy as np
import pytest
from perf_support import synthetic_graph
from reference_negatives import sample_negatives_reference, typed_pools_reference

from ced.model.negatives import TypedPools, negative_context, sample_negatives, typed_pools


def _check_case(
    edges: np.ndarray,
    relations: np.ndarray,
    splits: np.ndarray,
    types: np.ndarray,
    triples: list[tuple[int, int, int]],
    *,
    count: int,
    seed: int,
) -> None:
    reference_subjects, reference_objects = typed_pools_reference(edges, relations, splits, types)
    pools = typed_pools(edges, relations, splits, types)
    for relation in reference_subjects:
        assert np.array_equal(pools.subjects[relation], reference_subjects[relation])
        assert np.array_equal(pools.objects[relation], reference_objects[relation])
    positives = {
        (int(head), int(relation), int(tail))
        for head, relation, tail in zip(edges[0], relations, edges[1], strict=True)
    }
    context = negative_context(edges[0], relations, edges[1])
    expected = sample_negatives_reference(
        triples, reference_subjects, reference_objects, positives, count=count, seed=seed
    )
    produced = sample_negatives(triples, pools, context, count=count, seed=seed)
    assert np.array_equal(produced[0], expected[0])
    assert np.array_equal(produced[1], expected[1])
    assert np.array_equal(produced[2], expected[2])


def test_vectorized_negatives_match_reference_on_contract_fixture() -> None:
    edges = np.array([[10, 10, 11, 12], [12, 13, 13, 13]], dtype=np.int64)
    relations = np.array([0, 1, 0, 1], dtype=np.int64)
    splits = np.array([0, 0, 0, 2], dtype=np.int8)
    types = np.array([*([9] * 10), 1, 1, 2, 2], dtype=np.int64)
    _check_case(edges, relations, splits, types, [(10, 0, 12), (10, 1, 13)], count=16, seed=7)


def test_vectorized_negatives_match_reference_when_pool_exhausts() -> None:
    # Every candidate is a positive, so the wraparound scan must exhaust and
    # leave the cells invalid instead of looping or fabricating a candidate.
    edges = np.array([[0, 1], [1, 0]], dtype=np.int64)
    relations = np.array([0, 0], dtype=np.int64)
    splits = np.array([0, 0], dtype=np.int8)
    types = np.array([0, 0], dtype=np.int64)
    _check_case(edges, relations, splits, types, [(0, 0, 1), (1, 0, 0)], count=6, seed=3)


def test_vectorized_negatives_match_reference_on_random_graphs() -> None:
    for case in range(4):
        graph = synthetic_graph(
            nodes=60 + 20 * case, edges=400 + 200 * case, relations=1 + case, type_sets=2 + case, seed=100 + case
        )
        edges, relations, splits = graph["edge_index"], graph["relation_id"], graph["split"]
        train = np.flatnonzero(splits == 0)
        selected = train[np.linspace(0, len(train) - 1, 20).astype(int)]
        triples = [(int(edges[0, i]), int(relations[i]), int(edges[1, i])) for i in selected]
        _check_case(edges, relations, splits, graph["type_set_id"], triples, count=12, seed=500 + case)


def test_negative_context_rejects_oversized_encoding() -> None:
    # The int64 key packing must fail loudly instead of silently aliasing keys.
    heads = np.array([2**31], dtype=np.int64)
    relations = np.array([2**31], dtype=np.int64)
    tails = np.array([2**31], dtype=np.int64)
    with pytest.raises(ValueError, match="int64"):
        negative_context(heads, relations, tails)


def test_empty_inputs_keep_reference_shapes() -> None:
    pools = TypedPools(subjects={}, objects={})
    context = negative_context(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    produced = sample_negatives([], pools, context, count=4, seed=1)
    for array in produced:
        assert array.shape == (0, 4)
