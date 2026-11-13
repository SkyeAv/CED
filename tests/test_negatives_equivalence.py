"""Contract tests for uniform filtered negatives without replacement.

Why these exist: duplicate corruptions and wraparound first-valid selection bias
training toward a few candidates. Small-pool equivalence uses an independent
eligible-list reference; large-pool tests pin filtering, diversity, and termination.
"""

from __future__ import annotations

from pathlib import Path

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
    # Exhausted sides must terminate and retain explicitly invalid cells.
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


def test_negatives_are_unique_and_exhaustion_is_explicit() -> None:
    """One eligible candidate must not be repeated to fill a nominal negative count."""
    pools = TypedPools(subjects={0: np.arange(4)}, objects={0: np.arange(10, 14)})
    context = negative_context(np.array([0, 1, 2]), np.zeros(3, dtype=np.int64), np.array([10, 10, 10]))
    heads, tails, valid = sample_negatives([(0, 0, 10)], pools, context, count=12, seed=7)
    assert valid[0, 0::2].sum() == 1
    assert valid[0, 1::2].sum() == 3
    assert heads[0, 0::2][valid[0, 0::2]].tolist() == [3]
    assert len(np.unique(tails[0, 1::2][valid[0, 1::2]])) == 3
    assert np.all(heads[~valid] == 0)
    assert np.all(tails[~valid] == 10)


@pytest.mark.parametrize("count", [0, 1, 7])
def test_empty_and_missing_pools_preserve_endpoints(count: int) -> None:
    """Missing pools are explicit absence, not fabricated zero-ID negatives."""
    pools = TypedPools(subjects={}, objects={0: np.empty(0, dtype=np.int64)})
    context = negative_context(np.array([3]), np.array([0]), np.array([4]))
    heads, tails, valid = sample_negatives([(3, 0, 4)], pools, context, count=count, seed=9)
    assert heads.shape == tails.shape == valid.shape == (1, count)
    assert np.all(heads == 3)
    assert np.all(tails == 4)
    assert not valid.any()


def test_global_ids_outside_positive_index_do_not_alias_known_triples() -> None:
    """Sparse global IDs must not be rejected through compact-key arithmetic aliasing."""
    pools = TypedPools(subjects={0: np.array([100, 101])}, objects={})
    context = negative_context(np.array([1]), np.array([0]), np.array([1]))
    heads, _, valid = sample_negatives([(0, 0, 0)], pools, context, count=4, seed=4)
    assert set(heads[0, 0::2][valid[0, 0::2]].tolist()) == {100, 101}
    assert valid[0, 0::2].sum() == 2


def test_negative_count_is_rejected() -> None:
    """Negative output dimensions fail with a clear sampler-specific error."""
    context = negative_context(np.array([0]), np.array([0]), np.array([1]))
    with pytest.raises(ValueError, match="negative count must be nonnegative"):
        sample_negatives([], TypedPools({}, {}), context, count=-1, seed=0)


def test_uniform_sampling_has_no_first_valid_bias() -> None:
    """Long positive runs must not inflate the next eligible candidate's probability."""
    pools = TypedPools(subjects={0: np.arange(10)}, objects={})
    context = negative_context(np.arange(8), np.zeros(8, dtype=np.int64), np.zeros(8, dtype=np.int64))
    frequencies = {8: 0, 9: 0}
    for seed in range(400):
        heads, _, valid = sample_negatives([(0, 0, 0)], pools, context, count=1, seed=seed)
        assert valid[0, 0]
        frequencies[int(heads[0, 0])] += 1
    assert 140 < frequencies[8] < 260
    assert 140 < frequencies[9] < 260


def test_large_dense_pool_terminates_with_all_eligible_candidates() -> None:
    """Bounded rejection must fall back instead of losing a rare eligible candidate."""
    size = 10000
    heads = np.arange(size - 1, dtype=np.int64)
    context = negative_context(heads, np.zeros(size - 1, dtype=np.int64), np.zeros(size - 1, dtype=np.int64))
    pools = TypedPools(subjects={0: np.arange(size)}, objects={})
    first = sample_negatives([(0, 0, 0)], pools, context, count=8, seed=11)
    second = sample_negatives([(0, 0, 0)], pools, context, count=8, seed=11)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)
    assert first[2].sum() == 1
    assert first[0][first[2]].tolist() == [size - 1]


def test_large_pool_disk_and_memory_sampling_agree(tmp_path: Path) -> None:
    """SQLite candidate lookup must preserve the same bounded draw order as NumPy."""
    from ced.model.disk_graph import DiskGraph

    size = 5000
    arrays = {
        "head": np.array([0, 1], dtype=np.int64),
        "tail": np.array([1, 2], dtype=np.int64),
        "relation_id": np.zeros(2, dtype=np.int64),
        "split": np.zeros(2, dtype=np.int8),
        "type_set_id": np.zeros(size, dtype=np.int64),
    }
    edges = np.stack((arrays["head"], arrays["tail"]))
    pools = typed_pools(edges, arrays["relation_id"], arrays["split"], arrays["type_set_id"])
    context = negative_context(arrays["head"], arrays["relation_id"], arrays["tail"])
    triples = [(0, 0, 1), (1, 0, 2)]
    expected = sample_negatives(triples, pools, context, count=16, seed=17)
    disk = DiskGraph(tmp_path / "graph.db", arrays)
    try:
        actual = disk.negatives(triples, count=16, seed=17)
        for left, right in zip(expected, actual, strict=True):
            assert np.array_equal(left, right)
        for row in range(len(triples)):
            assert len(np.unique(actual[0][row, 0::2])) == 8
            assert len(np.unique(actual[1][row, 1::2])) == 8
        with pytest.raises(ValueError, match="negative count must be nonnegative"):
            disk.negatives(triples, count=-1, seed=17)
    finally:
        disk.close()


def test_empty_inputs_keep_reference_shapes() -> None:
    pools = TypedPools(subjects={}, objects={})
    context = negative_context(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    produced = sample_negatives([], pools, context, count=4, seed=1)
    for array in produced:
        assert array.shape == (0, 4)
