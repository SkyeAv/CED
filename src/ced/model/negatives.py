from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TypedPools:
    subjects: dict[int, np.ndarray]
    objects: dict[int, np.ndarray]


@dataclass(frozen=True)
class NegativeContext:
    """Sorted int64 keys for every known positive triple plus the encoding bounds.

    Encoding a (head, relation, tail) triple as one int64 lets candidate
    filtering run in vectorized numpy instead of per-cell Python set lookups.
    The context is built once per graph (all statements, not only train) so
    repeated per-chunk negative sampling never re-encodes the positives.
    """

    keys: np.ndarray
    node_count: int
    relation_count: int


def negative_context(heads: np.ndarray, relations: np.ndarray, tails: np.ndarray) -> NegativeContext:
    """Encode all known positive triples once for membership tests."""
    heads = np.asarray(heads, dtype=np.int64)
    relations = np.asarray(relations, dtype=np.int64)
    tails = np.asarray(tails, dtype=np.int64)
    if heads.size == 0:
        return NegativeContext(np.empty(0, dtype=np.int64), 0, 0)
    node_count = int(max(heads.max(), tails.max())) + 1
    relation_count = int(relations.max()) + 1
    if node_count * node_count * relation_count > np.iinfo(np.int64).max:
        raise ValueError("graph exceeds int64 triple-key encoding capacity")
    keys = np.unique((heads * relation_count + relations) * node_count + tails)
    return NegativeContext(keys, node_count, relation_count)


def _contains(keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    if keys.size == 0:
        return np.zeros(query.shape, dtype=np.bool_)
    position = np.searchsorted(keys, query)
    np.minimum(position, keys.size - 1, out=position)
    return keys[position] == query


def typed_pools(edge_index: np.ndarray, relations: np.ndarray, splits: np.ndarray, types: np.ndarray) -> TypedPools:
    """Typed candidate pools per relation side, built from one node partition.

    Nodes are partitioned by type once (stable sort keeps ids ascending inside
    each block); a pool is the sorted union of the blocks whose type appears on
    that side of the relation's training edges. This avoids scanning every node
    per relation and deduplicates pools shared between relations.
    """
    order = np.argsort(types, kind="stable")
    unique_types, starts = np.unique(types[order], return_index=True)
    boundaries = [*starts.tolist(), order.size]
    blocks = {
        int(kind): order[boundaries[index] : boundaries[index + 1]] for index, kind in enumerate(unique_types.tolist())
    }
    train = splits == 0
    cache: dict[frozenset[int], np.ndarray] = {}
    subjects: dict[int, np.ndarray] = {}
    objects: dict[int, np.ndarray] = {}
    for relation in np.unique(relations[train]).tolist():
        selected = train & (relations == relation)
        for side, destination in ((0, subjects), (1, objects)):
            key = frozenset(types[edge_index[side, selected]].tolist())
            pool = cache.get(key)
            if pool is None:
                pool = np.sort(np.concatenate([blocks[kind] for kind in key])) if key else np.empty(0, dtype=np.int64)
                cache[key] = pool
            destination[relation] = pool
    return TypedPools(subjects=subjects, objects=objects)


def _pool(pools: TypedPools, relation: int, side: int) -> np.ndarray | None:
    dictionary = pools.subjects if side == 0 else pools.objects
    return dictionary.get(relation)


def sample_negatives(
    triples: Iterable[tuple[int, int, int]],
    pools: TypedPools,
    context: NegativeContext,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Typed corruption negatives identical to the reference row/column loop.

    Preserves the exact RNG contract of the original implementation: one
    integers() draw per drawable cell in row-major order, then a wraparound
    scan from that offset taking the first candidate that is not a known
    positive. Cells whose pool is missing or empty stay invalid.
    """
    rows = list(triples)
    subjects = np.zeros((len(rows), count), dtype=np.int64)
    objects = np.zeros_like(subjects)
    valid = np.zeros((len(rows), count), dtype=np.bool_)
    if not rows or count <= 0:
        return subjects, objects, valid
    row_heads = np.fromiter((row[0] for row in rows), dtype=np.int64, count=len(rows))
    row_relations = np.fromiter((row[1] for row in rows), dtype=np.int64, count=len(rows))
    row_tails = np.fromiter((row[2] for row in rows), dtype=np.int64, count=len(rows))
    subjects[:] = row_heads[:, None]
    objects[:] = row_tails[:, None]
    random = np.random.default_rng(seed)
    cell_rows = np.repeat(np.arange(len(rows)), count)
    cell_columns = np.tile(np.arange(count), len(rows))
    pairs = row_relations[cell_rows] * 2 + cell_columns % 2
    unique_pairs, inverse = np.unique(pairs, return_inverse=True)
    pool_lengths = np.array(
        [
            len(pool) if (pool := _pool(pools, int(pair) // 2, int(pair) % 2)) is not None else 0
            for pair in unique_pairs.tolist()
        ],
        dtype=np.int64,
    )
    lengths = pool_lengths[inverse]
    starts = np.zeros(cell_rows.size, dtype=np.int64)
    drawable = lengths > 0
    if drawable.any():
        # Row-major draws over exactly the cells the reference implementation
        # would draw for, in the same order.
        starts[drawable] = random.integers(lengths[drawable])
    for pair, length in zip(unique_pairs.tolist(), pool_lengths.tolist(), strict=True):
        if length <= 0:
            continue
        relation, side = int(pair) // 2, int(pair) % 2
        members = np.flatnonzero(pairs == pair)
        pool = _pool(pools, relation, side)
        assert pool is not None
        fixed = row_tails[cell_rows[members]] if side == 0 else row_heads[cell_rows[members]]
        group_starts = starts[members]
        resolved = np.zeros(members.size, dtype=np.bool_)
        values = np.zeros(members.size, dtype=np.int64)
        active = np.arange(members.size)
        offset = 0
        while active.size:
            candidates = pool[(group_starts[active] + offset) % length]
            query = (
                (candidates * context.relation_count + relation) * context.node_count + fixed[active]
                if side == 0
                else (fixed[active] * context.relation_count + relation) * context.node_count + candidates
            )
            accepted = ~_contains(context.keys, query)
            if accepted.any():
                chosen = active[accepted]
                values[chosen] = candidates[accepted]
                resolved[chosen] = True
                active = active[~accepted]
                if not active.size:
                    break
            offset += 1
            if offset >= length:
                break
        chosen = members[resolved]
        if chosen.size:
            if side == 0:
                subjects[cell_rows[chosen], cell_columns[chosen]] = values[resolved]
            else:
                objects[cell_rows[chosen], cell_columns[chosen]] = values[resolved]
            valid[cell_rows[chosen], cell_columns[chosen]] = True
    return subjects, objects, valid
