from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

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


_EXHAUSTIVE_POOL_LIMIT = 4096
_REJECTION_ROUNDS = 8


def _encoded(candidates: np.ndarray, relation: int, side: int, fixed: int, context: NegativeContext) -> np.ndarray:
    """Encode one candidate array without per-candidate Python membership checks."""
    fixed_array = np.full(candidates.shape, fixed, dtype=np.int64)
    return (
        (candidates * context.relation_count + relation) * context.node_count + fixed_array
        if side == 0
        else (fixed_array * context.relation_count + relation) * context.node_count + candidates
    )


def _eligible(pool: np.ndarray, relation: int, side: int, fixed: int, context: NegativeContext) -> np.ndarray:
    # Nodes outside the positive index cannot be positives. Encoding them with
    # the smaller positive-only radix can alias a different known triple.
    bounded = (pool >= 0) & (pool < context.node_count)
    known = np.zeros(pool.shape, dtype=np.bool_)
    if relation < context.relation_count and fixed < context.node_count:
        known[bounded] = _contains(context.keys, _encoded(pool[bounded], relation, side, fixed, context))
    return pool[~known]


def choose_candidates(
    size: int,
    count: int,
    random: np.random.Generator,
    lookup: Callable[[np.ndarray], np.ndarray],
    eligible: Callable[[np.ndarray], np.ndarray],
    exhaustive: Callable[[], np.ndarray],
) -> np.ndarray:
    """Draw a uniform ordered subset, bounding work before an exact fallback."""
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if size <= _EXHAUSTIVE_POOL_LIMIT or count >= size:
        available = exhaustive()
        return random.choice(available, size=min(count, len(available)), replace=False)
    chosen: list[int] = []
    drawn: set[int] = set()
    for _ in range(_REJECTION_ROUNDS):
        remaining = count - len(chosen)
        if remaining == 0:
            return np.asarray(chosen, dtype=np.int64)
        draws = lookup(random.choice(size, size=min(size, max(remaining * 2, 16)), replace=False))
        fresh = [int(candidate) for candidate in draws if int(candidate) not in drawn]
        if not fresh:
            continue
        drawn.update(fresh)
        candidates = np.asarray(fresh, dtype=np.int64)
        accepted = eligible(candidates)
        chosen.extend(int(candidate) for candidate in accepted[:remaining])
    available = exhaustive()
    if chosen:
        available = available[~np.isin(available, np.asarray(chosen, dtype=np.int64))]
    remaining = count - len(chosen)
    selected = random.choice(available, size=min(remaining, len(available)), replace=False)
    return np.asarray([*chosen, *selected.tolist()], dtype=np.int64)


def sample_negatives(
    triples: Iterable[tuple[int, int, int]],
    pools: TypedPools,
    context: NegativeContext,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample typed corruptions uniformly without replacement on each side.

    Invalid cells retain the original endpoints and are marked false. A bounded
    rejection phase handles large sparse pools; exact eligible enumeration is
    used for small pools and as the deterministic fallback for dense pools.
    """
    if count < 0:
        raise ValueError("negative count must be nonnegative")
    rows = list(triples)
    subjects = np.zeros((len(rows), count), dtype=np.int64)
    objects = np.zeros_like(subjects)
    valid = np.zeros((len(rows), count), dtype=np.bool_)
    if not rows or count == 0:
        return subjects, objects, valid
    random = np.random.default_rng(seed)
    for index, (head, relation, tail) in enumerate(rows):
        subjects[index, :], objects[index, :] = head, tail
        for side in (0, 1):
            columns = np.arange(side, count, 2)
            pool = _pool(pools, relation, side)
            if pool is None or len(pool) == 0:
                continue
            fixed = tail if side == 0 else head
            eligible = partial(_eligible, relation=relation, side=side, fixed=fixed, context=context)
            chosen = choose_candidates(
                len(pool), len(columns), random, pool.__getitem__, eligible, partial(eligible, pool)
            )
            for column, candidate in zip(columns, chosen, strict=False):
                if side == 0:
                    subjects[index, column] = candidate
                else:
                    objects[index, column] = candidate
                valid[index, column] = True
    return subjects, objects, valid
