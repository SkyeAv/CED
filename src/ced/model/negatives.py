from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TypedPools:
    subjects: dict[int, np.ndarray]
    objects: dict[int, np.ndarray]


def typed_pools(edge_index: np.ndarray, relations: np.ndarray, splits: np.ndarray, types: np.ndarray) -> TypedPools:
    subjects = {}
    objects = {}
    for relation in np.unique(relations[splits == 0]):
        edges = edge_index[:, (splits == 0) & (relations == relation)]
        subjects[int(relation)] = np.flatnonzero(np.isin(types, np.unique(types[edges[0]]))).astype(np.int64)
        objects[int(relation)] = np.flatnonzero(np.isin(types, np.unique(types[edges[1]]))).astype(np.int64)
    return TypedPools(subjects=subjects, objects=objects)


def sample_negatives(
    triples: Iterable[tuple[int, int, int]],
    pools: TypedPools,
    positives: set[tuple[int, int, int]],
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(triples)
    subjects = np.zeros((len(rows), count), dtype=np.int64)
    objects = np.zeros_like(subjects)
    valid = np.zeros((len(rows), count), dtype=np.bool_)
    random = np.random.default_rng(seed)
    for index, (subject, relation, object_id) in enumerate(rows):
        for column in range(count):
            corrupt_head = column % 2 == 0
            candidates = pools.subjects.get(relation) if corrupt_head else pools.objects.get(relation)
            subjects[index, column], objects[index, column] = subject, object_id
            if candidates is None or not len(candidates):
                continue
            # Exhaustive fallback guarantees a valid candidate when one exists.
            start = int(random.integers(len(candidates)))
            for offset in range(len(candidates)):
                candidate = int(candidates[(start + offset) % len(candidates)])
                head, tail = (candidate, object_id) if corrupt_head else (subject, candidate)
                if (head, relation, tail) not in positives:
                    subjects[index, column], objects[index, column] = head, tail
                    valid[index, column] = True
                    break
    return subjects, objects, valid
