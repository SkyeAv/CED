"""Independent reference for typed, uniform, without-replacement negatives.

The reference deliberately materializes each small eligible pool and uses one
NumPy choice per positive and side. It is not an implementation copy: production
adds bounded large-pool draws and deterministic exhaustive fallback.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def typed_pools_reference(
    edge_index: np.ndarray, relations: np.ndarray, splits: np.ndarray, types: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    subjects = {}
    objects = {}
    for relation in np.unique(relations[splits == 0]):
        edges = edge_index[:, (splits == 0) & (relations == relation)]
        subjects[int(relation)] = np.flatnonzero(np.isin(types, np.unique(types[edges[0]]))).astype(np.int64)
        objects[int(relation)] = np.flatnonzero(np.isin(types, np.unique(types[edges[1]]))).astype(np.int64)
    return subjects, objects


def sample_negatives_reference(
    triples: Iterable[tuple[int, int, int]],
    subjects: dict[int, np.ndarray],
    objects: dict[int, np.ndarray],
    positives: set[tuple[int, int, int]],
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(triples)
    subjects_out = np.zeros((len(rows), count), dtype=np.int64)
    objects_out = np.zeros_like(subjects_out)
    valid = np.zeros((len(rows), count), dtype=np.bool_)
    random = np.random.default_rng(seed)
    for index, (subject, relation, object_id) in enumerate(rows):
        subjects_out[index, :], objects_out[index, :] = subject, object_id
        for side in (0, 1):
            columns = np.arange(side, count, 2)
            pool = subjects.get(relation) if side == 0 else objects.get(relation)
            if pool is None or not len(pool):
                continue
            eligible = np.array(
                [
                    int(candidate)
                    for candidate in pool
                    if ((int(candidate), relation, object_id) if side == 0 else (subject, relation, int(candidate)))
                    not in positives
                ],
                dtype=np.int64,
            )
            chosen = random.choice(eligible, size=min(len(columns), len(eligible)), replace=False)
            for column, candidate in zip(columns, chosen, strict=False):
                if side == 0:
                    subjects_out[index, column] = candidate
                else:
                    objects_out[index, column] = candidate
                valid[index, column] = True
    return subjects_out, objects_out, valid
