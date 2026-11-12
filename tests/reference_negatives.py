"""Frozen pre-optimization reference for equivalence testing (do not edit).

Verbatim copy of typed_pools/sample_negatives as of commit 9edde86 plus the
set-based positives contract. tests/test_negatives_equivalence.py proves the
vectorized production implementation returns identical arrays; keeping this
reference in-tree makes that check independent of git history.
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
        for column in range(count):
            corrupt_head = column % 2 == 0
            candidates = subjects.get(relation) if corrupt_head else objects.get(relation)
            subjects_out[index, column], objects_out[index, column] = subject, object_id
            if candidates is None or not len(candidates):
                continue
            # Exhaustive fallback guarantees a valid candidate when one exists.
            start = int(random.integers(len(candidates)))
            for offset in range(len(candidates)):
                candidate = int(candidates[(start + offset) % len(candidates)])
                head, tail = (candidate, object_id) if corrupt_head else (subject, candidate)
                if (head, relation, tail) not in positives:
                    subjects_out[index, column], objects_out[index, column] = head, tail
                    valid[index, column] = True
                    break
    return subjects_out, objects_out, valid
