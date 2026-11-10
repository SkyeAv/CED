"""Frozen pre-optimization reference for sample_neighborhood (do not edit).

Verbatim copy of batches.sample_neighborhood as of commit c0a7e8f (which was
byte-identical to 9edde86). tests/test_batches_equivalence.py proves the CSR
implementation returns identical arrays; keeping this reference in-tree makes
that check independent of git history.
"""

from __future__ import annotations

import numpy as np


def sample_neighborhood_reference(
    edge_index: np.ndarray,
    relations: np.ndarray,
    splits: np.ndarray,
    seeds: np.ndarray,
    *,
    fanout: list[int],
    seed: int,
    node_limit: int,
    edge_types: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use only training statements; endpoint IDs remain global until final relabeling."""
    random = np.random.default_rng(seed)
    nodes = set(map(int, seeds))
    if len(nodes) > node_limit:
        raise ValueError("seed and negative endpoints exceed batch node limit")
    frontier = sorted(nodes)
    selected: set[int] = set()
    train = np.flatnonzero(splits == 0)
    for limit in fanout:
        if limit < 0:
            raise ValueError("neighbor fanout must be nonnegative")
        following = set()
        for node in frontier:
            candidates = train[(edge_index[0, train] == node) | (edge_index[1, train] == node)]
            if len(candidates) > limit:
                candidates = random.choice(candidates, size=limit, replace=False)
            for index in sorted(map(int, candidates)):
                endpoints = set(map(int, edge_index[:, index]))
                if len(nodes | endpoints) > node_limit:
                    continue
                selected.add(index)
                following.update(endpoints - nodes)
                nodes.update(endpoints)
        frontier = sorted(following)
    global_ids = np.array(sorted(nodes), dtype=np.int64)
    statement_ids = np.array(sorted(selected), dtype=np.int64)
    local = np.searchsorted(global_ids, edge_index[:, statement_ids]).astype(np.int64)
    # Reverse messages are internal, reference the same statement, and have distinct relation IDs.
    relation_count = int(relations.max()) + 1 if relations.size else 0
    parameter_ids = np.zeros(len(relations), dtype=np.int64) if edge_types is None else edge_types
    return (
        global_ids,
        np.concatenate((local, local[::-1]), axis=1),
        np.tile(statement_ids, 2),
        np.concatenate((relations[statement_ids], relations[statement_ids] + relation_count)),
        np.tile(parameter_ids[statement_ids], 2),
    )
