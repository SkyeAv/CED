from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SampledNeighborhood:
    global_node_id: np.ndarray
    edge_index: np.ndarray
    statement_id: np.ndarray
    relation_id: np.ndarray
    edge_type_id: np.ndarray


def sample_neighborhood(
    edge_index: np.ndarray,
    relations: np.ndarray,
    splits: np.ndarray,
    seeds: np.ndarray,
    *,
    fanout: list[int],
    seed: int,
    node_limit: int,
    edge_types: np.ndarray | None = None,
) -> SampledNeighborhood:
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
    return SampledNeighborhood(
        global_node_id=global_ids,
        edge_index=np.concatenate((local, local[::-1]), axis=1),
        statement_id=np.tile(statement_ids, 2),
        relation_id=np.concatenate((relations[statement_ids], relations[statement_ids] + relation_count)),
        edge_type_id=np.tile(parameter_ids[statement_ids], 2),
    )
