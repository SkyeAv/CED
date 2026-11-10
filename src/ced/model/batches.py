from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SampledNeighborhood:
    global_node_id: np.ndarray
    edge_index: np.ndarray
    statement_id: np.ndarray
    relation_id: np.ndarray
    edge_type_id: np.ndarray


class TrainAdjacency:
    """CSR adjacency over training statements, built once and reused per chunk.

    Statement indices inside each node block are ascending, matching the order
    of the historical boolean-mask scan (train[(head==n) | (tail==n)]) so RNG
    consumption and sampled results are unchanged. A self-loop statement is
    listed exactly once because the mask OR cannot duplicate a position.
    """

    def __init__(self, edge_index: np.ndarray, splits: np.ndarray) -> None:
        self.train = np.flatnonzero(splits == 0)
        node_count = int(edge_index.max()) + 1 if edge_index.size else 0
        heads = edge_index[0, self.train]
        tails = edge_index[1, self.train]
        keep = heads != tails
        nodes = np.concatenate((heads, tails[keep]))
        statements = np.concatenate((self.train, self.train[keep]))
        order = np.lexsort((statements, nodes))
        self.statements = statements[order]
        self.indptr = np.zeros(node_count + 1, dtype=np.int64)
        if node_count:
            np.cumsum(np.bincount(nodes, minlength=node_count), out=self.indptr[1:])

    def neighbors(self, node: int) -> np.ndarray:
        if node >= self.indptr.size - 1:
            return np.empty(0, dtype=np.int64)
        return self.statements[self.indptr[node] : self.indptr[node + 1]]


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
    adjacency: TrainAdjacency | None = None,
) -> SampledNeighborhood:
    """Use only training statements; endpoint IDs remain global until final relabeling.

    adjacency may be supplied to reuse one CSR across many calls (production
    chunking); when omitted it is built here, matching the historical per-call
    boolean-mask scan at O(frontier x degree) instead of O(frontier x edges).
    """
    random = np.random.default_rng(seed)
    nodes = set(map(int, seeds))
    if len(nodes) > node_limit:
        raise ValueError("seed and negative endpoints exceed batch node limit")
    frontier = sorted(nodes)
    selected: set[int] = set()
    lookup = adjacency if adjacency is not None else TrainAdjacency(edge_index, splits)
    for limit in fanout:
        if limit < 0:
            raise ValueError("neighbor fanout must be nonnegative")
        following = set()
        for node in frontier:
            candidates = lookup.neighbors(node)
            if len(candidates) > limit:
                candidates = random.choice(candidates, size=limit, replace=False)
            if not candidates.size:
                continue
            # One bulk endpoint fetch per node; the per-candidate limit check
            # depends on the growing node set, so the scan stays ordered.
            ordered = np.sort(candidates)
            endpoints = edge_index[:, ordered]
            size = len(nodes)
            for position in range(ordered.size):
                head = int(endpoints[0, position])
                tail = int(endpoints[1, position])
                additions = (head not in nodes) + (tail not in nodes and tail != head)
                if size + additions > node_limit:
                    continue
                selected.add(int(ordered[position]))
                if head not in nodes:
                    following.add(head)
                if tail not in nodes and tail != head:
                    following.add(tail)
                nodes.add(head)
                nodes.add(tail)
                size += additions
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
