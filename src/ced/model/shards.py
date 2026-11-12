from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ced.config.models import RuntimeConfig
from ced.model.batches import TrainAdjacency, sample_neighborhood
from ced.model.disk_graph import DiskGraph
from ced.model.negatives import negative_context, sample_negatives, typed_pools
from ced.model.schemas import GraphBatch, NegativeBatch, TrainingExample


def _edge_type_ids(edge_index: np.ndarray, relations: np.ndarray, splits: np.ndarray, types: np.ndarray) -> np.ndarray:
    """Assign train edge-type ids by first occurrence without a Python row loop."""
    edge_types = np.zeros(len(relations), dtype=np.int64)
    train = np.flatnonzero(splits == 0)
    if not train.size:
        return edge_types
    keys = np.column_stack((types[edge_index[0, train]], relations[train], types[edge_index[1, train]]))
    relation_count = int(relations.max()) + 1
    type_count = int(types.max()) + 1
    # One-dimensional unique is substantially faster than structured row unique.
    # Retain the overflow-free row path for arbitrary negative or huge ids.
    if np.all(keys >= 0) and 0 < relation_count * type_count * type_count < 2**63:
        encoded = (keys[:, 0] * relation_count + keys[:, 1]) * type_count + keys[:, 2]
        _, first_positions, inverse = np.unique(encoded, return_index=True, return_inverse=True)
    else:
        _, first_positions, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    appearance = np.argsort(first_positions)
    renumbered = np.empty(len(first_positions), dtype=np.int64)
    renumbered[appearance] = np.arange(len(first_positions), dtype=np.int64)
    edge_types[train] = renumbered[inverse]
    return edge_types


def _positive_triples(
    edge_index: np.ndarray, relations: np.ndarray, positive: np.ndarray
) -> list[tuple[int, int, int]]:
    """Convert selected columns to Python triples without per-row scalar indexing."""
    return list(
        zip(
            edge_index[0, positive].tolist(),
            relations[positive].tolist(),
            edge_index[1, positive].tolist(),
            strict=True,
        )
    )


def write_batch_shards(
    root: Path, arrays: Mapping[str, np.ndarray], runtime: RuntimeConfig, *, in_memory: bool = True
) -> list[TrainingExample]:
    from ced.model.prepare import save_tensor

    head, tail, relations, splits, types = (
        arrays[key] for key in ("head", "tail", "relation_id", "split", "type_set_id")
    )
    # The PyG convention (2, E) is built once from the streamed head/tail columns;
    # nothing downstream keeps a transposed copy on disk.
    edges = np.stack((head, tail))
    examples = []
    # Keep seed positives small enough to accommodate all global negative endpoints.
    chunk_size = max(1, runtime.batch.node_limit // (2 * (runtime.split.negatives + 1)))
    disk: DiskGraph | None = None
    pools = typed_pools(edges, relations, splits, types) if in_memory else None
    # Built once for all chunks: the historical per-call full-edge mask scan was
    # the dominant prepare cost on large graphs.
    adjacency = TrainAdjacency(edges, splits) if in_memory else None
    # Encoded once for all chunks: repeated per-chunk negative sampling must not
    # re-walk every positive statement in Python.
    context = negative_context(edges[0], relations, edges[1]) if in_memory else None
    edge_types = _edge_type_ids(edges, relations, splits, types)
    for split_id, label in enumerate(("train", "validation", "test")):
        selected = np.flatnonzero(splits == split_id)
        for start in range(0, len(selected), chunk_size):
            positive = selected[start : start + chunk_size]
            triples = _positive_triples(edges, relations, positive)
            if in_memory:
                assert pools is not None
                assert context is not None
                head, tail, valid = sample_negatives(
                    triples, pools, context, count=runtime.split.negatives, seed=runtime.split.seed + start + split_id
                )
            else:
                disk = disk or DiskGraph(root / "graph.db", arrays)
                assert disk is not None
                head, tail, valid = disk.negatives(
                    triples, count=runtime.split.negatives, seed=runtime.split.seed + start + split_id
                )
            seeds = np.unique(np.concatenate((edges[:, positive].ravel(), head[valid], tail[valid])))
            if in_memory:
                assert adjacency is not None
                neighborhood = sample_neighborhood(
                    edges,
                    relations,
                    splits,
                    seeds,
                    fanout=runtime.batch.neighbor_fanout,
                    seed=runtime.split.seed + start,
                    node_limit=runtime.batch.node_limit,
                    edge_types=edge_types,
                    adjacency=adjacency,
                )
            else:
                assert disk is not None
                neighborhood = disk.neighborhood(
                    edges,
                    relations,
                    seeds,
                    fanout=runtime.batch.neighbor_fanout,
                    seed=runtime.split.seed + start,
                    node_limit=runtime.batch.node_limit,
                    edge_types=edge_types,
                )
            directory = root / "batches" / f"{label}-{start:09d}"
            directory.mkdir(parents=True)

            def save(name: str, values: np.ndarray, directory: Path = directory):
                ref = save_tensor(directory, name, values)
                ref.path = str((directory / ref.path).relative_to(root))
                return ref

            graph = GraphBatch(
                global_node_id=save("global_node_id", neighborhood.global_node_id),
                seed_id=save("seed_id", seeds),
                type_set_id=save("type_set_id", types[neighborhood.global_node_id]),
                edge_index=save("edge_index", neighborhood.edge_index),
                relation_id=save("relation_id", neighborhood.relation_id),
                edge_type_id=save("edge_type_id", neighborhood.edge_type_id),
                statement_id=save("statement_id", neighborhood.statement_id),
                positive_statement_id=save("positive_statement_id", positive.astype(np.int64)),
                negative=NegativeBatch(
                    subject=save("negative_subject", head),
                    object=save("negative_object", tail),
                    valid=save("negative_valid", valid),
                ),
            )
            examples.append(TrainingExample(graph=graph, split=label))
    if disk is not None:
        disk.close()
        (root / "graph.db").unlink()
    return examples
