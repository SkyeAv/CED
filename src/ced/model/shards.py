from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ced.config.models import RuntimeConfig
from ced.model.batches import sample_neighborhood
from ced.model.disk_graph import DiskGraph
from ced.model.negatives import negative_context, sample_negatives, typed_pools
from ced.model.schemas import GraphBatch, NegativeBatch, TrainingExample


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
    # Encoded once for all chunks: repeated per-chunk negative sampling must not
    # re-walk every positive statement in Python.
    context = negative_context(edges[0], relations, edges[1]) if in_memory else None
    parameter_keys: dict[tuple[int, int, int], int] = {}
    edge_types = np.zeros(len(relations), dtype=np.int64)
    for index in np.flatnonzero(splits == 0):
        key = (int(types[edges[0, index]]), int(relations[index]), int(types[edges[1, index]]))
        edge_types[index] = parameter_keys.setdefault(key, len(parameter_keys))
    for split_id, label in enumerate(("train", "validation", "test")):
        selected = np.flatnonzero(splits == split_id)
        for start in range(0, len(selected), chunk_size):
            positive = selected[start : start + chunk_size]
            triples = [(int(edges[0, i]), int(relations[i]), int(edges[1, i])) for i in positive]
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
                neighborhood = sample_neighborhood(
                    edges,
                    relations,
                    splits,
                    seeds,
                    fanout=runtime.batch.neighbor_fanout,
                    seed=runtime.split.seed + start,
                    node_limit=runtime.batch.node_limit,
                    edge_types=edge_types,
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
