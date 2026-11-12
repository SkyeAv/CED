"""Perf benchmarks for the pipeline/training hotspots (marker `perf`).

Why this exists: the performance stories in this worktree each change one
hotspot and must prove a wall-clock improvement without changing outputs.
These benchmarks are the mechanical evidence: fixed synthetic workloads,
seeded deterministically, run on wenceslaus with CED_CPU_WORKERS capped.
Run: pytest -m perf -q   (excluded from the default suite).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from perf_support import feature_tables, synthetic_graph, timed, topology_tables, write_record

pytestmark = pytest.mark.perf


def test_perf_typed_pools_and_negatives() -> None:
    from ced.model.negatives import negative_context, sample_negatives, typed_pools

    graph = synthetic_graph(nodes=2000, edges=20000, relations=8, type_sets=4, seed=1)
    edges, relations, splits, types = (
        graph["edge_index"],
        graph["relation_id"],
        graph["split"],
        graph["type_set_id"],
    )
    train = np.flatnonzero(splits == 0)
    triple_indices = train[np.linspace(0, len(train) - 1, 200).astype(int)]
    triples = [(int(edges[0, i]), int(relations[i]), int(edges[1, i])) for i in triple_indices]
    positives = {(int(h), int(r), int(t)) for h, r, t in zip(edges[0], relations, edges[1], strict=True)}
    _ = positives  # retained for workload parity; production passes encoded keys
    context = negative_context(edges[0], relations, edges[1])
    workload = {"nodes": 2000, "edges": 20000, "relations": 8, "type_sets": 4, "triples": len(triples), "count": 8}

    pools_seconds = timed(lambda: typed_pools(edges, relations, splits, types), iterations=3)
    pools = typed_pools(edges, relations, splits, types)
    negatives_seconds = timed(lambda: sample_negatives(triples, pools, context, count=8, seed=7), iterations=3)
    write_record("typed_pools", seconds=pools_seconds, iterations=3, workload=workload)
    write_record("sample_negatives", seconds=negatives_seconds, iterations=3, workload=workload)


def test_perf_neighborhood() -> None:
    from ced.model.batches import TrainAdjacency, sample_neighborhood

    graph = synthetic_graph(nodes=5000, edges=50000, relations=8, type_sets=4, seed=2)
    edges, relations, splits = graph["edge_index"], graph["relation_id"], graph["split"]
    seeds = np.unique(edges[0, np.flatnonzero(splits == 0)[:100]])
    workload = {"nodes": 5000, "edges": 50000, "relations": 8, "seeds": int(seeds.size), "fanout": [15, 10]}
    seconds = timed(
        lambda: sample_neighborhood(
            edges, relations, splits, seeds, fanout=[15, 10], seed=7, node_limit=4096, edge_types=graph["edge_type_id"]
        ),
        iterations=3,
    )
    write_record("sample_neighborhood", seconds=seconds, iterations=3, workload=workload)
    # Production chunking builds the CSR once and reuses it per chunk.
    adjacency = TrainAdjacency(edges, splits)
    shared_seconds = timed(
        lambda: sample_neighborhood(
            edges,
            relations,
            splits,
            seeds,
            fanout=[15, 10],
            seed=7,
            node_limit=4096,
            edge_types=graph["edge_type_id"],
            adjacency=adjacency,
        ),
        iterations=3,
    )
    write_record("sample_neighborhood_shared_adjacency", seconds=shared_seconds, iterations=3, workload=workload)


def test_perf_prepare_topology(tmp_path: Path) -> None:
    import duckdb

    from ced.config.models import SplitSettings
    from ced.model.prepare import _write_topology

    entities, predicates, statements = 500, 20, 20000
    connection = duckdb.connect()
    try:
        topology_tables(connection, entities=entities, predicates=predicates, statements=statements, seed=13)
        arrays = {
            "edge_index": np.zeros((2, statements), dtype=np.int64),
            "relation_id": np.zeros(statements, dtype=np.int64),
            "split": np.zeros(statements, dtype=np.int8),
        }
        settings = SplitSettings()
        records = tmp_path / "split-records.ndjson"
        workload = {"entities": entities, "predicates": predicates, "statements": statements}
        seconds = timed(
            lambda: _write_topology(connection, arrays, records, settings, statements),
            iterations=3,
        )
        write_record("prepare_topology", seconds=seconds, iterations=3, workload=workload)
    finally:
        connection.close()


def test_perf_prepare_topology_memmap(tmp_path: Path) -> None:
    import duckdb

    from ced.config.models import SplitSettings
    from ced.model.prepare import _write_topology

    statements = 20000
    connection = duckdb.connect()
    try:
        topology_tables(connection, entities=500, predicates=20, statements=statements, seed=13)
        arrays = {
            "edge_index": np.lib.format.open_memmap(
                tmp_path / "edge_index.npy", mode="w+", dtype=np.int64, shape=(2, statements)
            ),
            "relation_id": np.lib.format.open_memmap(
                tmp_path / "relation_id.npy", mode="w+", dtype=np.int64, shape=(statements,)
            ),
            "split": np.lib.format.open_memmap(tmp_path / "split.npy", mode="w+", dtype=np.int8, shape=(statements,)),
        }
        settings = SplitSettings()
        records = tmp_path / "split-records.ndjson"
        seconds = timed(
            lambda: _write_topology(connection, arrays, records, settings, statements),
            iterations=3,
        )
        write_record(
            "prepare_topology_memmap",
            seconds=seconds,
            iterations=3,
            workload={"statements": statements, "backing": "memmap"},
        )
    finally:
        connection.close()


def test_perf_write_ordered(tmp_path: Path) -> None:
    import duckdb

    from ced.model.feature_store import _write_ordered

    owners, fields = 2000, 6
    connection = duckdb.connect()
    try:
        feature_tables(connection, owners=owners, fields=fields, seed=17)
        categorical_query = (
            "SELECT owner_index, field_id, ordinal, value_id FROM categorical_events "
            "ORDER BY owner_index, field_id, ordinal"
        )
        numeric_query = (
            "SELECT owner_index, field_id, numeric, valid FROM numeric_events ORDER BY owner_index, field_id"
        )
        workload = {"owners": owners, "fields": fields}
        categorical_root = tmp_path / "categorical"
        numeric_root = tmp_path / "numeric"
        categorical_root.mkdir()
        numeric_root.mkdir()
        categorical_seconds = timed(
            lambda: _write_ordered(connection, categorical_query, categorical_root, "categorical", owners),
            iterations=3,
        )
        numeric_seconds = timed(
            lambda: _write_ordered(connection, numeric_query, numeric_root, "numeric", owners),
            iterations=3,
        )
        write_record("write_ordered_categorical", seconds=categorical_seconds, iterations=3, workload=workload)
        write_record("write_ordered_numeric", seconds=numeric_seconds, iterations=3, workload=workload)
    finally:
        connection.close()


def test_perf_encoder_forward() -> None:
    torch = pytest.importorskip("torch")
    from ced.model.encoder import RelationalAttentionEncoder

    graph = synthetic_graph(nodes=5000, edges=20000, relations=8, type_sets=4, seed=3)
    edge_index = torch.from_numpy(graph["edge_index"])
    edge_type_id = torch.from_numpy(graph["edge_type_id"])
    nodes = torch.randn(5000, 32)
    statements = torch.randn(edge_index.shape[1], 16)
    edge_types = int(edge_type_id.max()) + 1
    encoder = RelationalAttentionEncoder(32, 16, edge_types=edge_types, layers=2)
    workload = {"nodes": 5000, "edges": 20000, "edge_types": edge_types, "layers": 2, "dimension": 32}
    seconds = timed(
        lambda: encoder(nodes, edge_index, edge_type_id, statements),
        iterations=3,
    )
    write_record("encoder_forward", seconds=seconds, iterations=3, workload=workload)


def test_perf_auxiliary_heads() -> None:
    torch = pytest.importorskip("torch")
    from ced.model.losses import AuxiliaryAttributeDecoder

    states = torch.randn(5000, 32)
    categorical_fields = torch.randint(1, 9, (5000,))
    numeric_fields = torch.randint(1, 9, (5000,))
    decoder = AuxiliaryAttributeDecoder(32, dict.fromkeys(range(1, 9), 16), range(1, 9))
    workload = {"states": 5000, "fields": 8, "dimension": 32}
    seconds = timed(
        lambda: decoder(
            states,
            categorical_fields,
            torch.zeros(5000, dtype=torch.long),
            numeric_fields,
            torch.zeros(5000),
            torch.ones(5000, dtype=torch.bool),
        ),
        iterations=5,
    )
    write_record("auxiliary_heads", seconds=seconds, iterations=5, workload=workload)


def test_perf_train_step(tmp_path: Path) -> None:
    import torch

    from ced.model.prepare import save_tensor
    from ced.model.schemas import GraphBatch, NegativeBatch, TrainingExample
    from ced.model.training import build_model, train_step

    graph = synthetic_graph(nodes=3000, edges=12000, relations=6, type_sets=3, seed=4)
    nodes = np.unique(graph["edge_index"].ravel())
    # train_step indexes statement_features by edge_type_id, so keep it within relations.
    graph["edge_type_id"] = graph["relation_id"].copy()
    # Real batches carry locally relabeled edge indices (see batches.py np.searchsorted).
    edge_index = np.searchsorted(nodes, graph["edge_index"]).astype(np.int64)
    directory = tmp_path / "batch"
    directory.mkdir()

    def save(name: str, values: np.ndarray):
        reference = save_tensor(directory, name, values)
        reference.path = f"batch/{reference.path}"
        return reference

    count = 500
    valid = np.ones((count, 4), dtype=bool)
    batch = GraphBatch(
        global_node_id=save("global_node_id", nodes.astype(np.int64)),
        seed_id=save("seed_id", np.array([0], dtype=np.int64)),
        type_set_id=save("type_set_id", graph["type_set_id"][nodes].astype(np.int64)),
        edge_index=save("edge_index", edge_index),
        relation_id=save("relation_id", graph["relation_id"]),
        edge_type_id=save("edge_type_id", graph["edge_type_id"]),
        statement_id=save("statement_id", np.arange(graph["edge_index"].shape[1], dtype=np.int64)),
        positive_statement_id=save("positive_statement_id", np.arange(count, dtype=np.int64)),
        negative=NegativeBatch(
            subject=save("negative_subject", np.tile(nodes[:1], (count, 4))),
            object=save("negative_object", np.tile(nodes[1:2], (count, 4))),
            valid=save("negative_valid", valid),
        ),
    )
    document = {"relations": 6, "entities": int(nodes.size), "statements": int(graph["edge_index"].shape[1])}
    edge_types = int(graph["edge_type_id"].max()) + 1
    model = build_model(document, tmp_path, edge_types, seed=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    workload = {"nodes": int(nodes.size), "edges": 12000, "positives": count, "negatives": 4, "features": "none"}
    seconds = timed(
        lambda: train_step(
            model,
            optimizer,
            tmp_path,
            document,
            batch,
            None,
            device=torch.device("cpu"),
            auxiliary_weight=0.1,
        ),
        iterations=3,
    )
    _ = TrainingExample(graph=batch, split="train")
    write_record("train_step", seconds=seconds, iterations=3, workload=workload)


def test_perf_evaluation_helpers() -> None:
    torch = pytest.importorskip("torch")
    from ced.model.evaluation import _build_positives, _full_messages

    graph = synthetic_graph(nodes=5000, edges=20000, relations=8, type_sets=4, seed=5)
    edge_index = torch.from_numpy(graph["edge_index"])
    relations = torch.from_numpy(graph["relation_id"])
    splits = torch.from_numpy(graph["split"].astype(np.int64))
    types = torch.from_numpy(graph["type_set_id"])
    workload = {"nodes": 5000, "edges": 20000, "relations": 8}

    messages_seconds = timed(lambda: _full_messages(edge_index, relations, splits, types), iterations=3)
    positives_seconds = timed(lambda: _build_positives(edge_index, relations), iterations=3)
    write_record("evaluation_full_messages", seconds=messages_seconds, iterations=3, workload=workload)
    write_record("evaluation_positives", seconds=positives_seconds, iterations=3, workload=workload)


def test_perf_evaluation_relation_scoring(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    torch = pytest.importorskip("torch")
    from ced.model.evaluation import _score_relation_batch
    from ced.model.training import build_model

    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    document = json.loads((root / "manifest.json").read_text())
    model = build_model(document, root, edge_type_count=2, seed=5)
    nodes, targets = 5000, 32
    state = torch.randn(nodes, 32, generator=torch.Generator().manual_seed(7))
    heads = [index % nodes for index in range(targets)]
    tails = [(index * 17 + 1) % nodes for index in range(targets)]
    relation_ids = torch.full((nodes,), 0, dtype=torch.long)

    def reference_scores() -> None:
        with torch.no_grad():
            for head, tail in zip(heads, tails, strict=True):
                model.decoder.score(state[head].expand_as(state), relation_ids, state)
                model.decoder.score(state, relation_ids, state[tail].expand_as(state))

    def batched_scores() -> None:
        with torch.no_grad():
            _score_relation_batch(model, state, heads, tails, 0)

    workload = {"nodes": nodes, "targets": targets, "relations_per_batch": 1, "dimension": 32}
    reference_seconds = timed(reference_scores, iterations=3)
    batched_seconds = timed(batched_scores, iterations=3)
    write_record("evaluation_scoring_reference", seconds=reference_seconds, iterations=3, workload=workload)
    write_record("evaluation_scoring_batched", seconds=batched_seconds, iterations=3, workload=workload)


def test_perf_shards_construction() -> None:
    from ced.model.shards import _edge_type_ids, _positive_triples

    graph = synthetic_graph(nodes=2000, edges=20000, relations=8, type_sets=4, seed=31)
    edges, relations = graph["edge_index"], graph["relation_id"]
    splits, types = graph["split"], graph["type_set_id"]
    train = np.flatnonzero(splits == 0)

    def reference_edge_types() -> np.ndarray:
        parameter_keys: dict[tuple[int, int, int], int] = {}
        edge_types = np.zeros(len(relations), dtype=np.int64)
        for index in train:
            key = (int(types[edges[0, index]]), int(relations[index]), int(types[edges[1, index]]))
            edge_types[index] = parameter_keys.setdefault(key, len(parameter_keys))
        return edge_types

    positive = train[:2048]

    def reference_triples() -> list[tuple[int, int, int]]:
        return [(int(edges[0, i]), int(relations[i]), int(edges[1, i])) for i in positive]

    assert np.array_equal(_edge_type_ids(edges, relations, splits, types), reference_edge_types())
    assert _positive_triples(edges, relations, positive) == reference_triples()
    workload = {"nodes": 2000, "edges": 20000, "train_edges": len(train), "triples": len(positive)}
    edge_reference = timed(reference_edge_types, iterations=3)
    edge_vectorized = timed(lambda: _edge_type_ids(edges, relations, splits, types), iterations=3)
    triple_reference = timed(reference_triples, iterations=3)
    triple_vectorized = timed(lambda: _positive_triples(edges, relations, positive), iterations=3)
    write_record("shards_edge_types_reference", seconds=edge_reference, iterations=3, workload=workload)
    write_record("shards_edge_types", seconds=edge_vectorized, iterations=3, workload=workload)
    write_record("shards_triples_reference", seconds=triple_reference, iterations=3, workload=workload)
    write_record("shards_triples", seconds=triple_vectorized, iterations=3, workload=workload)


def test_perf_merge_nodes(tmp_path: Path) -> None:
    import duckdb
    from reference_merge import merge_nodes_reference

    from ced.kg.merge import merge_nodes
    from ced.kg.records import CanonicalBatch, NodeRecord
    from ced.kg.writer import CanonicalBatchWriter

    identities, duplicates = 1000, 8

    def write_fixture(root: Path) -> None:
        nodes = [
            NodeRecord(
                id=f"X:{identity}",
                names=[f"Name {identity}", f"Alias {duplicate}"],
                categories=[f"biolink:Category{duplicate % 3}"],
                provided_by=[f"source-{duplicate}"],
                original_label=f"Name {identity}",
                native_ids={"shared": f"N:{identity}", f"source-{duplicate}": f"N:{identity}"},
                attributes={"duplicate": duplicate},
            )
            for identity in range(identities)
            for duplicate in range(duplicates)
        ]
        writer = CanonicalBatchWriter(root / "raw", shard_rows=512)
        writer.write(CanonicalBatch(nodes=nodes))
        writer.close()

    def measure(label: str, implementation: Callable[..., int]) -> None:
        roots = [tmp_path / f"{label}-{iteration}" for iteration in range(3)]
        for root in roots:
            write_fixture(root)
        remaining = iter(roots)

        def run() -> None:
            connection = duckdb.connect()
            try:
                connection.execute("SET threads=4")
                implementation(next(remaining), connection, group_rows=256)
            finally:
                connection.close()

        seconds = timed(run, iterations=3, warmup=0)
        workload = {"identities": identities, "duplicates": duplicates, "rows": identities * duplicates}
        write_record(label, seconds=seconds, iterations=3, workload=workload)

    measure("merge_nodes_reference", merge_nodes_reference)
    measure("merge_nodes", merge_nodes)
