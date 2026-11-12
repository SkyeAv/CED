from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

from ced.acquisition.base import atomic_write_json, sha256_file
from ced.config.models import RuntimeConfig
from ced.kg.audit import validate_checksums
from ced.model.feature_store import prepare_features
from ced.model.pruning import prune_isolated_trees
from ced.model.schemas import DatasetManifest, TensorRef, VocabularyArtifact
from ced.model.shards import write_batch_shards
from ced.model.split import split_bounds
from ced.model.tensors import write_tensor
from ced.preflight import require_cpu_worker, require_disk


def save_tensor(root: Path, name: str, values: np.ndarray) -> TensorRef:
    """Serialize one in-memory tensor to safetensors and return its manifest reference."""
    array = np.ascontiguousarray(values)
    path = root / f"{name}.safetensors"
    write_tensor(path, array)
    return TensorRef(path=path.name, dtype=str(array.dtype), shape=list(array.shape), sha256=sha256_file(path))


def prepare_dataset(runtime: RuntimeConfig, snapshot: str) -> Path:
    """Create disk-backed numeric topology without collecting the graph in Python."""
    require_cpu_worker(runtime)
    source = runtime.artifact_path("snapshots", snapshot)
    manifest = json.loads((source / "manifest.json").read_text())
    validate_checksums(source, json.loads((source / "checksums.json").read_text()))
    dataset_id = hashlib.sha256(
        f"{snapshot}\0{runtime.split.model_dump_json()}\0{runtime.preprocessing.model_dump_json()}\0topology-v4".encode()
    ).hexdigest()
    target = runtime.artifact_path("datasets", dataset_id)
    if target.exists():
        raise FileExistsError(target)
    root = target.with_name(target.name + ".partial")
    if root.exists():
        raise RuntimeError(f"incomplete dataset requires explicit repair: {root}")
    counts = manifest["counts"]
    require_disk(runtime.artifact_path(), counts["statements"] * 128 + counts["nodes"] * 32, runtime.disk_reserve_gb)
    root.mkdir(parents=True)
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={runtime.cpu_workers}")
        connection.execute("SET memory_limit=?", [runtime.memory_limit])
        connection.execute("SET temp_directory=?", [str(root / "spill")])
        connection.read_parquet(str(source / "nodes/*.parquet")).create_view("nodes")
        connection.read_parquet(str(source / "statements/*.parquet")).create_view("statements")
        pruning = prune_isolated_trees(connection, runtime.preprocessing.isolated_tree_max_nodes)
        if pruning.input_nodes != counts["nodes"] or pruning.input_statements != counts["statements"]:
            raise ValueError("snapshot manifest counts do not match pruning inputs")
        entity_count = pruning.input_nodes - pruning.pruned_nodes
        total = pruning.input_statements - pruning.pruned_statements
        if entity_count == 0 or total == 0:
            raise ValueError("isolated-tree pruning must retain at least one node and one statement")
        connection.execute(
            """CREATE TABLE entities AS
            SELECT n.id, CAST(row_number() OVER (ORDER BY n.id) - 1 AS BIGINT) AS idx, n.categories, n.mapping_status
            FROM nodes n LEFT JOIN pruned_nodes p ON n.id = p.id WHERE p.id IS NULL"""
        )
        connection.execute(
            """CREATE VIEW retained_statements AS
            SELECT s.* FROM statements s
            LEFT JOIN pruned_nodes h ON s.subject = h.id LEFT JOIN pruned_nodes t ON s.object = t.id
            WHERE h.id IS NULL AND t.id IS NULL"""
        )
        connection.execute(
            "CREATE TABLE relations AS SELECT predicate AS id, row_number() OVER (ORDER BY predicate)-1 AS idx FROM (SELECT DISTINCT predicate FROM retained_statements)"
        )
        connection.execute(
            "CREATE TABLE types AS SELECT categories, row_number() OVER (ORDER BY categories)-1 AS idx FROM (SELECT DISTINCT list_sort(categories) AS categories FROM entities)"
        )
        connection.execute(
            """COPY (
                SELECT e.id, e.idx, e.categories, e.mapping_status,
                       CAST(coalesce(t.idx, 0) AS BIGINT) AS type_set_id
                FROM entities e LEFT JOIN types t ON list_sort(e.categories) = t.categories
                ORDER BY e.idx)
               TO '{entities}' (FORMAT PARQUET, COMPRESSION ZSTD)""".format(entities=root / "entities.parquet")
        )
        for name in ("relations", "types"):
            connection.execute(f"COPY {name} TO '{root / (name + '.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        train_bound, valid_bound = split_bounds(runtime.split)
        seed = str(runtime.split.seed)
        # One streaming COPY replaces the per-row Python loop and its write-mode
        # array files. group_id and split are computed in SQL; the split CASE is
        # pinned to the Python reference by tests/test_prepare_topology.py.
        connection.execute(
            """COPY (
                SELECT row_number() OVER (ORDER BY g.id) - 1 AS index, g.id AS statement_id, g.group_id,
                    CAST(CASE WHEN g.role = 'schema' THEN 3
                        WHEN ('0x' || substr(sha256(? || chr(0) || g.group_id), 1, 16))::UBIGINT
                             < ('0x' || ?)::UBIGINT THEN 0
                        WHEN ('0x' || substr(sha256(? || chr(0) || g.group_id), 1, 16))::UBIGINT
                             < ('0x' || ?)::UBIGINT THEN 1
                        ELSE 2 END AS TINYINT) AS split,
                    CAST(g.head AS BIGINT) AS head, CAST(g.relation AS BIGINT) AS relation_id,
                    CAST(g.tail AS BIGINT) AS tail
                FROM (
                    SELECT s.id, s.relation_role AS role, h.idx AS head, r.idx AS relation, t.idx AS tail,
                        sha256(s.subject || chr(0) || s.predicate || chr(0) || s.object) AS group_id
                    FROM retained_statements s JOIN entities h ON s.subject = h.id JOIN entities t ON s.object = t.id
                    JOIN relations r ON s.predicate = r.id) g
                ORDER BY g.id)
               TO '{topology}' (FORMAT PARQUET, COMPRESSION ZSTD)""".format(topology=root / "topology.parquet"),
            [seed, train_bound, seed, valid_bound],
        )
        topology_rows = connection.execute(
            f"SELECT count(*) FROM read_parquet('{root / 'topology.parquet'}')"
        ).fetchone()
        assert topology_rows is not None
        if topology_rows[0] != total:
            raise ValueError("statement topology row count mismatch")
        # Shard sampling needs random access to the topology. It runs from RAM at a
        # documented budget of roughly 26 bytes per statement; see
        # .ralph/specs/performance-and-memory.md.
        topology = pq.read_table(root / "topology.parquet", columns=["head", "tail", "relation_id", "split"])
        arrays = {
            "head": topology.column("head").to_numpy(zero_copy_only=False),
            "tail": topology.column("tail").to_numpy(zero_copy_only=False),
            "relation_id": topology.column("relation_id").to_numpy(zero_copy_only=False),
            "split": topology.column("split").to_numpy(zero_copy_only=False),
            "type_set_id": pq.read_table(root / "entities.parquet", columns=["type_set_id"])
            .column("type_set_id")
            .to_numpy(zero_copy_only=False),
        }
        tensors = {name: save_tensor(root, name, values) for name, values in arrays.items()}
        relation_count = connection.execute("SELECT count(*) FROM relations").fetchone()
        assert relation_count is not None
        fields, feature_artifact = prepare_features(connection, source, root, entity_count, total)
        # The in-memory positive filter is only acceptable for small graphs; production shards use DiskGraph.
        in_memory = total <= 100_000 and entity_count <= 200_000
        examples = write_batch_shards(root, arrays, runtime, in_memory=in_memory)
        checksums = {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and "spill" not in path.parts
        }
        vocabularies = {}
        for name in ("entities", "relations", "types"):
            rows = connection.execute(f"SELECT count(*) FROM {name}").fetchone()
            assert rows is not None
            vocabularies[name] = VocabularyArtifact(
                path=f"{name}.parquet", rows=rows[0], sha256=checksums[f"{name}.parquet"]
            )
        result = DatasetManifest(
            dataset_id=dataset_id,
            snapshot_id=snapshot,
            seed=runtime.split.seed,
            pruning=pruning,
            entities=entity_count,
            relations=relation_count[0],
            statements=total,
            tensors=tensors,
            batches=examples,
            feature_fields=fields,
            features=feature_artifact,
            checksums=checksums,
            vocabularies=vocabularies,
        )
        atomic_write_json(root / "manifest.json", result.model_dump(mode="json"))
    finally:
        connection.close()
    root.rename(target)
    return target
