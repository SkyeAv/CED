from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np

from ced.acquisition.base import atomic_write_json, sha256_file
from ced.config.models import RuntimeConfig
from ced.kg.audit import validate_checksums
from ced.model.feature_store import prepare_features
from ced.model.schemas import DatasetManifest, TensorFormat, TensorRef, VocabularyArtifact
from ced.model.shards import write_batch_shards
from ced.model.split import group_split, statement_group
from ced.model.tensors import write_tensor
from ced.preflight import require_cpu_worker, require_disk


def save_tensor(root: Path, name: str, values: np.ndarray, *, format: TensorFormat = "npy") -> TensorRef:
    """Serialize one in-memory tensor and return its manifest reference."""
    array = np.ascontiguousarray(values)
    if format == "parquet":
        path = root / f"{name}.parquet"
        write_tensor(path, array)
    elif format == "npy":
        path = root / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
    else:
        raise ValueError(f"unsupported tensor format: {format}")
    return TensorRef(
        path=path.name, format=format, dtype=str(array.dtype), shape=list(array.shape), sha256=sha256_file(path)
    )


def prepare_dataset(runtime: RuntimeConfig, snapshot: str) -> Path:
    """Create disk-backed numeric topology without collecting the graph in Python."""
    require_cpu_worker(runtime)
    source = runtime.artifact_path("snapshots", snapshot)
    manifest = json.loads((source / "manifest.json").read_text())
    validate_checksums(source, json.loads((source / "checksums.json").read_text()))
    dataset_id = hashlib.sha256(f"{snapshot}\0{runtime.split.model_dump_json()}\0topology-v1".encode()).hexdigest()
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
        connection.execute(
            "CREATE TABLE entities AS SELECT id, row_number() OVER (ORDER BY id)-1 AS idx, categories, mapping_status FROM nodes"
        )
        connection.execute(
            "CREATE TABLE relations AS SELECT predicate AS id, row_number() OVER (ORDER BY predicate)-1 AS idx FROM (SELECT DISTINCT predicate FROM statements)"
        )
        connection.execute(
            "CREATE TABLE types AS SELECT categories, row_number() OVER (ORDER BY categories)-1 AS idx FROM (SELECT DISTINCT list_sort(categories) AS categories FROM nodes)"
        )
        for name in ("entities", "relations", "types"):
            connection.execute(f"COPY {name} TO '{root / (name + '.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        total = counts["statements"]
        arrays = {
            "edge_index": np.lib.format.open_memmap(
                root / "edge_index.npy", mode="w+", dtype="int64", shape=(2, total)
            ),
            "relation_id": np.lib.format.open_memmap(
                root / "relation_id.npy", mode="w+", dtype="int64", shape=(total,)
            ),
            "split": np.lib.format.open_memmap(root / "split.npy", mode="w+", dtype="int8", shape=(total,)),
            "type_set_id": np.lib.format.open_memmap(
                root / "type_set_id.npy", mode="w+", dtype="int64", shape=(counts["nodes"],)
            ),
        }
        position = 0
        with (root / "split-records.ndjson").open("w", encoding="utf-8") as stream:
            query = """SELECT s.id, s.subject, s.predicate, s.object, s.relation_role, h.idx AS head, r.idx AS relation, t.idx AS tail
                FROM statements s JOIN entities h ON s.subject=h.id JOIN entities t ON s.object=t.id
                JOIN relations r ON s.predicate=r.id ORDER BY s.id"""
            for batch in connection.execute(query).fetch_record_batch(5000):
                for row in batch.to_pylist():
                    group = statement_group(row["subject"], row["predicate"], row["object"])
                    split = group_split(group, runtime.split, schema=row["relation_role"] == "schema")
                    arrays["edge_index"][:, position] = [row["head"], row["tail"]]
                    arrays["relation_id"][position] = row["relation"]
                    arrays["split"][position] = split
                    stream.write(
                        json.dumps({"index": position, "statement_id": row["id"], "group_id": group, "split": split})
                        + "\n"
                    )
                    position += 1
        if position != total:
            raise ValueError("statement topology row count mismatch")
        for batch in connection.execute(
            "SELECT e.idx, t.idx AS type_id FROM entities e JOIN types t ON list_sort(e.categories)=t.categories ORDER BY e.idx"
        ).fetch_record_batch(5000):
            for row in batch.to_pylist():
                arrays["type_set_id"][row["idx"]] = row["type_id"]
        tensors = {}
        for name, values in arrays.items():
            values.flush()
            tensors[name] = TensorRef(
                path=f"{name}.npy",
                dtype=str(values.dtype),
                shape=list(values.shape),
                sha256=sha256_file(root / f"{name}.npy"),
            )
        relation_count = connection.execute("SELECT count(*) FROM relations").fetchone()
        assert relation_count is not None
        fields, feature_artifact = prepare_features(connection, source, root, counts["nodes"], total)
        # The in-memory positive filter is only acceptable for small graphs; production shards use DiskGraph.
        in_memory = counts["statements"] <= 100_000 and counts["nodes"] <= 200_000
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
            entities=counts["nodes"],
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
