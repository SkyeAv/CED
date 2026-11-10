from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ced.acquisition.base import atomic_write_json, sha256_file
from ced.model.prepare import save_tensor
from ced.model.schemas import DatasetManifest, GraphBatch, NegativeBatch, TrainingExample


def write_table(path: Path, column: str, values: list[Any]) -> None:
    pq.write_table(pa.table({column: values}), path)


def synthetic_dataset(root: Path, *, batches: int = 3, nodes_per_batch: int = 64) -> int:
    root.mkdir(parents=True, exist_ok=True)
    statements = 3
    tensors = {
        name: save_tensor(root, name, values)
        for name, values in {
            "edge_index": np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int64),
            "relation_id": np.zeros(statements, dtype=np.int64),
            "split": np.array([0, 1, 2], dtype=np.int8),
            "type_set_id": np.zeros(3, dtype=np.int64),
        }.items()
    }
    write_table(root / "entities.parquet", "id", ["CED:Family_1", "CED:Family_2", "X:3"])
    write_table(root / "relations.parquet", "id", ["p"])
    write_table(root / "types.parquet", "categories", [["biolink:Gene"]])
    pq.write_table(
        pa.table({"idx": pa.array([], pa.int64()), "kind": pa.array([], pa.string())}),
        root / "fitted_fields.parquet",
    )
    pq.write_table(
        pa.table({"field_id": pa.array([], pa.int64()), "value_id": pa.array([], pa.int64())}),
        root / "fitted_categories.parquet",
    )
    examples = []
    for index in range(batches):
        directory = root / "batches" / f"group-{index:05d}"
        directory.mkdir(parents=True, exist_ok=True)

        def save(name: str, values: np.ndarray, directory: Path = directory, index: int = index):
            reference = save_tensor(directory, name, values)
            reference.path = f"batches/group-{index:05d}/{reference.path}"
            return reference

        graph = GraphBatch(
            global_node_id=save("global_node_id", np.zeros(nodes_per_batch, dtype=np.int64)),
            seed_id=save("seed_id", np.array([0], dtype=np.int64)),
            type_set_id=save("type_set_id", np.zeros(2, dtype=np.int64)),
            edge_index=save("edge_index", np.array([[0, 1], [1, 0]], dtype=np.int64)),
            relation_id=save("relation_id", np.array([0, 1], dtype=np.int64)),
            edge_type_id=save("edge_type_id", np.array([0, 1], dtype=np.int64)),
            statement_id=save("statement_id", np.array([0, 0], dtype=np.int64)),
            positive_statement_id=save("positive_statement_id", np.array([0], dtype=np.int64)),
            negative=NegativeBatch(
                subject=save("negative_subject", np.zeros((1, 2), dtype=np.int64)),
                object=save("negative_object", np.ones((1, 2), dtype=np.int64)),
                valid=save("negative_valid", np.array([[True, False]])),
            ),
        )
        examples.append(TrainingExample(graph=graph, split="train"))
    manifest = DatasetManifest(
        dataset_id="synthetic",
        snapshot_id="snapshot",
        seed=7,
        entities=3,
        relations=1,
        statements=statements,
        tensors=tensors,
        batches=examples,
    )
    atomic_write_json(root / "manifest.json", manifest.model_dump(mode="json"))
    checksums = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest.checksums = checksums
    atomic_write_json(root / "manifest.json", manifest.model_dump(mode="json"))
    return sum(path.stat().st_size for path in (root / "batches" / "group-00000").iterdir())
