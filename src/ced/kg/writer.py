from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from ced.kg.records import CanonicalBatch, canonical_json

JSON = pa.string()
NODE_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("names", pa.list_(pa.string())),
        ("categories", pa.list_(pa.string())),
        ("in_taxon", pa.list_(pa.string())),
        ("provided_by", pa.list_(pa.string())),
        ("source_ids", pa.list_(pa.string())),
        ("mapping_status", pa.string()),
        ("original_label", pa.string()),
        ("native_ids_json", JSON),
        ("attributes_json", JSON),
    ]
)
STATEMENT_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source_id", pa.string()),
        ("native_id", pa.string()),
        ("subject", pa.string()),
        ("predicate", pa.string()),
        ("object", pa.string()),
        ("relation_role", pa.string()),
        ("target_eligible", pa.bool_()),
        ("categories", pa.list_(pa.string())),
        ("qualifiers_json", JSON),
        ("attributes_json", JSON),
        ("publications", pa.list_(pa.string())),
        ("sources_json", JSON),
        ("original_record_json", JSON),
    ]
)
ATTRIBUTE_SCHEMA = pa.schema(
    [
        ("owner_kind", pa.string()),
        ("owner_id", pa.string()),
        ("source_id", pa.string()),
        ("path", pa.string()),
        ("value_type", pa.string()),
        ("ordinal", pa.int32()),
        ("raw_json", JSON),
        ("text", pa.string()),
        ("numeric", pa.float64()),
        ("provenance_json", JSON),
    ]
)
IDENTITY_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source_id", pa.string()),
        ("native_id", pa.string()),
        ("mapping_status", pa.string()),
        ("original_label", pa.string()),
    ]
)
PREDICATE_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source_id", pa.string()),
        ("native_id", pa.string()),
        ("name", pa.string()),
        ("definition", pa.string()),
        ("attributes_json", JSON),
    ]
)
TERM_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source_id", pa.string()),
        ("kind", pa.string()),
        ("native_id", pa.string()),
        ("name", pa.string()),
        ("definition", pa.string()),
        ("attributes_json", JSON),
    ]
)
CONFLICT_SCHEMA = pa.schema(
    [
        ("source_id", pa.string()),
        ("level", pa.string()),
        ("category", pa.string()),
        ("taxon", pa.string()),
        ("value", pa.string()),
        ("name_key", pa.string()),
        ("loser_key", pa.string()),
        ("winner_id", pa.string()),
    ]
)


class ParquetShardWriter:
    def __init__(self, root: Path, schema: pa.Schema, shard_rows: int = 1_000_000, group_rows: int = 5000) -> None:
        self.root = root
        self.schema = schema
        self.shard_rows = shard_rows
        self.group_rows = group_rows
        self.buffer: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None
        self.shard = 0
        self.rows_in_shard = 0
        self.rows = 0
        root.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.root / f"part-{self.shard:05d}.parquet", self.schema, compression="zstd"
            )

    def _flush(self) -> None:
        if not self.buffer:
            return
        self._open()
        assert self.writer is not None
        self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
        count = len(self.buffer)
        self.rows_in_shard += count
        self.rows += count
        self.buffer = []
        if self.rows_in_shard >= self.shard_rows:
            self._close_shard()

    def _close_shard(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.shard += 1
            self.rows_in_shard = 0

    def write(self, rows: list[dict[str, Any]]) -> None:
        self.buffer.extend(rows)
        if len(self.buffer) >= self.group_rows:
            self._flush()

    def close(self) -> int:
        self._flush()
        self._close_shard()
        if self.shard == 0:
            self._open()
            self._close_shard()
        return self.rows


def node_row(node: Any) -> dict[str, Any]:
    return {
        "id": node.id,
        "names": node.names,
        "categories": node.categories,
        "in_taxon": node.in_taxon,
        "provided_by": node.provided_by,
        "source_ids": sorted(node.native_ids),
        "mapping_status": node.mapping_status,
        "original_label": node.original_label,
        "native_ids_json": canonical_json(node.native_ids),
        "attributes_json": canonical_json(node.attributes),
    }


def statement_row(statement: Any) -> dict[str, Any]:
    return {
        "id": statement.id,
        "source_id": statement.source_id,
        "native_id": statement.native_id,
        "subject": statement.subject,
        "predicate": statement.predicate,
        "object": statement.object,
        "relation_role": statement.relation_role,
        "target_eligible": statement.target_eligible,
        "categories": statement.categories,
        "qualifiers_json": canonical_json(statement.qualifiers),
        "attributes_json": canonical_json(statement.attributes),
        "publications": statement.publications,
        "sources_json": canonical_json(statement.sources),
        "original_record_json": canonical_json(statement.original_record),
    }


class CanonicalBatchWriter:
    def __init__(self, root: Path, shard_rows: int = 1_000_000) -> None:
        self.writers = {
            "nodes": ParquetShardWriter(root / "nodes", NODE_SCHEMA, shard_rows),
            "statements": ParquetShardWriter(root / "statements", STATEMENT_SCHEMA, shard_rows),
            "attributes": ParquetShardWriter(root / "attributes", ATTRIBUTE_SCHEMA, shard_rows),
            "node_identities": ParquetShardWriter(root / "node_identities", IDENTITY_SCHEMA, shard_rows),
            "predicates": ParquetShardWriter(root / "predicates", PREDICATE_SCHEMA, shard_rows),
            "ontology_terms": ParquetShardWriter(root / "ontology_terms", TERM_SCHEMA, shard_rows),
            "conflicts": ParquetShardWriter(root / "conflicts", CONFLICT_SCHEMA, shard_rows),
        }

    def write(self, batch: CanonicalBatch) -> None:
        self.writers["nodes"].write([node_row(node) for node in batch.nodes])
        self.writers["statements"].write([statement_row(statement) for statement in batch.statements])
        self.writers["attributes"].write(
            [
                {
                    **event.model_dump(exclude={"provenance"}),
                    "provenance_json": canonical_json(event.provenance),
                }
                for event in batch.attributes
            ]
        )
        self.writers["node_identities"].write(
            [
                {
                    "id": node.id,
                    "source_id": source,
                    "native_id": native,
                    "mapping_status": node.mapping_status,
                    "original_label": node.original_label,
                }
                for node in batch.nodes
                for source, native in node.native_ids.items()
            ]
        )
        self.writers["predicates"].write(
            [
                {**record.model_dump(exclude={"attributes"}), "attributes_json": canonical_json(record.attributes)}
                for record in batch.predicates
            ]
        )
        self.writers["ontology_terms"].write(
            [
                {**record.model_dump(exclude={"attributes"}), "attributes_json": canonical_json(record.attributes)}
                for record in batch.terms
            ]
        )
        self.writers["conflicts"].write([record.model_dump() for record in batch.conflicts])

    def close(self) -> dict[str, int]:
        return {name: writer.close() for name, writer in self.writers.items()}


class NdjsonShardWriter:
    def __init__(self, root: Path, shard_rows: int = 1_000_000) -> None:
        self.root = root
        self.shard_rows = shard_rows
        self.root.mkdir(parents=True, exist_ok=True)
        self.stream = None
        self.shard = 0
        self.rows = 0

    def write(self, record: dict[str, Any]) -> None:
        if self.rows % self.shard_rows == 0:
            if self.stream is not None:
                self.stream.close()
            self.stream = (self.root / f"part-{self.shard:05d}.ndjson").open("wb")
            self.shard += 1
        assert self.stream is not None
        self.stream.write(orjson.dumps(record, option=orjson.OPT_NON_STR_KEYS) + b"\n")
        self.rows += 1

    def close(self) -> int:
        if self.stream is not None:
            self.stream.close()
        return self.rows
