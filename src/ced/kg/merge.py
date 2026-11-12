from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from ced.kg.records import canonical_json
from ced.kg.writer import NODE_SCHEMA, NdjsonShardWriter, ParquetShardWriter


def raw_files(root: Path, name: str) -> list[str]:
    """Every raw partition for one table, in a deterministic order.

    Canonicalization writes one partition per source (`raw/<source>/<table>/`), while callers
    that merge a hand-built `raw/<table>/` tree still work; both layouts are accepted and the
    sorted union keeps merge output independent of how many workers produced it.
    """
    files = sorted((root / "raw" / name).glob("*.parquet"))
    files += sorted((root / "raw").glob(f"*/{name}/*.parquet"))
    unique = list(dict.fromkeys(files))
    if not unique:
        raise FileNotFoundError(f"no raw parquet files for {name} under {root / 'raw'}")
    return [str(path) for path in unique]


def merge_nodes(root: Path, connection: duckdb.DuckDBPyConnection, *, group_rows: int = 5000) -> int:
    """External sort, then reduce one canonical identity at a time."""
    connection.read_parquet(raw_files(root, "nodes")).create_view("raw_nodes")
    reader = connection.execute(
        "SELECT * FROM raw_nodes ORDER BY id, native_ids_json, attributes_json"
    ).fetch_record_batch(group_rows)
    writer = ParquetShardWriter(root / "nodes", NODE_SCHEMA)
    kgx = NdjsonShardWriter(root / "kgx/nodes")
    current: dict[str, Any] | None = None
    current_native_ids: dict[str, str] = {}

    def emit() -> None:
        if current is None:
            return
        # Serialize once per identity, not once per duplicate row.
        current["native_ids_json"] = canonical_json(current_native_ids)
        writer.write([current])
        kgx.write(
            {
                "id": current["id"],
                "name": current["names"],
                "category": current["categories"],
                "in_taxon": current["in_taxon"],
                "provided_by": current["provided_by"],
                "ced_mapping_status": current["mapping_status"],
            }
        )

    for batch in reader:
        for row in batch.to_pylist():
            if current is None or current["id"] != row["id"]:
                emit()
                current = row
                current_native_ids = json.loads(row["native_ids_json"])
                continue
            for field in ("names", "categories", "in_taxon", "provided_by", "source_ids"):
                current[field] = sorted(set(current[field]) | set(row[field]))
            if row["mapping_status"] in {"local", "unresolved_local"}:
                current["mapping_status"] = row["mapping_status"]
            current["original_label"] = current["original_label"] or row["original_label"]
            current_native_ids |= json.loads(row["native_ids_json"])
            # Every original attribute remains authoritative in the long event table.
            current["attributes_json"] = "{}"
    emit()
    writer.close()
    return kgx.close()


def merge_graph(root: Path, *, threads: int = 4, memory: str = "4GB") -> dict[str, int]:
    if not 1 <= threads <= 4:
        raise ValueError("merge requires one to four workers")
    temporary = root / "spill"
    temporary.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={threads}")
        connection.execute("SET memory_limit=?", [memory])
        connection.execute("SET temp_directory=?", [str(temporary)])
        connection.execute("SET preserve_insertion_order=false")
        counts = {"nodes": merge_nodes(root, connection)}
        for name in ("statements", "attributes", "node_identities", "predicates", "ontology_terms", "conflicts"):
            connection.read_parquet(raw_files(root, name)).create_view(f"raw_{name}")
        collision = connection.execute("""
            SELECT id FROM raw_node_identities WHERE mapping_status IN ('local', 'unresolved_local')
            GROUP BY id HAVING count(DISTINCT lower(trim(original_label))) > 1 LIMIT 1
        """).fetchone()
        if collision:
            raise ValueError(f"local-ID collision: {collision[0]}")
        # Identical native IDs with different payloads are errors, not silently selected rows.
        collision = connection.execute("""
            SELECT id FROM (SELECT DISTINCT * FROM raw_statements) GROUP BY id HAVING count(*)>1 LIMIT 1
        """).fetchone()
        if collision:
            raise ValueError(f"conflicting statement identity: {collision[0]}")
        connection.read_parquet(str(root / "nodes/*.parquet")).create_view("nodes")
        missing = connection.execute("""
            SELECT endpoint FROM (
              SELECT subject AS endpoint FROM raw_statements UNION SELECT object FROM raw_statements
            ) e ANTI JOIN nodes n ON e.endpoint=n.id LIMIT 1
        """).fetchone()
        if missing:
            raise ValueError(f"statement references missing node: {missing[0]}")
        for name in ("statements", "attributes", "node_identities", "predicates", "ontology_terms", "conflicts"):
            destination = root / name
            destination.mkdir()
            distinct = "" if name == "attributes" else "DISTINCT "
            connection.execute(
                f"COPY (SELECT {distinct}* FROM raw_{name} ORDER BY ALL) TO '{destination}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, FILE_SIZE_BYTES '128MB')"
            )
            count = connection.execute(f"SELECT count(*) FROM read_parquet('{destination}/*.parquet')").fetchone()
            assert count is not None
            counts[name] = count[0]
        writer = NdjsonShardWriter(root / "kgx/edges")
        for batch in connection.execute("SELECT DISTINCT * FROM raw_statements ORDER BY id").fetch_record_batch(5000):
            for row in batch.to_pylist():
                writer.write(
                    {
                        "id": row["id"],
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"],
                        "category": row["categories"],
                        "publications": row["publications"],
                        "qualifiers": json.loads(row["qualifiers_json"]),
                        "sources": json.loads(row["sources_json"]),
                        "ced_attributes": json.loads(row["attributes_json"]),
                        "ced_source": row["source_id"],
                        "ced_native_id": row["native_id"],
                        "ced_relation_role": row["relation_role"],
                    }
                )
        writer.close()
        return counts
    finally:
        connection.close()
