from __future__ import annotations

from pathlib import Path

import duckdb

from ced.acquisition.base import atomic_write_json, sha256_file


def validate_checksums(root: Path, checksums: dict[str, str]) -> None:
    for relative, expected in checksums.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"checksum path escapes snapshot: {relative}")
        if sha256_file(path) != expected:
            raise ValueError(f"checksum mismatch: {relative}")


def write_reports(root: Path, *, threads: int, memory: str) -> None:
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={threads}")
        connection.execute("SET memory_limit=?", [memory])
        connection.execute("SET temp_directory=?", [str(root / "spill")])
        connection.execute("SET preserve_insertion_order=false")
        for name in ("nodes", "statements", "node_identities", "ontology_terms"):
            connection.read_parquet(str(root / name / "*.parquet")).create_view(name)
        mappings = connection.execute(
            "SELECT source_id, mapping_status, count(*) AS count FROM node_identities GROUP BY ALL ORDER BY ALL"
        ).fetchall()
        predicates = connection.execute(
            "SELECT source_id, predicate, relation_role, count(*) AS count FROM statements GROUP BY ALL ORDER BY ALL"
        ).fetchall()
        ontology = connection.execute(
            "SELECT source_id, kind, count(DISTINCT id) AS count FROM ontology_terms GROUP BY ALL ORDER BY ALL"
        ).fetchall()
        atomic_write_json(
            root / "mapping-report.json",
            [{"source_id": source, "mapping_status": status, "count": count} for source, status, count in mappings],
        )
        atomic_write_json(
            root / "predicate-report.json",
            [
                {"source_id": source, "predicate": predicate, "relation_role": role, "count": count}
                for source, predicate, role, count in predicates
            ],
        )
        atomic_write_json(
            root / "ontology-report.json",
            [{"source_id": source, "kind": kind, "count": count} for source, kind, count in ontology],
        )
    finally:
        connection.close()
