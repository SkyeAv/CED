from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from ced.acquisition.pipeline import SourceArtifacts
from ced.kg.base import batches
from ced.kg.records import (
    CanonicalBatch,
    ConflictRecord,
    JsonValue,
    NodeRecord,
    OntologyTermRecord,
    PredicateRecord,
    StatementRecord,
    attribute_events,
    stable_statement_id,
)

CURIE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*:\S+$")


def canonical_id(kind: str, key: str, raw_uuid: bytes) -> str:
    if kind in {"entity", "taxon"}:
        if not CURIE.fullmatch(key):
            raise ValueError(f"malformed Amalgam canonical KEY for {kind}: {key!r}")
        return key
    return f"AMALGAM:{uuid.UUID(bytes=bytes(raw_uuid))}"


def amalgam_term(value: str) -> str:
    return value if value.startswith(("biolink:", "AMALGAM:")) else f"AMALGAM:{value}"


class AmalgamCompactAdapter:
    def __init__(self, artifacts: SourceArtifacts, temporary_directory: Path | None = None) -> None:
        self.artifacts = artifacts
        self.source = artifacts.source
        inventories = artifacts.paths("inventory")
        if self.source.inventory is not None and not inventories:
            raise ValueError("Amalgam inventory manifest is required")
        if inventories:
            self.validate_inventory(inventories[0])
        self.size = int(self.source.mapping.options.get("batch_size", 5000))
        self.threads = int(self.source.mapping.options.get("threads", 4))
        if not 1 <= self.threads <= 4:
            raise ValueError("Amalgam adapter requires one to four workers")
        self.memory = str(self.source.mapping.options.get("memory_limit", "4GB"))
        self.temporary = temporary_directory or Path(".ced/tmp/amalgam-duckdb")
        self.connection = duckdb.connect()
        self.connection.execute(f"SET threads={self.threads}")
        self.connection.execute("SET memory_limit=?", [self.memory])
        self.temporary.mkdir(parents=True, exist_ok=True)
        self.connection.execute("SET temp_directory=?", [str(self.temporary)])
        for role in ("vertices", "relations", "attributes", "categories", "predicates", "taxa", "conflicts"):
            paths = [str(path) for path in self.artifacts.paths(role)]
            if not paths:
                raise ValueError(f"Amalgam role has no artifacts: {role}")
            for path in paths:
                schema = pq.read_schema(path)
                for field in schema:
                    if field.name in {
                        "ID",
                        "SUBJECT_ID",
                        "OBJECT_ID",
                        "PREDICATE_ID",
                        "CATEGORY_ID",
                        "TAXON_ID",
                        "WINNER_ID",
                        "VERTEX_ID",
                    }:
                        expected = pa.binary(16) if role == "vertices" and field.name == "VERTEX_ID" else pa.int32()
                        if field.type != expected:
                            raise ValueError(
                                f"incompatible Amalgam schema: {role}.{field.name}: {field.type}; expected {expected}"
                            )
            self.connection.read_parquet(paths).create_view(role)
        self.validate_references()
        self.connection.execute(
            "CREATE VIEW typed_relations AS SELECT r.*, p.NAME AS PREDICATE_NAME, p.KIND AS PREDICATE_KIND "
            "FROM relations r JOIN predicates p ON r.PREDICATE_ID=p.ID"
        )

    def validate_inventory(self, path: Path) -> None:
        document = json.loads(path.read_text(encoding="utf-8"))
        if self.source.inventory is None:
            raise ValueError("Amalgam source must declare an inventory")
        expected = self.source.inventory.materialize(document)
        fetched = {item.origin: item for group in self.artifacts.files.values() for item in group}
        total = 0
        for spec in expected:
            candidates = [item for origin, item in fetched.items() if origin.endswith(spec.path)]
            if len(candidates) != 1 or candidates[0].sha256 != spec.sha256:
                raise ValueError(f"Amalgam verified artifact missing or hash mismatch: {spec.path}")
            artifact = candidates[0]
            count = pq.ParquetFile(artifact.path).metadata.num_rows
            if count != document["compact"]["row_counts"][spec.path]:
                raise ValueError(f"Amalgam row count mismatch: {spec.path}")
            total += artifact.size
        if total != document["compact"]["total_bytes"]:
            raise ValueError("Amalgam total byte count mismatch")
        runs = self.artifacts.files.get("run_manifest", [])
        if len(runs) != 1 or runs[0].sha256 != document["source_run_manifest_sha256"]:
            raise ValueError("Amalgam source run manifest hash mismatch")

    def validate_references(self) -> None:
        for table in ("vertices", "categories", "predicates", "taxa"):
            invalid = self.connection.execute(
                f"SELECT ID FROM {table} GROUP BY ID HAVING count(*)>1 OR ID IS NULL LIMIT 1"
            ).fetchone()
            if invalid:
                raise ValueError(f"duplicate or null Amalgam dictionary ID in {table}")
        references = {
            "relations": {
                "SUBJECT_ID": "vertices",
                "OBJECT_ID": "vertices",
                "PREDICATE_ID": "predicates",
                "CATEGORY_ID": "categories",
                "TAXON_ID": "taxa",
            },
            "attributes": {"SUBJECT_ID": "vertices", "PREDICATE_ID": "predicates"},
            "categories": {"VERTEX_ID": "vertices"},
            "predicates": {"VERTEX_ID": "vertices"},
            "taxa": {"VERTEX_ID": "vertices"},
            "conflicts": {"WINNER_ID": "vertices"},
        }
        for table, columns in references.items():
            for column, target in columns.items():
                invalid = self.connection.execute(
                    f"SELECT s.{column} FROM {table} s ANTI JOIN {target} t ON s.{column}=t.ID WHERE s.{column} IS NOT NULL LIMIT 1"
                ).fetchone()
                if invalid:
                    raise ValueError(f"dangling Amalgam reference: {table}.{column}={invalid[0]}")

    def close(self) -> None:
        self.connection.close()

    def rows(self, sql: str) -> Iterator[dict[str, Any]]:
        reader = self.connection.execute(sql).fetch_record_batch(self.size)
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                return
            yield from batch.to_pylist()

    def pack(self, sql: str, convert: Callable[[dict[str, Any]], CanonicalBatch]) -> Iterator[CanonicalBatch]:
        for chunk in batches(self.rows(sql), self.size, lambda rows: rows):
            combined = CanonicalBatch()
            for row in chunk:
                item = convert(row)
                combined.nodes.extend(item.nodes)
                combined.statements.extend(item.statements)
                combined.attributes.extend(item.attributes)
                combined.predicates.extend(item.predicates)
                combined.terms.extend(item.terms)
                combined.conflicts.extend(item.conflicts)
            yield combined

    def node(self, row: dict[str, Any]) -> CanonicalBatch:
        identifier = canonical_id(row["KIND"], row["KEY"], row["VERTEX_ID"])
        category = amalgam_term(row["CATEGORY"] or row["KIND"])
        taxon = row.get("TAXON_KEY")
        attributes: dict[str, JsonValue] = {
            "amalgam": {
                "uuid": str(uuid.UUID(bytes=bytes(row["VERTEX_ID"]))),
                "dictionary_id": row["ID"],
                "kind": row["KIND"],
                "key": row["KEY"],
                "category": row["CATEGORY"],
                "taxon": taxon,
            }
        }
        node = NodeRecord(
            id=identifier,
            names=[row["NAME"]] if row["NAME"] else [],
            categories=[category],
            in_taxon=[taxon] if taxon else [],
            provided_by=[self.source.provenance.provided_by],
            native_ids={self.source.name: row["KEY"]},
            attributes=attributes,
        )
        terms = [
            OntologyTermRecord(
                id=category, source_id=self.source.name, kind="category", name=row["CATEGORY"] or row["KIND"]
            )
        ]
        if taxon:
            terms.append(OntologyTermRecord(id=taxon, source_id=self.source.name, kind="taxon", name=taxon))
        return CanonicalBatch(
            nodes=[node], attributes=attribute_events("node", identifier, self.source.name, attributes), terms=terms
        )

    def predicate(self, row: dict[str, Any]) -> CanonicalBatch:
        raw = row["NAME"]
        identifier = self.source.mapping.predicate_overrides.get(raw, amalgam_term(raw))
        native = str(row["ID"])
        attributes: dict[str, JsonValue] = {
            "amalgam": {
                "dictionary_id": row["ID"],
                "vertex_dictionary_id": row["VERTEX_ID"],
                "kind": row["KIND"],
                "raw_name": raw,
            }
        }
        return CanonicalBatch(
            predicates=[
                PredicateRecord(
                    id=identifier,
                    source_id=self.source.name,
                    native_id=native,
                    name=raw,
                    definition=row["DEFINITION"],
                    attributes=attributes,
                )
            ],
            terms=[
                OntologyTermRecord(
                    id=identifier,
                    source_id=self.source.name,
                    kind="attribute" if row["KIND"] == "attribute" else "predicate",
                    native_id=native,
                    name=raw,
                    definition=row["DEFINITION"],
                    attributes=attributes,
                )
            ],
        )

    def attribute(self, row: dict[str, Any]) -> CanonicalBatch:
        owner = canonical_id(row["KIND"], row["KEY"], row["VERTEX_ID"])
        values = {
            "amalgam_attributes": {
                row["PREDICATE_NAME"]: {
                    "value": row["VALUE"],
                    "L1": row["L1"],
                    "L2": row["L2"],
                    "predicate_dictionary_id": row["PREDICATE_ID"],
                }
            }
        }
        nodes = []
        if row["PREDICATE_NAME"] in {"has-name", "has-synonym"}:
            nodes.append(NodeRecord(id=owner, names=[row["VALUE"]], native_ids={self.source.name: row["KEY"]}))
        return CanonicalBatch(nodes=nodes, attributes=attribute_events("node", owner, self.source.name, values))

    def relation(self, row: dict[str, Any]) -> CanonicalBatch:
        subject = canonical_id(row["SUBJECT_KIND"], row["SUBJECT_KEY"], row["SUBJECT_UUID"])
        if row["OBJECT_ID"] is not None:
            object_id = canonical_id(row["OBJECT_KIND"], row["OBJECT_KEY"], row["OBJECT_UUID"])
            object_uuid = row["OBJECT_UUID"]
        elif row["CATEGORY_ID"] is not None:
            object_id = canonical_id(row["CATEGORY_KIND"], row["CATEGORY_KEY"], row["CATEGORY_UUID"])
            object_uuid = row["CATEGORY_UUID"]
        elif row["TAXON_ID"] is not None:
            object_id = canonical_id(row["TAXON_KIND"], row["TAXON_KEY"], row["TAXON_UUID"])
            object_uuid = row["TAXON_UUID"]
        else:
            raise ValueError("Amalgam relation has no object, category, or taxon reference")
        raw_predicate = row["PREDICATE_NAME"]
        predicate = self.source.mapping.predicate_overrides.get(raw_predicate, amalgam_term(raw_predicate))
        role = "schema" if raw_predicate in self.source.mapping.schema_relations else "relation"
        semantic = f"{subject}\0{predicate}\0{object_id}\0{role}"
        native = hashlib.sha256(semantic.encode()).hexdigest()
        attributes: dict[str, JsonValue] = {
            "amalgam": {
                "subject_dictionary_id": row["SUBJECT_ID"],
                "predicate_dictionary_id": row["PREDICATE_ID"],
                "object_dictionary_id": row["OBJECT_ID"],
                "category_dictionary_id": row["CATEGORY_ID"],
                "taxon_dictionary_id": row["TAXON_ID"],
                "subject_uuid": str(uuid.UUID(bytes=bytes(row["SUBJECT_UUID"]))),
                "object_uuid": str(uuid.UUID(bytes=bytes(object_uuid))) if object_uuid else None,
                "raw_predicate": raw_predicate,
                "category": row["CATEGORY_NAME"],
                "taxon": row["TAXON_NAME"],
            }
        }
        statement = StatementRecord(
            id=stable_statement_id(self.source.name, native, {}),
            source_id=self.source.name,
            native_id=native,
            subject=subject,
            predicate=predicate,
            object=object_id,
            relation_role=role,
            target_eligible=role != "schema",
            attributes=attributes,
            original_record={"table": "relations"},
        )
        term = OntologyTermRecord(id=predicate, source_id=self.source.name, kind="predicate", name=raw_predicate)
        return CanonicalBatch(
            statements=[statement],
            attributes=attribute_events("statement", statement.id, self.source.name, attributes),
            terms=[term],
        )

    def conflict(self, row: dict[str, Any]) -> CanonicalBatch:
        winner = canonical_id(row["KIND"], row["KEY"], row["VERTEX_ID"])
        return CanonicalBatch(
            conflicts=[
                ConflictRecord(
                    source_id=self.source.name,
                    level=row["LEVEL"],
                    category=row["CATEGORY"],
                    taxon=row["TAXON"],
                    value=row["VALUE"],
                    name_key=row["NAME_KEY"],
                    loser_key=row["LOSER_KEY"],
                    winner_id=winner,
                )
            ]
        )

    def batches(self, artifacts: SourceArtifacts | None = None) -> Iterator[CanonicalBatch]:
        node_sql = """
            SELECT v.*, c.NAME AS CATEGORY, tv.KEY AS TAXON_KEY FROM vertices v
            LEFT JOIN typed_relations hc ON hc.SUBJECT_ID=v.ID AND hc.PREDICATE_NAME='has-category'
            LEFT JOIN categories c ON hc.CATEGORY_ID=c.ID
            LEFT JOIN typed_relations it ON it.SUBJECT_ID=v.ID AND it.PREDICATE_NAME='in-taxon'
            LEFT JOIN taxa t ON it.TAXON_ID=t.ID LEFT JOIN vertices tv ON t.VERTEX_ID=tv.ID
            ORDER BY v.ID
        """
        yield from self.pack(node_sql, self.node)
        yield from self.pack("SELECT * FROM predicates ORDER BY ID", self.predicate)
        yield from self.pack(
            "SELECT c.*, v.KEY, v.KIND FROM categories c JOIN vertices v ON c.VERTEX_ID=v.ID ORDER BY c.ID",
            self.category,
        )
        yield from self.pack(
            "SELECT t.*, v.KEY, v.KIND, v.VERTEX_ID AS TAXON_UUID FROM taxa t JOIN vertices v ON t.VERTEX_ID=v.ID ORDER BY t.ID",
            self.taxon,
        )
        attribute_sql = """
            SELECT a.*, p.NAME AS PREDICATE_NAME, v.KEY, v.KIND, v.VERTEX_ID
            FROM attributes a JOIN predicates p ON a.PREDICATE_ID=p.ID JOIN vertices v ON a.SUBJECT_ID=v.ID
            ORDER BY a.SUBJECT_ID, a.PREDICATE_ID, a.VALUE, a.L1, a.L2
        """
        yield from self.pack(attribute_sql, self.attribute)
        relation_sql = """
            SELECT r.*, sv.KIND AS SUBJECT_KIND, sv.KEY AS SUBJECT_KEY, sv.VERTEX_ID AS SUBJECT_UUID,
                ov.KIND AS OBJECT_KIND, ov.KEY AS OBJECT_KEY, ov.VERTEX_ID AS OBJECT_UUID,
                cv.KIND AS CATEGORY_KIND, cv.KEY AS CATEGORY_KEY, cv.VERTEX_ID AS CATEGORY_UUID,
                tv.KIND AS TAXON_KIND, tv.KEY AS TAXON_KEY, tv.VERTEX_ID AS TAXON_UUID,
                c.NAME AS CATEGORY_NAME, t.NAME AS TAXON_NAME
            FROM typed_relations r JOIN vertices sv ON r.SUBJECT_ID=sv.ID
            LEFT JOIN vertices ov ON r.OBJECT_ID=ov.ID
            LEFT JOIN categories c ON r.CATEGORY_ID=c.ID LEFT JOIN vertices cv ON c.VERTEX_ID=cv.ID
            LEFT JOIN taxa t ON r.TAXON_ID=t.ID LEFT JOIN vertices tv ON t.VERTEX_ID=tv.ID
            ORDER BY r.SUBJECT_ID, r.PREDICATE_ID, r.OBJECT_ID, r.CATEGORY_ID, r.TAXON_ID
        """
        yield from self.pack(relation_sql, self.relation)
        conflict_sql = "SELECT c.*, v.KIND, v.KEY, v.VERTEX_ID FROM conflicts c JOIN vertices v ON c.WINNER_ID=v.ID ORDER BY c.WINNER_ID, c.LOSER_KEY"
        yield from self.pack(conflict_sql, self.conflict)

    def category(self, row: dict[str, Any]) -> CanonicalBatch:
        identifier = amalgam_term(row["NAME"])
        term = OntologyTermRecord(
            id=identifier,
            source_id=self.source.name,
            kind="category",
            native_id=str(row["ID"]),
            name=row["NAME"],
            definition=row["DEFINITION"],
            attributes={
                "amalgam": {"dictionary_id": row["ID"], "vertex_dictionary_id": row["VERTEX_ID"], "key": row["KEY"]}
            },
        )
        return CanonicalBatch(terms=[term])

    def taxon(self, row: dict[str, Any]) -> CanonicalBatch:
        identifier = canonical_id(row["KIND"], row["KEY"], row["TAXON_UUID"])
        term = OntologyTermRecord(
            id=identifier,
            source_id=self.source.name,
            kind="taxon",
            native_id=str(row["ID"]),
            name=row["NAME"],
            definition=row["RANK"],
            attributes={
                "amalgam": {"dictionary_id": row["ID"], "vertex_dictionary_id": row["VERTEX_ID"], "key": row["KEY"]}
            },
        )
        return CanonicalBatch(terms=[term])
