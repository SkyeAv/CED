from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import Field

from ced.config.models import ConfigBase

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
CORE_NODE_FIELDS = {"id", "name", "category", "in_taxon", "provided_by"}
CORE_EDGE_FIELDS = {"id", "subject", "predicate", "object", "category", "publications", "sources", "ced_predicate"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def stable_statement_id(source_id: str, native_id: str | None, record: dict[str, Any]) -> str:
    identity = f"native\0{native_id}" if native_id else f"record\0{canonical_json(record)}"
    return hashlib.sha256(f"{source_id}\0{identity}".encode()).hexdigest()


class NodeRecord(ConfigBase):
    id: str
    names: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    in_taxon: list[str] = Field(default_factory=list)
    provided_by: list[str] = Field(default_factory=list)
    mapping_status: str = "source_native"
    original_label: str | None = None
    native_ids: dict[str, str] = Field(default_factory=dict)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class StatementRecord(ConfigBase):
    id: str
    source_id: str
    native_id: str | None = None
    subject: str
    predicate: str
    object: str
    relation_role: Literal["relation", "schema"] = "relation"
    target_eligible: bool = True
    categories: list[str] = Field(default_factory=list)
    qualifiers: dict[str, JsonValue] = Field(default_factory=dict)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    publications: list[str] = Field(default_factory=list)
    sources: list[JsonValue] = Field(default_factory=list)
    original_record: dict[str, JsonValue] = Field(default_factory=dict)


class PredicateRecord(ConfigBase):
    id: str
    source_id: str
    native_id: str | None = None
    name: str
    definition: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class OntologyTermRecord(ConfigBase):
    id: str
    source_id: str
    kind: Literal["predicate", "category", "taxon", "qualifier", "attribute"]
    native_id: str | None = None
    name: str
    definition: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class AttributeEvent(ConfigBase):
    owner_kind: Literal["node", "statement"]
    owner_id: str
    source_id: str
    path: str
    value_type: Literal["null", "bool", "int", "float", "string", "list", "object", "nonfinite"]
    ordinal: int = 0
    raw_json: str
    text: str | None = None
    numeric: float | None = None
    provenance: dict[str, JsonValue] = Field(default_factory=dict)


class ConflictRecord(ConfigBase):
    source_id: str
    level: str
    category: str
    taxon: str
    value: str
    name_key: str
    loser_key: str
    winner_id: str


class CanonicalBatch(ConfigBase):
    nodes: list[NodeRecord] = Field(default_factory=list)
    statements: list[StatementRecord] = Field(default_factory=list)
    attributes: list[AttributeEvent] = Field(default_factory=list)
    predicates: list[PredicateRecord] = Field(default_factory=list)
    terms: list[OntologyTermRecord] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)


class SourceLineage(ConfigBase):
    source_id: str
    revision: str
    license_name: str
    provided_by: str
    artifacts: dict[str, str]


class SnapshotManifest(ConfigBase):
    schema_version: int = 1
    snapshot_hash: str
    created_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    code_version: str = "0.1.0"
    sources: list[SourceLineage]
    counts: dict[str, int]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class BuildReport(ConfigBase):
    counts: dict[str, int]
    mapping_counts: dict[str, int]
    predicate_counts: dict[str, int]
    dropped_blank_rows: int = 0
    local_id_collisions: int = 0


def attribute_events(
    owner_kind: Literal["node", "statement"], owner_id: str, source_id: str, values: dict[str, Any], prefix: str = ""
) -> list[AttributeEvent]:
    events: list[AttributeEvent] = []

    def walk(value: Any, path: str, ordinal: int = 0) -> None:
        if value is None:
            kind = "null"
        elif isinstance(value, bool):
            kind = "bool"
        elif isinstance(value, int):
            kind = "int"
        elif isinstance(value, float):
            kind = "float" if math.isfinite(value) else "nonfinite"
        elif isinstance(value, str):
            kind = "string"
        elif isinstance(value, list):
            kind = "list"
        elif isinstance(value, dict):
            kind = "object"
        else:
            value = str(value)
            kind = "string"
        numeric = (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            else None
        )
        events.append(
            AttributeEvent(
                owner_kind=owner_kind,
                owner_id=owner_id,
                source_id=source_id,
                path=path,
                value_type=kind,
                ordinal=ordinal,
                raw_json=canonical_json(value),
                text=value if isinstance(value, str) else None,
                numeric=numeric,
            )
        )
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]", index)

    for key, value in values.items():
        walk(value, f"{prefix}.{key}" if prefix else key)
    return events
