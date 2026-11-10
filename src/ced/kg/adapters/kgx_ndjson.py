from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import orjson

from ced.acquisition.pipeline import SourceArtifacts
from ced.kg.base import batches
from ced.kg.records import (
    CORE_EDGE_FIELDS,
    CORE_NODE_FIELDS,
    CanonicalBatch,
    NodeRecord,
    OntologyTermRecord,
    PredicateRecord,
    StatementRecord,
    attribute_events,
    stable_statement_id,
    string_list,
)


def read_ndjson(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = orjson.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"KGX record must be an object: {path}:{line_number}")
                yield line_number, value


class KgxNdjsonAdapter:
    def __init__(self, artifacts: SourceArtifacts) -> None:
        self.artifacts = artifacts
        self.source = artifacts.source
        self.size = int(self.source.mapping.options.get("batch_size", 5000))
        self.predicates: set[str] = set()
        self.terms: set[tuple[str, str]] = set()

    def chunks(self, role: str) -> Iterator[list[tuple[Path, int, dict[str, Any]]]]:
        def records() -> Iterator[tuple[Path, int, dict[str, Any]]]:
            for path in self.artifacts.paths(role):
                for line, raw in read_ndjson(path):
                    yield path, line, raw

        return batches(records(), self.size, lambda chunk: chunk)

    def predicate(self, raw: dict[str, Any]) -> str:
        original = str(raw.get("ced_original_predicate", raw["predicate"]))
        return str(raw.get("ced_predicate", self.source.mapping.predicate_overrides.get(original, raw["predicate"])))

    def overlay_nodes(self, raw: dict[str, Any]) -> list[NodeRecord]:
        nodes: list[NodeRecord] = []
        for side in ("subject", "object"):
            status = raw.get(f"{side}_mapping_status")
            original = raw.get(f"original_{side}")
            if status or original:
                nodes.append(
                    NodeRecord(
                        id=str(raw[side]),
                        names=[str(original)] if original is not None else [],
                        mapping_status=str(status or "fullmap"),
                        original_label=str(original) if original is not None else None,
                        native_ids={self.source.name: str(raw[side])},
                    )
                )
        for key, original in raw.items():
            if not key.startswith("original_") or key in {"original_subject", "original_object"}:
                continue
            identifier = raw.get(key.removeprefix("original_"))
            if not isinstance(identifier, str) or ":" not in identifier:
                continue
            nodes.append(
                NodeRecord(
                    id=identifier,
                    names=[str(original)] if original is not None else [],
                    categories=["biolink:NamedThing"],
                    mapping_status=str(raw.get(f"{key.removeprefix('original_')}_mapping_status", "fullmap")),
                    original_label=str(original) if original is not None else None,
                    native_ids={self.source.name: identifier},
                )
            )
        return nodes

    def ontology_terms(
        self, kind: Literal["predicate", "category", "taxon", "qualifier", "attribute"], values: list[str]
    ) -> list[OntologyTermRecord]:
        terms = []
        for value in values:
            if (kind, value) not in self.terms:
                self.terms.add((kind, value))
                terms.append(OntologyTermRecord(id=value, source_id=self.source.name, kind=kind, name=value))
        return terms

    def statement(self, raw: dict[str, Any], path: Path, line: int) -> StatementRecord:
        predicate = self.predicate(raw)
        native_id = str(raw["id"]) if raw.get("id") is not None else None
        attributes = {key: value for key, value in raw.items() if key not in CORE_EDGE_FIELDS}
        role = (
            "schema"
            if predicate in self.source.mapping.schema_relations
            or raw.get("ced_original_predicate") in self.source.mapping.schema_relations
            else "relation"
        )
        qualifiers_raw = raw.get("qualifiers", {})
        if isinstance(qualifiers_raw, list):
            qualifiers = {
                str(item.get("qualifier_type_id")): item.get("qualifier_value")
                for item in qualifiers_raw
                if isinstance(item, dict)
            }
        elif isinstance(qualifiers_raw, dict):
            qualifiers = qualifiers_raw
        else:
            qualifiers = {}
        qualifiers = dict(qualifiers)
        qualifiers.update({key: value for key, value in raw.items() if key.endswith("_qualifier")})
        return StatementRecord(
            id=stable_statement_id(self.source.name, native_id, raw),
            source_id=self.source.name,
            native_id=native_id,
            subject=str(raw["subject"]),
            predicate=predicate,
            object=str(raw["object"]),
            relation_role=role,
            target_eligible=role != "schema",
            categories=string_list(raw.get("category")),
            qualifiers=qualifiers,
            attributes=attributes,
            publications=string_list(raw.get("publications")),
            sources=raw.get("sources", []) if isinstance(raw.get("sources", []), list) else [raw["sources"]],
            original_record={"file": str(path), "line": line},
        )

    def batches(self, artifacts: SourceArtifacts | None = None) -> Iterator[CanonicalBatch]:
        for chunk in self.chunks("nodes"):
            nodes = []
            events = []
            terms = []
            for _, _, raw in chunk:
                identifier = str(raw["id"])
                if not identifier or any(character.isspace() for character in identifier):
                    raise ValueError(f"invalid canonical node ID: {identifier!r}")
                attributes = {key: value for key, value in raw.items() if key not in CORE_NODE_FIELDS}
                nodes.append(
                    NodeRecord(
                        id=identifier,
                        names=string_list(raw.get("name")),
                        categories=string_list(raw.get("category")),
                        in_taxon=string_list(raw.get("in_taxon")),
                        provided_by=string_list(raw.get("provided_by")) or [self.source.provenance.provided_by],
                        native_ids={self.source.name: identifier},
                        attributes=attributes,
                    )
                )
                terms.extend(self.ontology_terms("attribute", list(attributes)))
                events.extend(attribute_events("node", identifier, self.source.name, attributes))
                terms.extend(self.ontology_terms("category", nodes[-1].categories))
                terms.extend(self.ontology_terms("taxon", nodes[-1].in_taxon))
            yield CanonicalBatch(nodes=nodes, attributes=events, terms=terms)
        for chunk in self.chunks("edges"):
            statements = []
            events = []
            nodes = []
            predicates = []
            terms = []
            for path, line, raw in chunk:
                statement = self.statement(raw, path, line)
                statements.append(statement)
                terms.extend(self.ontology_terms("qualifier", list(statement.qualifiers)))
                terms.extend(self.ontology_terms("attribute", list(statement.attributes)))
                nodes.extend(self.overlay_nodes(raw))
                events.extend(attribute_events("statement", statement.id, self.source.name, statement.attributes))
                if statement.predicate not in self.predicates:
                    self.predicates.add(statement.predicate)
                    predicates.append(
                        PredicateRecord(id=statement.predicate, source_id=self.source.name, name=statement.predicate)
                    )
                    terms.extend(self.ontology_terms("predicate", [statement.predicate]))
            yield CanonicalBatch(
                nodes=nodes, statements=statements, attributes=events, predicates=predicates, terms=terms
            )
