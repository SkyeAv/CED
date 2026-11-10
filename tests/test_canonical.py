from __future__ import annotations

import json
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ced.acquisition.pipeline import SourceArtifacts
from ced.config.sources import ArtifactSpec, LocalAcquisition, ProvenanceConfig, SourceConfig
from ced.kg.adapters.amalgam_compact import AmalgamCompactAdapter, canonical_id
from ced.kg.adapters.kgx_ndjson import KgxNdjsonAdapter
from ced.kg.merge import merge_graph
from ced.kg.records import CanonicalBatch, NodeRecord, StatementRecord, attribute_events, stable_statement_id
from ced.kg.writer import CanonicalBatchWriter


def fixture_source(root: Path, adapter: str = "kgx_ndjson") -> SourceConfig:
    return SourceConfig(
        name="fixture",
        adapter=adapter,
        artifacts=[ArtifactSpec(path="fixture", role="metadata")],
        acquisition=LocalAcquisition(backend="local", root=root),
        provenance=ProvenanceConfig(title="Fixture", license_name="CC0", provided_by="infores:fixture"),
    )


def test_attribute_events_preserve_values() -> None:
    values = {"zero": 0, "false": False, "blank": "", "null": None, "nested": [1, {"x": "y"}], "nan": float("nan")}
    events = {event.path: event for event in attribute_events("node", "X:1", "fixture", values)}
    assert events["zero"].numeric == 0
    assert events["false"].value_type == "bool"
    assert events["blank"].text == ""
    assert events["null"].value_type == "null"
    assert events["nested[1].x"].text == "y"
    assert events["nan"].value_type == "nonfinite"
    assert events["nan"].numeric is None
    assert stable_statement_id("a", None, {"x": 1, "y": 2}) == stable_statement_id("a", None, {"y": 2, "x": 1})
    assert stable_statement_id("a", "native", {}) != stable_statement_id("b", "native", {})


def test_kgx_merge_union_and_evidence(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes.ndjson"
    edges = tmp_path / "edges.ndjson"
    nodes.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "X:1", "name": "One", "category": ["biolink:Gene"], "custom": {"zero": 0}},
                {"id": "X:1", "name": "Uno", "category": ["AMALGAM:Gene"]},
                {"id": "X:2", "name": "Two", "category": ["AMALGAM:Disease"]},
            ]
        )
    )
    edges.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "e1", "subject": "X:1", "predicate": "biolink:related_to", "object": "X:2", "evidence": False},
                {
                    "id": "e2",
                    "subject": "X:1",
                    "predicate": "AMALGAM:expanded",
                    "object": "X:2",
                    "qualifiers": {"AMALGAM:context": "yes"},
                },
                {
                    "id": "e3",
                    "subject": "X:1",
                    "predicate": "biolink:related_to",
                    "object": "X:2",
                    "evidence": "different",
                },
            ]
        )
    )
    source = fixture_source(tmp_path)
    artifacts = SourceArtifacts(source=source, files={}, members={"nodes": [nodes], "edges": [edges]})
    root = tmp_path / "snapshot"
    writer = CanonicalBatchWriter(root / "raw", shard_rows=2)
    for batch in KgxNdjsonAdapter(artifacts).batches():
        writer.write(batch)
    writer.close()
    counts = merge_graph(root, memory="128MB")
    assert counts["nodes"] == 2
    assert counts["statements"] == 3
    merged = pq.read_table(root / "nodes").to_pylist()
    one = next(row for row in merged if row["id"] == "X:1")
    assert one["names"] == ["One", "Uno"]
    assert set(one["categories"]) == {"biolink:Gene", "AMALGAM:Gene"}
    terms = pq.read_table(root / "ontology_terms").to_pylist()
    assert {"biolink:Gene", "AMALGAM:Gene", "AMALGAM:expanded"} <= {row["id"] for row in terms}


def compact_fixture(tmp_path: Path) -> SourceArtifacts:
    source = fixture_source(tmp_path, "amalgam_compact")
    source.mapping.schema_relations = ["has-category", "in-taxon", "never-in-taxon"]
    vertices = [
        {"ID": 1, "VERTEX_ID": uuid.UUID(int=1).bytes, "KIND": "entity", "KEY": "HGNC:1", "NAME": "Gene"},
        {"ID": 2, "VERTEX_ID": uuid.UUID(int=2).bytes, "KIND": "entity", "KEY": "MONDO:1", "NAME": "Disease"},
        {"ID": 3, "VERTEX_ID": uuid.UUID(int=3).bytes, "KIND": "category", "KEY": "Gene", "NAME": "Gene"},
        {"ID": 4, "VERTEX_ID": uuid.UUID(int=4).bytes, "KIND": "taxon", "KEY": "NCBITaxon:9606", "NAME": "Human"},
    ]
    tables = {
        "vertices": vertices,
        "categories": [{"ID": 1, "VERTEX_ID": 3, "NAME": "Gene", "DEFINITION": "A gene"}],
        "predicates": [
            {"ID": i, "VERTEX_ID": 3, "NAME": name, "KIND": kind, "DEFINITION": name}
            for i, name, kind in [
                (1, "expanded", "relation"),
                (2, "has-category", "relation"),
                (3, "in-taxon", "relation"),
                (4, "has-name", "attribute"),
            ]
        ],
        "taxa": [{"ID": 9606, "VERTEX_ID": 4, "NAME": "Human", "RANK": "species"}],
        "relations": [
            {"SUBJECT_ID": 1, "PREDICATE_ID": 1, "OBJECT_ID": 2, "CATEGORY_ID": None, "TAXON_ID": None},
            {"SUBJECT_ID": 1, "PREDICATE_ID": 2, "OBJECT_ID": None, "CATEGORY_ID": 1, "TAXON_ID": None},
            {"SUBJECT_ID": 1, "PREDICATE_ID": 3, "OBJECT_ID": None, "CATEGORY_ID": None, "TAXON_ID": 9606},
        ],
        "attributes": [{"SUBJECT_ID": 1, "PREDICATE_ID": 4, "VALUE": "Gene symbol", "L1": "gene symbol", "L2": "gene"}],
        "conflicts": [
            {
                "LEVEL": "L1",
                "CATEGORY": "Gene",
                "TAXON": "9606",
                "VALUE": "same",
                "NAME_KEY": "same",
                "LOSER_KEY": "old",
                "WINNER_ID": 1,
            }
        ],
    }
    members = {}
    for name, rows in tables.items():
        path = tmp_path / f"{name}.parquet"
        table = pa.Table.from_pylist(rows)
        fields = []
        for field in table.schema:
            if field.name in {
                "ID",
                "SUBJECT_ID",
                "PREDICATE_ID",
                "OBJECT_ID",
                "CATEGORY_ID",
                "TAXON_ID",
                "WINNER_ID",
            } or (field.name == "VERTEX_ID" and name != "vertices"):
                fields.append(pa.field(field.name, pa.int32()))
            elif field.name == "VERTEX_ID":
                fields.append(pa.field(field.name, pa.binary(16)))
            else:
                fields.append(field)
        pq.write_table(table.cast(pa.schema(fields)), path)
        members[name] = [path]
    return SourceArtifacts(source=source, files={}, members=members)


def test_compact_adapter_preserves_schema_and_attributes(tmp_path: Path) -> None:
    artifacts = compact_fixture(tmp_path)
    adapter = AmalgamCompactAdapter(artifacts, tmp_path / "spill")
    try:
        output = list(adapter.batches())
    finally:
        adapter.close()
    statements = [statement for batch in output for statement in batch.statements]
    assert len(statements) == 3
    assert sum(statement.target_eligible for statement in statements) == 1
    assert any(statement.predicate == "AMALGAM:expanded" for statement in statements)
    assert any(statement.object == "NCBITaxon:9606" for statement in statements)
    nodes = [node for batch in output for node in batch.nodes]
    gene = next(node for node in nodes if node.id == "HGNC:1")
    assert "AMALGAM:Gene" in gene.categories
    assert "NCBITaxon:9606" in gene.in_taxon
    events = [event for batch in output for event in batch.attributes]
    assert any(event.path.endswith("L2") and event.text == "gene" for event in events)
    assert any(batch.conflicts for batch in output)
    assert canonical_id("category", "Gene", uuid.UUID(int=3).bytes) == f"AMALGAM:{uuid.UUID(int=3)}"
    with pytest.raises(ValueError, match="malformed"):
        canonical_id("entity", "not a curie", uuid.UUID(int=1).bytes)


def test_missing_endpoint_rejected(tmp_path: Path) -> None:
    writer = CanonicalBatchWriter(tmp_path / "raw")
    writer.write(
        CanonicalBatch(
            nodes=[NodeRecord(id="X:1")],
            statements=[StatementRecord(id="e", source_id="s", subject="X:1", predicate="p", object="X:missing")],
        )
    )
    writer.close()
    with pytest.raises(ValueError, match="missing node"):
        merge_graph(tmp_path, memory="128MB")


def test_compact_rejects_dangling_reference(tmp_path: Path) -> None:
    artifacts = compact_fixture(tmp_path)
    path = artifacts.paths("relations")[0]
    table = pq.read_table(path)
    index = table.schema.get_field_index("SUBJECT_ID")
    table = table.set_column(index, "SUBJECT_ID", pa.array([999, 1, 1], type=pa.int32()))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="dangling Amalgam reference"):
        AmalgamCompactAdapter(artifacts, tmp_path / "spill")


def test_checksum_validation_rejects_changes_and_traversal(tmp_path: Path) -> None:
    from ced.acquisition.base import sha256_file
    from ced.kg.audit import validate_checksums

    path = tmp_path / "artifact"
    path.write_text("original")
    checksums = {"artifact": sha256_file(path)}
    validate_checksums(tmp_path, checksums)
    path.write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_checksums(tmp_path, checksums)
    with pytest.raises(ValueError, match="escapes snapshot"):
        validate_checksums(tmp_path, {"../escape": "invalid"})


def test_native_statement_conflict_rejected(tmp_path: Path) -> None:
    writer = CanonicalBatchWriter(tmp_path / "raw")
    writer.write(
        CanonicalBatch(
            nodes=[NodeRecord(id="X:1"), NodeRecord(id="X:2")],
            statements=[
                StatementRecord(
                    id="same",
                    source_id="s",
                    subject="X:1",
                    predicate="p",
                    object="X:2",
                    attributes={"evidence": "first"},
                ),
                StatementRecord(
                    id="same",
                    source_id="s",
                    subject="X:1",
                    predicate="p",
                    object="X:2",
                    attributes={"evidence": "different"},
                ),
            ],
        )
    )
    writer.close()
    with pytest.raises(ValueError, match="conflicting statement"):
        merge_graph(tmp_path, memory="128MB")


def test_installed_biolink_ontology_is_offline() -> None:
    from ced.kg.ontology import biolink_ontology

    terms = biolink_ontology().terms
    assert any(term.id == "biolink:Gene" and term.kind == "category" for term in terms)
    assert any(term.id == "biolink:related_to" and term.kind == "predicate" for term in terms)
    assert any(term.id == "biolink:qualifier" and term.kind == "qualifier" for term in terms)


def test_amalgam_ontology_retains_expanded_terms(tmp_path: Path) -> None:
    from ced.kg.ontology import amalgam_ontology

    path = tmp_path / "ontology.yaml"
    path.write_text(
        "classes:\n  - {id: 1, name: ExpandedClass, parent: Entity, definition: Expanded}\npredicates:\n  - {id: 2, name: expanded-relation, kind: relation}\n  - {id: 3, name: expanded-attribute, kind: attribute}\n"
    )
    terms = amalgam_ontology(path).terms
    assert {(term.id, term.kind) for term in terms} == {
        ("AMALGAM:ExpandedClass", "category"),
        ("AMALGAM:expanded-relation", "predicate"),
        ("AMALGAM:expanded-attribute", "attribute"),
    }
