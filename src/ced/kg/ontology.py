from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml

from ced.kg.records import CanonicalBatch, OntologyTermRecord


def biolink_ontology() -> CanonicalBatch:
    """Read the installed, locked vocabulary without fetching ontology documents."""
    path = files("biolink_model").joinpath("schema/biolink_model.yaml")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    terms = []
    for name, declaration in document.get("classes", {}).items():
        declaration = declaration or {}
        local = declaration.get("class_uri", "biolink:" + "".join(part.capitalize() for part in name.split()))
        terms.append(
            OntologyTermRecord(
                id=local,
                source_id="biolink",
                kind="category",
                name=name,
                definition=declaration.get("description"),
                attributes={"is_a": declaration.get("is_a"), "mixins": declaration.get("mixins", [])},
            )
        )
    slots = document.get("slots", {})

    def descends(name: str, ancestor: str) -> bool:
        visited = set()
        while name and name not in visited:
            if name == ancestor:
                return True
            visited.add(name)
            name = (slots.get(name) or {}).get("is_a", "")
        return False

    for name, declaration in slots.items():
        declaration = declaration or {}
        kind = (
            "predicate" if descends(name, "related to") else "qualifier" if descends(name, "qualifier") else "attribute"
        )
        local = declaration.get("slot_uri", "biolink:" + name.replace(" ", "_"))
        terms.append(
            OntologyTermRecord(
                id=local,
                source_id="biolink",
                kind=kind,
                name=name,
                definition=declaration.get("description"),
                attributes={
                    "is_a": declaration.get("is_a"),
                    "range": declaration.get("range"),
                    "domain": declaration.get("domain"),
                    "mixins": declaration.get("mixins", []),
                },
            )
        )
    return CanonicalBatch(terms=terms)


def amalgam_ontology(path: Path) -> CanonicalBatch:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    terms = []
    declarations: list[tuple[Literal["category", "predicate"], str]] = [
        ("category", "classes"),
        ("predicate", "predicates"),
    ]
    for kind, field in declarations:
        for declaration in document.get(field, []):
            name = declaration["name"]
            term_kind = "attribute" if declaration.get("kind") == "attribute" else kind
            terms.append(
                OntologyTermRecord(
                    id=f"AMALGAM:{name}",
                    source_id="amalgam",
                    kind=term_kind,
                    native_id=str(declaration["id"]),
                    name=name,
                    definition=declaration.get("definition"),
                    attributes={"parent": declaration.get("parent"), "kind": declaration.get("kind")},
                )
            )
    return CanonicalBatch(terms=terms)
