import json
from pathlib import Path

import pytest

from ced.config.models import RuntimeConfig
from ced.config.sources import load_sources
from ced.kg.adapters.tablassert import CedIdentityFactory, TablassertAdapter


def adapter(tmp_path: Path) -> TablassertAdapter:
    runtime = RuntimeConfig(remote_project=Path.cwd(), artifact_root=tmp_path / ".ced")
    source = load_sources(Path("src/ced/config/sources"))["ced"]
    return TablassertAdapter(runtime, source)


def test_local_identity_is_stable_and_scoped() -> None:
    factory = CedIdentityFactory({})
    assert factory.make("Family 1", "PopulationOfIndividualOrganisms") == "CED:Family_1"
    first = factory.make("unmapped variant", "SequenceVariant")
    assert first == factory.make("unmapped variant", "SequenceVariant")
    assert first.startswith("CED:")
    assert first != factory.make("unmapped variant", "Protein")


def test_all_family_and_book_sections_validate(tmp_path: Path) -> None:
    _, sections = adapter(tmp_path).sections()
    assert len(sections) == 612
    custom = 0
    for section in sections:
        tcode = section.tcode
        assert isinstance(tcode.source.url, list)
        assert tcode.source.delimiter == "\t"
        assert all(annotation.encoding is not None for annotation in tcode.annotations)
        lineage = {annotation.annotation: annotation.encoding for annotation in tcode.annotations}
        assert lineage["ced_source_config"]
        assert lineage["ced_section_index"]
        if lineage["ced_original_predicate"] == "has_mutation":
            custom += 1
            assert tcode.statement.predicate == "related_to"
            assert lineage["ced_predicate"] == "ced:has_mutation"
    assert custom == 70


@pytest.mark.fullmap
def test_fullmap_fallback_and_extension_build(tmp_path: Path) -> None:
    selected = adapter(tmp_path)
    tables = [
        Path("src/ced/graphs/tables/families/dna-mutations.yaml"),
        Path("src/ced/graphs/tables/families/treatments.yaml"),
        Path("src/ced/graphs/tables/books/camurati-engelmann-disease-genereviews.yaml"),
    ]
    result = selected.build(table_paths=tables, section_limit=1)
    nodes = [json.loads(line) for line in result.nodes.read_text(encoding="utf-8").splitlines()]
    edges = [json.loads(line) for line in result.edges.read_text(encoding="utf-8").splitlines()]
    assert "CED:Family_1" in {node["id"] for node in nodes}
    mutation = next(edge for edge in edges if edge.get("ced_original_predicate") == "has_mutation")
    assert mutation["ced_predicate"] == "ced:has_mutation"
    assert mutation["subject"] == "CED:Family_1"
    assert mutation["subject_mapping_status"] == "unresolved_local"
    treatment = next(edge for edge in edges if edge.get("population_context_qualifier") == "CED:Family_1")
    assert treatment["population_context_qualifier_mapping_status"] == "unresolved_local"
    book = next(edge for edge in edges if edge.get("update_date") == "2023-08-31")
    assert book["supporting_text"]
    assert result.rig.is_file()
