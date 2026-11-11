from __future__ import annotations

from pathlib import Path

import pytest

from ced.config.sources import load_sources, select_sources

SOURCES = Path(__file__).resolve().parents[1] / "src/ced/config/sources"
RELEASE_ROOT = "https://kgx-storage.ci.transltr.io/releases/"
TRANSLATOR = {
    "ubergraph": "ubergraph/1.0.4",
    "bindingdb": "bindingdb/1.0.2",
    "chembl": "chembl/1.0.2",
    "dgidb": "dgidb/1.0.3",
    "diseases": "diseases/1.0.5",
    "drugcentral": "drugcentral/1.0.2",
    "drug-rep-hub": "drug_rep_hub/1.0.2",
    "gene2phenotype": "gene2phenotype/1.0.6",
    "geneticskp": "geneticskp/1.0.6",
    "sider": "sider/1.0.2",
    "ncbi-gene": "ncbi_gene/1.0.5",
    "panther": "panther/1.0.2",
    "pathbank": "pathbank/1.0.2",
}


@pytest.fixture(scope="module")
def catalog():
    return load_sources(SOURCES)


def translator(catalog) -> list:
    return select_sources(catalog, None, all_enabled=False, groups=["translator"])


def test_every_requested_release_is_declared(catalog) -> None:
    """The task is to ingest these thirteen releases; a missing or misnamed file silently drops
    a source from the graph, so the set is asserted explicitly."""
    assert set(TRANSLATOR) <= set(catalog)
    assert {source.name for source in translator(catalog)} == set(TRANSLATOR)


def test_group_excludes_the_bulk_releases(catalog) -> None:
    """A group fetch must stay bounded: multiomics alone is a 21 GB edge file, so it stays out
    of the translator group."""
    names = {source.name for source in translator(catalog)}
    assert not {"multiomics", "drug-approvals", "amalgam", "ced"} & names


def test_artifacts_pin_a_concrete_release(catalog) -> None:
    """/latest/ moves, which would break reproducibility and every pinned size; each artifact must
    name its release directory and its own basename."""
    for name, release in TRANSLATOR.items():
        source = catalog[name]
        for artifact in source.artifacts:
            prefix = f"{RELEASE_ROOT}{release}/"
            assert artifact.urls, f"{name}/{artifact.path} has no URL"
            for url in artifact.urls:
                assert url.startswith(prefix), f"{name}/{artifact.path} is not pinned to {prefix}"
                assert "/latest/" not in url
                assert url.rsplit("/", 1)[-1] == artifact.path


def test_artifacts_pin_an_exact_byte_size(catalog) -> None:
    """Sizes come from the release host's content-length; without them a truncated download
    verifies as complete and corrupts the graph."""
    for name in TRANSLATOR:
        for artifact in catalog[name].artifacts:
            assert artifact.size is not None, f"{name}/{artifact.path} has no pinned size"
            assert artifact.size > 0, f"{name}/{artifact.path} has a non-positive pinned size"


def test_archive_declares_nodes_and_edges_with_expanded_caps(catalog) -> None:
    """The KGX adapter reads the extracted members by role, and the archive guard needs bounds
    large enough for the real expansion (ubergraph is ~399 MiB compressed)."""
    for name in TRANSLATOR:
        archives = [artifact for artifact in catalog[name].artifacts if artifact.archive is not None]
        assert len(archives) == 1, f"{name} must declare exactly one archive"
        archive = archives[0].archive
        assert archive is not None
        assert archive.format == "tar.zst"
        assert set(archive.members) == {"nodes", "edges"}
        assert archive.members["nodes"] == ["*nodes*.ndjson"]
        assert archive.members["edges"] == ["*edges*.ndjson"]
        assert archive.max_members >= 2
        assert archive.max_total_bytes >= archive.max_member_bytes > archives[0].size
        assert archives[0].size == max(artifact.size or 0 for artifact in catalog[name].artifacts)


def test_provenance_is_evidence_backed(catalog) -> None:
    """License reporting and lineage come from these fields; an empty license or a non-infores
    provider would silently misattribute upstream data."""
    for name, release in TRANSLATOR.items():
        provenance = catalog[name].provenance
        assert provenance.license_name
        assert provenance.license_url
        assert provenance.provided_by.startswith("infores:")
        assert provenance.homepage == f"{RELEASE_ROOT}{release}/"
        assert provenance.citations
        assert any("Translator release" in citation for citation in provenance.citations)


def test_translator_sources_share_the_kgx_contract(catalog) -> None:
    """All thirteen come from one producer with conflated strict normalization; a source that
    drifts from that contract would be canonicalized with the wrong assumptions."""
    for name in TRANSLATOR:
        source = catalog[name]
        assert source.adapter == "kgx_ndjson"
        assert source.enabled
        assert source.acquisition.backend == "http"
        assert source.mapping.options["upstream_conflated"] is True
        assert {artifact.role for artifact in source.artifacts} == {
            "kgx_archive",
            "release_metadata",
            "graph_metadata",
            "test_data",
        }


def test_upstream_directory_names_survive_hyphenated_source_names(catalog) -> None:
    """SourceConfig names cannot contain underscores, but the release directories can; the
    mapping has to keep the upstream path verbatim."""
    for name, release in TRANSLATOR.items():
        upstream = release.split("/")[0]
        assert catalog[name].artifacts[0].path == f"{upstream}.tar.zst"
