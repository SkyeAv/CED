from __future__ import annotations

from pathlib import Path

import pytest

from ced.config.models import RuntimeConfig
from ced.config.sources import (
    ArtifactSpec,
    HttpAcquisition,
    LocalAcquisition,
    ProvenanceConfig,
    SourceConfig,
    select_sources,
)


def provenance() -> ProvenanceConfig:
    return ProvenanceConfig(title="Fixture", license_name="CC0-1.0", provided_by="infores:fixture")


def fixture(name: str, *, groups: list[str], enabled: bool = True) -> SourceConfig:
    return SourceConfig(
        name=name,
        enabled=enabled,
        adapter="kgx_ndjson",
        groups=groups,
        acquisition=LocalAcquisition(backend="local", root=Path("/tmp")),
        provenance=provenance(),
        artifacts=[ArtifactSpec(path=f"{name}.ndjson", role="nodes")],
    )


def catalog() -> dict[str, SourceConfig]:
    return {
        "chembl": fixture("chembl", groups=["translator"]),
        "sider": fixture("sider", groups=["translator"]),
        "retired": fixture("retired", groups=["translator"], enabled=False),
        "curated": fixture("curated", groups=["ced"]),
        "plain": fixture("plain", groups=[]),
    }


def test_group_selection_returns_only_that_group() -> None:
    """One command must address exactly the requested release set, otherwise 21 GB of
    multiomics rides along with a 618 MiB translator fetch."""
    selected = select_sources(catalog(), None, all_enabled=False, groups=["translator"])
    assert [source.name for source in selected] == ["chembl", "sider"]


def test_group_selection_skips_disabled_members() -> None:
    """A disabled source stays out of a group fetch instead of aborting it, matching how
    --all-enabled already treats disabled sources."""
    selected = select_sources(catalog(), None, all_enabled=False, groups=["translator"])
    assert "retired" not in [source.name for source in selected]
    with pytest.raises(ValueError, match="disabled sources: retired"):
        select_sources(catalog(), ["retired"], all_enabled=False)


def test_multiple_groups_union_without_duplicates() -> None:
    """Groups compose, so a fetch can span the translator releases and the curated CED tables
    without listing every source by name."""
    selected = select_sources(catalog(), None, all_enabled=False, groups=["translator", "ced"])
    assert [source.name for source in selected] == ["chembl", "sider", "curated"]


def test_unknown_group_raises() -> None:
    """A typo in a group name must fail loudly rather than silently selecting nothing."""
    with pytest.raises(ValueError, match="unknown groups: translatr"):
        select_sources(catalog(), None, all_enabled=False, groups=["translatr"])


def test_selection_modes_are_mutually_exclusive() -> None:
    """Two selection modes at once have no defined meaning; refusing them prevents a fetch
    that quietly covers a different set than the operator intended."""
    with pytest.raises(ValueError, match="not more than one"):
        select_sources(catalog(), ["chembl"], all_enabled=False, groups=["translator"])
    with pytest.raises(ValueError, match="not more than one"):
        select_sources(catalog(), None, all_enabled=True, groups=["translator"])


def test_group_names_are_validated() -> None:
    """Group names become CLI arguments and report keys, so they follow the same identifier
    rule as source names."""
    with pytest.raises(ValueError, match="invalid source groups"):
        fixture("bad", groups=["Translator_1"])


def test_fetch_sources_passes_group_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI flag has to reach the pipeline; a group that resolves to nothing must not turn
    into an all-enabled fetch."""
    from ced.acquisition import pipeline as acquisition_pipeline

    config = RuntimeConfig(remote_project=tmp_path, artifact_root=Path(".ced"), disk_reserve_gb=0)
    prepared: list[str] = []
    monkeypatch.setattr(acquisition_pipeline, "require_cpu_worker", lambda runtime: None)
    monkeypatch.setattr(acquisition_pipeline, "load_sources", lambda path: catalog())
    monkeypatch.setattr(
        acquisition_pipeline,
        "prepare_source",
        lambda runtime, source, *, allow_download: prepared.append(source.name) or _Prepared(source),
    )
    acquisition_pipeline.fetch_sources(config, [], groups=["translator"], all_enabled=False, assume_yes=False)
    assert prepared == ["chembl", "sider"]


class _Prepared:
    """Minimal PreparedSource stand-in: a group fetch of empty sources writes manifests only."""

    def __init__(self, source: SourceConfig) -> None:
        self.source = source
        self.backend = None
        self.artifacts = []
        self.inventory_bytes = 0


def test_cli_forwards_group_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch and build must forward --group unchanged, including over the SSH hop."""
    import ced.cli as cli

    monkeypatch.setenv("CED_REMOTE_WORKER", "1")
    recorded: dict[str, object] = {}

    def fake_fetch_sources(runtime, names, *, groups=None, all_enabled=False, assume_yes=False):
        recorded["fetch"] = (names, groups, all_enabled)

    def fake_build_graph(runtime, names, *, groups=None, fullmap_path=None):
        recorded["build"] = (names, groups)

    monkeypatch.setattr("ced.acquisition.pipeline.fetch_sources", fake_fetch_sources)
    monkeypatch.setattr("ced.kg.pipeline.build_graph", fake_build_graph)
    monkeypatch.setattr("ced.publication.publish_snapshot", lambda runtime, root: None)
    cli.fetch(source=None, group=["translator"], all_enabled=False, yes=True)
    cli.build(source=None, group=["translator"], fullmap_path=None)
    assert recorded["fetch"] == ([], ["translator"], False)
    assert recorded["build"] == ([], ["translator"])


def test_sources_list_prints_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Operators inspect groups before fetching anything; the listing is the only place that
    shows them without reading YAML."""
    import ced.cli as cli

    monkeypatch.setattr(cli, "load_runtime", lambda: RuntimeConfig(remote_project=tmp_path, artifact_root=Path(".ced")))
    monkeypatch.setattr("ced.config.sources.load_sources", lambda path: catalog())
    cli.sources_list()
    rows = {line.split("\t")[0]: line.split("\t") for line in capsys.readouterr().out.splitlines()}
    assert rows["chembl"][3] == "translator"
    assert rows["plain"][3] == "-"


def test_http_acquisition_accepts_grouped_source(tmp_path: Path) -> None:
    """The translator sources combine the http backend with group membership; both config
    concepts must validate together."""
    source = SourceConfig(
        name="chembl",
        adapter="kgx_ndjson",
        groups=["translator"],
        acquisition=HttpAcquisition(backend="http", connections=8),
        provenance=provenance(),
        artifacts=[ArtifactSpec(path="chembl.tar.zst", role="kgx_archive")],
    )
    assert source.groups == ["translator"]
    assert source.acquisition.backend == "http"
