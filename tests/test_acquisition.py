import hashlib
import io
import subprocess
import tarfile
from pathlib import Path

import pytest
import zstandard

from ced.acquisition.archive import extract_archive
from ced.acquisition.aria2 import Aria2Backend
from ced.acquisition.base import FetchedArtifact
from ced.acquisition.huggingface import HuggingFaceBackend
from ced.acquisition.pipeline import PreparedSource, fetch_sources
from ced.config.models import RuntimeConfig
from ced.config.sources import (
    ArchiveSpec,
    Aria2Acquisition,
    ArtifactInventorySpec,
    ArtifactSpec,
    HuggingFaceAcquisition,
    ProvenanceConfig,
    SourceConfig,
    load_sources,
    select_sources,
)

PAYLOAD = b'{"id":"HGNC:1"}\n'
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def runtime(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(remote_project=tmp_path, artifact_root=Path(".ced"), disk_reserve_gb=0)


def provenance() -> ProvenanceConfig:
    return ProvenanceConfig(title="Fixture", license_name="CC0-1.0", provided_by="infores:fixture")


def source(acquisition, artifacts: list[ArtifactSpec] | None = None, inventory=None) -> SourceConfig:
    return SourceConfig(
        name="fixture",
        adapter="kgx_ndjson",
        acquisition=acquisition,
        provenance=provenance(),
        artifacts=artifacts or [ArtifactSpec(path="nodes.ndjson", role="nodes", sha256=DIGEST)],
        inventory=inventory,
    )


def test_source_yaml_is_declarative(tmp_path: Path) -> None:
    directory = tmp_path / "sources"
    directory.mkdir()
    (directory / "fixture.yaml").write_text(
        f"""name: fixture
adapter: kgx_ndjson
acquisition: {{ backend: local, root: {tmp_path} }}
provenance: {{ title: Fixture, license_name: CC0-1.0, provided_by: 'infores:fixture' }}
artifacts:
  - {{ path: nodes.ndjson, role: nodes, sha256: {DIGEST} }}
""",
        encoding="utf-8",
    )
    sources = load_sources(directory)
    assert sources["fixture"].acquisition.backend == "local"
    assert select_sources(sources, ["fixture"])[0].name == "fixture"
    with pytest.raises(ValueError, match="fetch requires"):
        select_sources(sources, [], require_explicit=True)


def test_amalgam_inventory_is_hash_pinned() -> None:
    inventory = ArtifactInventorySpec(
        manifest="manifest.json",
        format="amalgam_compact",
        include=["vertices/"],
        roles={"vertices/": "vertices"},
    )
    document = {
        "schema_version": 1,
        "encoding": "global-int32-uuid-binary16-zstd9",
        "lossless": True,
        "compact": {
            "parquet_files": 1,
            "sha256": {"vertices/vertices.parquet": DIGEST},
            "row_counts": {"vertices/vertices.parquet": 1},
        },
    }
    artifacts = inventory.materialize(document)
    assert artifacts == [ArtifactSpec(path="vertices/vertices.parquet", role="vertices", sha256=DIGEST)]
    document["lossless"] = False
    with pytest.raises(ValueError, match="lossless"):
        inventory.materialize(document)


def test_aria2_resume_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime(tmp_path)
    selected = source(
        Aria2Acquisition(backend="aria2"),
        [ArtifactSpec(path="nodes.ndjson", role="nodes", urls=["https://example.invalid/nodes"], sha256=DIGEST)],
    )
    backend = Aria2Backend(config, selected)
    calls: list[list[str]] = []

    def download(command: list[str], check: bool) -> None:
        assert check
        calls.append(command)
        values = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in command if item.startswith("--")}
        (Path(values["--dir"]) / values["--out"]).write_bytes(PAYLOAD)

    monkeypatch.setattr("ced.acquisition.aria2.require_program", lambda name: "/usr/bin/aria2c")
    monkeypatch.setattr(subprocess, "run", download)
    first = backend.fetch(selected.artifacts[0], allow_download=True)
    second = backend.fetch(selected.artifacts[0], allow_download=False)
    assert first.path == second.path
    assert len(calls) == 1
    assert "--continue=true" in calls[0]


def test_aria2_checksum_failure_retains_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selected = source(
        Aria2Acquisition(backend="aria2"),
        [ArtifactSpec(path="nodes.ndjson", role="nodes", urls=["https://example.invalid/nodes"], sha256=DIGEST)],
    )
    backend = Aria2Backend(runtime(tmp_path), selected)
    monkeypatch.setattr("ced.acquisition.aria2.require_program", lambda name: "/usr/bin/aria2c")

    def download(command: list[str], check: bool) -> None:
        values = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in command if item.startswith("--")}
        (Path(values["--dir"]) / values["--out"]).write_bytes(b"corrupt")

    monkeypatch.setattr(subprocess, "run", download)
    with pytest.raises(ValueError, match="SHA-256"):
        backend.fetch(selected.artifacts[0], allow_download=True)
    assert backend.destination(selected.artifacts[0]).with_suffix(".ndjson.part").is_file()


def test_offline_fetch_fails_with_instruction(tmp_path: Path) -> None:
    selected = source(
        Aria2Acquisition(backend="aria2"),
        [ArtifactSpec(path="nodes.ndjson", role="nodes", urls=["https://example.invalid/nodes"])],
    )
    backend = Aria2Backend(runtime(tmp_path), selected)
    with pytest.raises(FileNotFoundError, match="ced fetch --source fixture"):
        backend.fetch(selected.artifacts[0], allow_download=False)


def test_huggingface_offline_uses_native_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = tmp_path / "cached.ndjson"
    payload.write_bytes(PAYLOAD)
    selected = source(HuggingFaceAcquisition(backend="huggingface", repo_id="SkyeAv/fixture", revision="a" * 40))
    backend = HuggingFaceBackend(runtime(tmp_path), selected)

    def download(**kwargs) -> str:
        assert kwargs["local_files_only"] is True
        assert kwargs["revision"] == "a" * 40
        return str(payload)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    artifact = backend.fetch(selected.artifacts[0], allow_download=False)
    assert isinstance(artifact, FetchedArtifact)
    assert artifact.path == payload


def make_archive(path: Path, symlink: bool = False) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as bundle:
        for name, data in (("nodes.ndjson", PAYLOAD), ("edges.ndjson", PAYLOAD), ("test-data.json", b"{}")):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
        if symlink:
            member = tarfile.TarInfo("escape")
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            bundle.addfile(member)
    path.write_bytes(zstandard.ZstdCompressor().compress(raw.getvalue()))


def test_safe_archive_extracts_only_declared_members(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.tar.zst"
    make_archive(archive)
    spec = ArchiveSpec(
        format="tar.zst", members={"nodes": ["*.nodes.ndjson", "nodes.ndjson"], "edges": ["edges.ndjson"]}
    )
    destination = tmp_path / "extracted"
    result = extract_archive(archive, spec, destination)
    assert result["nodes"][0].read_bytes() == PAYLOAD
    assert result["edges"][0].read_bytes() == PAYLOAD
    assert not (destination / "test-data.json").exists()
    assert extract_archive(archive, spec, destination) == result


def test_archive_rejects_links(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.tar.zst"
    make_archive(archive, symlink=True)
    spec = ArchiveSpec(format="tar.zst", members={"nodes": ["nodes.ndjson"], "edges": ["edges.ndjson"]})
    with pytest.raises(RuntimeError, match="safe archive extraction failed"):
        extract_archive(archive, spec, tmp_path / "bad")


def test_large_download_requires_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selected = source(Aria2Acquisition(backend="aria2"))
    prepared = PreparedSource(selected, Aria2Backend(runtime(tmp_path), selected), [], inventory_bytes=1024**3)
    monkeypatch.setattr("ced.acquisition.pipeline.require_cpu_worker", lambda config: None)
    monkeypatch.setattr("ced.acquisition.pipeline.load_sources", lambda path: {"fixture": selected})
    monkeypatch.setattr("ced.acquisition.pipeline.prepare_source", lambda *args, **kwargs: prepared)
    with pytest.raises(RuntimeError, match="--yes"):
        fetch_sources(runtime(tmp_path), ["fixture"], all_enabled=False, assume_yes=False)
