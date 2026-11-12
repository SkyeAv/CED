import hashlib
import io
import json
import logging
import subprocess
import tarfile
import threading
from pathlib import Path

import pytest
import zstandard

from ced.acquisition.archive import extract_archive
from ced.acquisition.aria2 import Aria2Backend
from ced.acquisition.base import FetchedArtifact
from ced.acquisition.huggingface import HuggingFaceBackend
from ced.acquisition.local import LocalBackend
from ced.acquisition.pipeline import (
    PreparedSource,
    fetch_sources,
    locate_source,
    materialize_source,
    parallel_map,
    prepare_source,
    resolve_source,
)
from ced.config.models import RuntimeConfig
from ced.config.sources import (
    ArchiveSpec,
    Aria2Acquisition,
    ArtifactInventorySpec,
    ArtifactSpec,
    HuggingFaceAcquisition,
    LocalAcquisition,
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
    with pytest.raises(RuntimeError, match=r"safe archive extraction of fixture\.tar\.zst failed") as failure:
        extract_archive(archive, spec, tmp_path / "bad")
    assert isinstance(failure.value.__cause__, ValueError)
    assert "archive links/devices are forbidden" in str(failure.value)


def test_archive_names_roles_whose_patterns_match_nothing(tmp_path: Path) -> None:
    """The real Translator releases ship nodes.jsonl, so a stale *.ndjson pattern matched nothing;
    the failure has to name the unmatched role and the archive instead of hiding the cause."""
    archive = tmp_path / "fixture.tar.zst"
    make_archive(archive)
    spec = ArchiveSpec(format="tar.zst", members={"nodes": ["*.jsonl"], "edges": ["edges.ndjson"]})
    with pytest.raises(RuntimeError, match=r"safe archive extraction of fixture\.tar\.zst failed") as failure:
        extract_archive(archive, spec, tmp_path / "missing")
    assert "archive roles have no members: nodes" in str(failure.value)
    assert (tmp_path / "missing").exists() is False


def test_local_backend_fetches_verified_files(tmp_path: Path) -> None:
    """Local acquisition must apply the same checksum verification as network backends:
    a mutated cached file is just as dangerous as a corrupted download."""
    root = tmp_path / "mirror"
    root.mkdir()
    payload_path = root / "nodes.ndjson"
    payload_path.write_bytes(PAYLOAD)
    selected = source(LocalAcquisition(backend="local", root=root))
    config = runtime(tmp_path)
    artifact = LocalBackend(config, selected).fetch(selected.artifacts[0], allow_download=False)
    assert artifact.path == payload_path
    assert artifact.sha256 == DIGEST
    assert artifact.size == len(PAYLOAD)
    assert artifact.origin == str(payload_path)

    with pytest.raises(FileNotFoundError, match="local artifact is missing"):
        LocalBackend(config, selected).fetch(
            ArtifactSpec(path="absent.ndjson", role="nodes", sha256=DIGEST), allow_download=False
        )
    with pytest.raises(TypeError, match="requires local acquisition"):
        LocalBackend(config, source(Aria2Acquisition(backend="aria2")))


def test_artifact_helpers_verify_copy_and_cache(tmp_path: Path) -> None:
    """verify_artifact, cached_or_verify and copy_atomic are the shared integrity floor for
    every backend; size, digest and mutation must each fail loudly and atomically."""
    from ced.acquisition.base import cached_or_verify, copy_atomic, verify_artifact

    payload_path = tmp_path / "payload.ndjson"
    payload_path.write_bytes(PAYLOAD)
    spec = ArtifactSpec(path="payload.ndjson", role="nodes", sha256=DIGEST, size=len(PAYLOAD))
    selected = source(LocalAcquisition(backend="local", root=tmp_path))

    assert verify_artifact(payload_path, spec) == DIGEST
    with pytest.raises(FileNotFoundError):
        verify_artifact(tmp_path / "absent", spec)
    with pytest.raises(ValueError, match="size mismatch"):
        verify_artifact(payload_path, spec.model_copy(update={"size": 99}))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_artifact(payload_path, spec.model_copy(update={"sha256": "0" * 64}))

    assert cached_or_verify(tmp_path / "absent", selected, spec, "fixture") is None
    cached = cached_or_verify(payload_path, selected, spec, "fixture")
    assert cached is not None
    assert cached.sha256 == DIGEST
    payload_path.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="cached artifact failed verification"):
        cached_or_verify(payload_path, selected, spec, "fixture")

    payload_path.write_bytes(PAYLOAD)
    destination = tmp_path / "nested" / "copy.ndjson"
    copy_atomic(payload_path, destination)
    assert destination.read_bytes() == PAYLOAD


def test_prepare_source_materializes_inventory(tmp_path: Path) -> None:
    """An inventory-backed source must fetch its manifest, hash-pin the materialized
    artifacts and carry the declared total size into the download-approval budget."""
    root = tmp_path / "mirror"
    root.mkdir()
    vertices = root / "vertices" / "vertices.parquet"
    vertices.parent.mkdir(parents=True, exist_ok=True)
    vertices.write_bytes(PAYLOAD)
    document = {
        "schema_version": 1,
        "encoding": "global-int32-uuid-binary16-zstd9",
        "lossless": True,
        "compact": {
            "parquet_files": 1,
            "total_bytes": len(PAYLOAD),
            "sha256": {"vertices/vertices.parquet": DIGEST},
            "row_counts": {"vertices/vertices.parquet": 1},
        },
    }
    document_json = json.dumps(document).encode()
    manifest_digest = hashlib.sha256(document_json).hexdigest()
    (root / "manifest.json").write_bytes(document_json)
    vertices_spec = ArtifactSpec(path="vertices/vertices.parquet", role="vertices", sha256=DIGEST)
    manifest_spec = ArtifactSpec(path="manifest.json", role="manifest", sha256=manifest_digest)
    selected = source(
        LocalAcquisition(backend="local", root=root),
        artifacts=[manifest_spec, vertices_spec],
        inventory=ArtifactInventorySpec(
            manifest="manifest.json", format="amalgam_compact", include=["vertices/"], roles={"vertices/": "vertices"}
        ),
    )
    prepared = prepare_source(runtime(tmp_path), selected, allow_download=False)
    assert [artifact.path for artifact in prepared.artifacts] == [
        "manifest.json",
        "vertices/vertices.parquet",
        "vertices/vertices.parquet",
    ]
    assert prepared.inventory_bytes == len(PAYLOAD)
    result = materialize_source(prepared, allow_download=False)
    assert [artifact.path.name for artifact in result.files["manifest"]] == ["manifest.json"]
    assert result.paths("vertices")[0].name == "vertices.parquet"


def test_optional_artifacts_may_stay_absent(tmp_path: Path) -> None:
    optional = ArtifactSpec(path="absent.ndjson", role="nodes", sha256=DIGEST, required=False)
    selected = source(LocalAcquisition(backend="local", root=tmp_path), artifacts=[optional])
    result = resolve_source(runtime(tmp_path), selected, allow_download=False)
    assert result.files == {}
    assert result.paths("nodes", extracted=False) == []


def test_locate_source_names_the_fetch_instruction(tmp_path: Path) -> None:
    selected = source(LocalAcquisition(backend="local", root=tmp_path))
    with pytest.raises(FileNotFoundError, match="ced fetch --source fixture"):
        locate_source(runtime(tmp_path), selected)


def test_large_download_requires_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selected = source(Aria2Acquisition(backend="aria2"))
    prepared = PreparedSource(selected, Aria2Backend(runtime(tmp_path), selected), [], inventory_bytes=1024**3)
    monkeypatch.setattr("ced.acquisition.pipeline.require_cpu_worker", lambda config: None)
    monkeypatch.setattr("ced.acquisition.pipeline.load_sources", lambda path: {"fixture": selected})
    monkeypatch.setattr("ced.acquisition.pipeline.prepare_source", lambda *args, **kwargs: prepared)
    with pytest.raises(RuntimeError, match="--yes"):
        fetch_sources(runtime(tmp_path), ["fixture"], all_enabled=False, assume_yes=False)


class RecordingBackend:
    """Stand-in acquisition backend proving bounded concurrency with a barrier.

    A barrier sized to the expected in-flight count fails loudly (BrokenBarrierError)
    when fewer tasks overlap, so a regression back to serial fetching cannot pass by luck.
    """

    def __init__(self, config: RuntimeConfig, payloads: dict[str, bytes], barrier: threading.Barrier) -> None:
        self.runtime = config
        self.payloads = payloads
        self.barrier = barrier
        self.requests: list[str] = []
        self.lock = threading.Lock()

    def fetch(self, artifact: ArtifactSpec, *, allow_download: bool) -> FetchedArtifact:
        with self.lock:
            self.requests.append(artifact.path)
        self.barrier.wait()
        destination = self.runtime.artifact_path("downloads", "fixture", artifact.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.payloads[artifact.path])
        return FetchedArtifact.from_path(
            SourceConfig(
                name="fixture",
                adapter="kgx_ndjson",
                acquisition=LocalAcquisition(backend="local", root=self.runtime.remote_project),
                provenance=ProvenanceConfig(title="Fixture", license_name="CC0-1.0", provided_by="infores:fixture"),
                artifacts=[artifact],
            ),
            artifact,
            destination,
            f"fixture://{artifact.path}",
        )


def test_parallel_map_keeps_input_order_and_proves_overlap() -> None:
    """Fetch manifests must be deterministic while downloads overlap; a barrier proves the
    overlap actually happens instead of relying on timing, which can be flaky on a busy host."""
    barrier = threading.Barrier(3, timeout=30)

    def worker(index: int) -> int:
        barrier.wait()
        return index * 2

    assert parallel_map(list(range(9)), worker, max_workers=3) == [index * 2 for index in range(9)]
    assert parallel_map([], worker, max_workers=3) == []


def test_parallel_map_reraises_first_failure_in_input_order(caplog: pytest.LogCaptureFixture) -> None:
    """locate_source turns the cold-cache FileNotFoundError into a fetch instruction; that
    contract survives concurrent fetching only if the first failure keeps its type and order."""

    def worker(index: int) -> int:
        if index == 1:
            raise FileNotFoundError("cold cache")
        if index == 2:
            raise ValueError("corrupt")
        return index

    with (
        caplog.at_level(logging.WARNING, logger="ced.acquisition.pipeline"),
        pytest.raises(FileNotFoundError, match="cold cache"),
    ):
        parallel_map(list(range(4)), worker, max_workers=2)
    assert any("corrupt" in record.getMessage() for record in caplog.records)


def named_source(name: str, artifacts: list[ArtifactSpec]) -> SourceConfig:
    return SourceConfig(
        name=name,
        adapter="kgx_ndjson",
        acquisition=LocalAcquisition(backend="local", root=Path("/tmp")),
        provenance=ProvenanceConfig(title="Fixture", license_name="CC0-1.0", provided_by="infores:fixture"),
        artifacts=artifacts,
    )


def test_materialize_source_fetches_artifacts_concurrently(tmp_path: Path) -> None:
    """A Translator release ships four artifacts at once; fetching them one by one wastes the
    link, so materialize_source must overlap artifacts up to download_jobs."""
    config = RuntimeConfig(remote_project=tmp_path, artifact_root=Path(".ced"), disk_reserve_gb=0, download_jobs=3)
    payloads = {f"part-{index}.ndjson": PAYLOAD for index in range(6)}
    artifacts = [ArtifactSpec(path=path, role="nodes", sha256=DIGEST) for path in payloads]
    backend = RecordingBackend(config, payloads, threading.Barrier(3, timeout=30))
    result = materialize_source(
        PreparedSource(named_source("fixture", artifacts), backend, artifacts), allow_download=True
    )
    assert [artifact.path.name for artifact in result.files["nodes"]] == list(payloads)


def test_fetch_sources_overlaps_sources_and_writes_ordered_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The task pool spans sources, not just artifacts: otherwise one large archive (ubergraph
    is ~399 MiB) serializes every smaller source behind it, and manifests must still list
    artifacts in declaration order because completion order is now arbitrary."""
    config = RuntimeConfig(remote_project=tmp_path, artifact_root=Path(".ced"), disk_reserve_gb=0, download_jobs=4)
    backends: dict[str, RecordingBackend] = {}
    sources = {}
    barrier = threading.Barrier(4, timeout=30)
    for name in ("alpha", "beta"):
        artifacts = [
            ArtifactSpec(path=f"{name}-first.ndjson", role="nodes", sha256=DIGEST),
            ArtifactSpec(path=f"{name}-second.ndjson", role="nodes", sha256=DIGEST),
        ]
        sources[name] = named_source(name, artifacts)
        backends[name] = RecordingBackend(config, {artifact.path: PAYLOAD for artifact in artifacts}, barrier)
    monkeypatch.setattr("ced.acquisition.pipeline.require_cpu_worker", lambda runtime: None)
    monkeypatch.setattr("ced.acquisition.pipeline.load_sources", lambda path: sources)
    monkeypatch.setattr("ced.acquisition.pipeline._backend", lambda runtime, source: backends[source.name])
    fetch_sources(config, ["alpha", "beta"], all_enabled=False, assume_yes=False)
    for name, backend in backends.items():
        assert len(backend.requests) == 2
        report = json.loads(config.artifact_path("downloads", name, "fetch-manifest.json").read_text(encoding="utf-8"))
        assert [item["origin"] for item in report["files"]["nodes"]] == [
            f"fixture://{name}-first.ndjson",
            f"fixture://{name}-second.ndjson",
        ]
