from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ced.acquisition.archive import extract_archive
from ced.acquisition.aria2 import Aria2Backend
from ced.acquisition.base import AcquisitionBackend, FetchedArtifact, atomic_write_json
from ced.acquisition.huggingface import HuggingFaceBackend
from ced.acquisition.local import LocalBackend
from ced.config.models import RuntimeConfig
from ced.config.sources import ArtifactSpec, LocalAcquisition, SourceConfig, load_sources, select_sources
from ced.preflight import require_cpu_worker, require_disk

LOGGER = logging.getLogger(__name__)
_LARGE_DOWNLOAD = 1024**3


class SourceArtifacts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: SourceConfig
    files: dict[str, list[FetchedArtifact]]
    members: dict[str, list[Path]]

    def paths(self, role: str, *, extracted: bool = True) -> list[Path]:
        if extracted and role in self.members:
            return self.members[role]
        return [artifact.path for artifact in self.files.get(role, [])]


@dataclass
class PreparedSource:
    source: SourceConfig
    backend: AcquisitionBackend
    artifacts: list[ArtifactSpec]
    inventory_bytes: int = 0


def _backend(runtime: RuntimeConfig, source: SourceConfig) -> AcquisitionBackend:
    if source.acquisition.backend == "huggingface":
        return HuggingFaceBackend(runtime, source)
    if source.acquisition.backend == "aria2":
        return Aria2Backend(runtime, source)
    if source.acquisition.backend == "local":
        return LocalBackend(runtime, source)
    raise ValueError(f"unsupported acquisition backend: {source.acquisition.backend}")


def prepare_source(runtime: RuntimeConfig, source: SourceConfig, *, allow_download: bool) -> PreparedSource:
    backend = _backend(runtime, source)
    if source.inventory is None:
        return PreparedSource(source, backend, source.artifacts)
    manifest_spec = next(artifact for artifact in source.artifacts if artifact.path == source.inventory.manifest)
    manifest_path = backend.fetch(manifest_spec, allow_download=allow_download).path
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = list(source.artifacts)
    artifacts.extend(source.inventory.materialize(document))
    return PreparedSource(source, backend, artifacts, int(document["compact"]["total_bytes"]))


def materialize_source(prepared: PreparedSource, *, allow_download: bool) -> SourceArtifacts:
    files: dict[str, list[FetchedArtifact]] = {}
    members: dict[str, list[Path]] = {}
    for artifact in prepared.artifacts:
        try:
            fetched = prepared.backend.fetch(artifact, allow_download=allow_download)
        except FileNotFoundError:
            if artifact.required:
                raise
            continue
        files.setdefault(artifact.role, []).append(fetched)
        if artifact.archive:
            destination = prepared.backend.runtime.artifact_path(
                "downloads", prepared.source.name, "extracted", fetched.sha256
            )
            extracted = extract_archive(fetched.path, artifact.archive, destination)
            for role, paths in extracted.items():
                members.setdefault(role, []).extend(paths)
    return SourceArtifacts(source=prepared.source, files=files, members=members)


def resolve_source(runtime: RuntimeConfig, source: SourceConfig, *, allow_download: bool = False) -> SourceArtifacts:
    return materialize_source(
        prepare_source(runtime, source, allow_download=allow_download), allow_download=allow_download
    )


def fetch_sources(runtime: RuntimeConfig, names: list[str], *, all_enabled: bool, assume_yes: bool) -> None:
    require_cpu_worker(runtime)
    sources = select_sources(
        load_sources(runtime.source_directory), names, all_enabled=all_enabled, require_explicit=True
    )
    prepared = [prepare_source(runtime, source, allow_download=True) for source in sources]
    remote = [item for item in prepared if not isinstance(item.source.acquisition, LocalAcquisition)]
    required = sum(item.inventory_bytes + sum(artifact.size or 0 for artifact in item.artifacts) for item in remote)
    if required >= _LARGE_DOWNLOAD and not assume_yes:
        raise RuntimeError(
            f"download requires {required} bytes; rerun with --yes after reviewing disk and license requirements"
        )
    require_disk(runtime.artifact_path(), required, runtime.disk_reserve_gb)
    for item in prepared:
        result = materialize_source(item, allow_download=True)
        report = result.model_dump(mode="json", exclude={"source"})
        atomic_write_json(runtime.artifact_path("downloads", item.source.name, "fetch-manifest.json"), report)
        LOGGER.info("fetched %s: %d files", item.source.name, sum(map(len, result.files.values())))


def locate_source(runtime: RuntimeConfig, source: SourceConfig) -> SourceArtifacts:
    try:
        return resolve_source(runtime, source, allow_download=False)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"source {source.name} is not cached; run: ced fetch --source {source.name}") from error
