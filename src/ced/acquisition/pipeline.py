from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ced.acquisition.archive import extract_archive
from ced.acquisition.aria2 import Aria2Backend
from ced.acquisition.base import AcquisitionBackend, FetchedArtifact, atomic_write_json
from ced.acquisition.http import HttpBackend
from ced.acquisition.huggingface import HuggingFaceBackend
from ced.acquisition.local import LocalBackend
from ced.config.models import RuntimeConfig
from ced.config.sources import ArtifactSpec, LocalAcquisition, SourceConfig, load_sources, select_sources
from ced.preflight import require_cpu_worker, require_disk

LOGGER = logging.getLogger(__name__)
_LARGE_DOWNLOAD = 1024**3
Materialized = tuple[FetchedArtifact, dict[str, list[Path]]]


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


def parallel_map[T, R](items: Sequence[T], worker: Callable[[T], R], *, max_workers: int) -> list[R]:
    """Run ``worker`` over ``items`` with at most ``max_workers`` tasks in flight.

    Acquisition work is network transfer, SHA-256 hashing and zstd extraction, all of which
    release the GIL, so a bounded thread pool gives real parallelism while RuntimeConfig
    keeps the CPU ceiling explicit. Results keep input order regardless of completion order,
    which is what makes fetch manifests and archive member lists deterministic. The first
    failure in input order is re-raised unchanged so callers still match on its type
    (notably the cold-cache FileNotFoundError that locate_source turns into a fetch
    instruction); any further failure is logged, never swallowed.
    """
    if not items:
        return []
    workers = max(1, min(max_workers, len(items)))
    if workers == 1:
        return [worker(item) for item in items]
    collected: dict[int, R] = {}
    failures: list[tuple[int, Exception]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ced-acquire") as pool:
        futures = {pool.submit(worker, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            index = futures[future]
            try:
                collected[index] = future.result()
            except Exception as error:
                failures.append((index, error))
                for pending in futures:
                    pending.cancel()
    if failures:
        failures.sort(key=lambda failure: failure[0])
        for index, error in failures[1:]:
            LOGGER.warning("additional acquisition failure for task %d: %s", index, error)
        raise failures[0][1]
    return [collected[index] for index in range(len(items))]


def _backend(runtime: RuntimeConfig, source: SourceConfig) -> AcquisitionBackend:
    if source.acquisition.backend == "huggingface":
        return HuggingFaceBackend(runtime, source)
    if source.acquisition.backend == "aria2":
        return Aria2Backend(runtime, source)
    if source.acquisition.backend == "http":
        return HttpBackend(runtime, source)
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


def _materialize_artifact(
    prepared: PreparedSource, artifact: ArtifactSpec, *, allow_download: bool
) -> Materialized | None:
    """Fetch one artifact and expand it when it is an archive; None means an absent optional one."""
    try:
        fetched = prepared.backend.fetch(artifact, allow_download=allow_download)
    except FileNotFoundError:
        if artifact.required:
            raise
        return None
    members: dict[str, list[Path]] = {}
    if artifact.archive:
        destination = prepared.backend.runtime.artifact_path(
            "downloads", prepared.source.name, "extracted", fetched.sha256
        )
        members = extract_archive(fetched.path, artifact.archive, destination)
    return fetched, members


def _assemble(prepared: PreparedSource, outcomes: Sequence[Materialized | None]) -> SourceArtifacts:
    files: dict[str, list[FetchedArtifact]] = {}
    members: dict[str, list[Path]] = {}
    for outcome in outcomes:
        if outcome is None:
            continue
        fetched, extracted = outcome
        files.setdefault(fetched.role, []).append(fetched)
        for role, paths in extracted.items():
            members.setdefault(role, []).extend(paths)
    return SourceArtifacts(source=prepared.source, files=files, members=members)


def materialize_source(prepared: PreparedSource, *, allow_download: bool) -> SourceArtifacts:
    outcomes = parallel_map(
        prepared.artifacts,
        lambda artifact: _materialize_artifact(prepared, artifact, allow_download=allow_download),
        max_workers=prepared.backend.runtime.download_jobs,
    )
    return _assemble(prepared, outcomes)


def resolve_source(runtime: RuntimeConfig, source: SourceConfig, *, allow_download: bool = False) -> SourceArtifacts:
    return materialize_source(
        prepare_source(runtime, source, allow_download=allow_download), allow_download=allow_download
    )


def fetch_sources(
    runtime: RuntimeConfig,
    names: list[str],
    *,
    groups: list[str] | None = None,
    all_enabled: bool = False,
    assume_yes: bool = False,
) -> None:
    require_cpu_worker(runtime)
    sources = select_sources(
        load_sources(runtime.source_directory),
        names,
        all_enabled=all_enabled,
        groups=groups,
        require_explicit=True,
    )
    prepared = [prepare_source(runtime, source, allow_download=True) for source in sources]
    remote = [item for item in prepared if not isinstance(item.source.acquisition, LocalAcquisition)]
    required = sum(item.inventory_bytes + sum(artifact.size or 0 for artifact in item.artifacts) for item in remote)
    if required >= _LARGE_DOWNLOAD and not assume_yes:
        raise RuntimeError(
            f"download requires {required} bytes; rerun with --yes after reviewing disk and license requirements"
        )
    require_disk(runtime.artifact_path(), required, runtime.disk_reserve_gb)
    tasks = [(item, artifact) for item in prepared for artifact in item.artifacts]
    outcomes = parallel_map(
        tasks,
        lambda task: _materialize_artifact(task[0], task[1], allow_download=True),
        max_workers=runtime.download_jobs,
    )
    cursor = 0
    for item in prepared:
        result = _assemble(item, outcomes[cursor : cursor + len(item.artifacts)])
        cursor += len(item.artifacts)
        report = result.model_dump(mode="json", exclude={"source"})
        atomic_write_json(runtime.artifact_path("downloads", item.source.name, "fetch-manifest.json"), report)
        LOGGER.info("fetched %s: %d files", item.source.name, sum(map(len, result.files.values())))


def locate_source(runtime: RuntimeConfig, source: SourceConfig) -> SourceArtifacts:
    try:
        return resolve_source(runtime, source, allow_download=False)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"source {source.name} is not cached; run: ced fetch --source {source.name}") from error
