from __future__ import annotations

import hashlib
import logging
import multiprocessing
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ced.acquisition.base import FetchedArtifact, atomic_write_json, sha256_file
from ced.acquisition.pipeline import SourceArtifacts, locate_source
from ced.config.models import RuntimeConfig
from ced.config.sources import SourceConfig, load_sources, select_sources
from ced.kg.adapters.amalgam_compact import AmalgamCompactAdapter
from ced.kg.adapters.kgx_ndjson import KgxNdjsonAdapter
from ced.kg.adapters.tablassert import TablassertAdapter
from ced.kg.audit import write_reports
from ced.kg.merge import merge_graph
from ced.kg.ontology import amalgam_ontology, biolink_ontology
from ced.kg.records import SnapshotManifest, SourceLineage, canonical_json
from ced.kg.writer import CanonicalBatchWriter
from ced.preflight import require_cpu_worker, require_disk

LOGGER = logging.getLogger(__name__)
ONTOLOGY_PARTITION = "_ontology"
MIN_WORKER_MEMORY_MB = 256
UNITS = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}
MEMORY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)([A-Za-z]*)")


def memory_bytes(value: str) -> int:
    match = MEMORY_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"unparseable memory limit: {value!r}")
    unit = (match.group(2) or "B").upper()
    if unit not in UNITS:
        raise ValueError(f"unknown memory unit in limit: {value!r}")
    return int(float(match.group(1)) * UNITS[unit])


def divide_memory(value: str, parts: int) -> str:
    """Split one memory ceiling across worker processes, in megabytes DuckDB accepts."""
    if parts < 1:
        raise ValueError(f"parts must be positive, got {parts}")
    return f"{max(MIN_WORKER_MEMORY_MB, memory_bytes(value) // parts // 1000**2)}MB"


@dataclass(frozen=True)
class CanonicalizationJob:
    located: SourceArtifacts
    raw_root: Path
    spill_root: Path
    threads: int
    memory: str


def bounded_source(source: SourceConfig, *, threads: int, memory: str) -> SourceConfig:
    """Cap a worker's own DuckDB budget so the whole pool stays inside the configured ceiling."""
    if source.adapter != "amalgam_compact":
        return source
    options = dict(source.mapping.options)
    options["threads"] = max(1, min(int(options.get("threads", threads)), threads))
    declared = memory_bytes(str(options.get("memory_limit", memory)))
    options["memory_limit"] = f"{max(MIN_WORKER_MEMORY_MB, min(declared, memory_bytes(memory)) // 1000**2)}MB"
    return source.model_copy(update={"mapping": source.mapping.model_copy(update={"options": options})})


def canonicalize_source(job: CanonicalizationJob) -> dict[str, int]:
    """Canonicalize one source into `raw/<source>/`. Runs in a worker process."""
    source = job.located.source
    writer = CanonicalBatchWriter(job.raw_root / source.name)
    try:
        adapter: KgxNdjsonAdapter | AmalgamCompactAdapter
        if source.adapter == "amalgam_compact":
            for metadata in job.located.paths("metadata"):
                if metadata.name == "ontology.yaml":
                    writer.write(amalgam_ontology(metadata))
            adapter = AmalgamCompactAdapter(job.located, job.spill_root)
        else:
            adapter = KgxNdjsonAdapter(job.located)
        try:
            for batch in adapter.batches():
                writer.write(batch)
        finally:
            if isinstance(adapter, AmalgamCompactAdapter):
                adapter.close()
        return writer.close()
    except BaseException as error:
        discard(writer)
        raise RuntimeError(f"canonicalization failed for source {source.name}: {error}") from error


def discard(writer: CanonicalBatchWriter) -> None:
    """Release a failed worker's writers; the staging partition is discarded by the caller."""
    try:
        writer.close()
    except Exception as error:
        LOGGER.warning("could not close writers after a canonicalization failure: %s", error)


def canonicalize_sources(jobs: list[CanonicalizationJob], *, workers: int) -> None:
    """Run every canonicalization job, in parallel processes when more than one worker is allowed."""
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            canonicalize_source(job)
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for job, counts in zip(jobs, pool.map(canonicalize_source, jobs), strict=True):
            LOGGER.info("canonicalized %s: %s", job.located.source.name, counts)


def build_graph(
    runtime: RuntimeConfig, names: list[str], *, groups: list[str] | None = None, fullmap_path: Path | None = None
) -> Path:
    require_cpu_worker(runtime)
    require_disk(runtime.artifact_path(), 0, runtime.disk_reserve_gb)
    selected = select_sources(
        load_sources(runtime.source_directory), names, all_enabled=not names and not groups, groups=groups
    )
    supported = {"tablassert", "kgx_ndjson", "amalgam_compact"}
    for source in selected:
        if source.adapter not in supported:
            raise ValueError(f"unsupported adapter: {source.adapter}")
    lineages = []
    artifacts = []
    for source in selected:
        if source.adapter == "tablassert":
            built = TablassertAdapter(runtime, source, fullmap_path).build()
            files = {}
            for role, path in (("nodes", built.nodes), ("edges", built.edges)):
                files[role] = [
                    FetchedArtifact(
                        source=source.name,
                        role=role,
                        path=path,
                        sha256=sha256_file(path),
                        size=path.stat().st_size,
                        origin="tablassert",
                    )
                ]
            located = SourceArtifacts(source=source, files=files, members={})
        else:
            located = locate_source(runtime, source)
        artifacts.append(located)
        lineages.append(
            SourceLineage(
                source_id=source.name,
                revision=str(getattr(source.acquisition, "revision", "local")),
                license_name=source.provenance.license_name,
                provided_by=source.provenance.provided_by,
                artifacts={str(item.path): item.sha256 for group in located.files.values() for item in group},
            )
        )
    parameters = {"sources": [source.model_dump(mode="json") for source in selected], "adapter_version": "3"}
    identity = canonical_json(
        {"lineages": [item.model_dump(mode="json") for item in lineages], "parameters": parameters}
    )
    snapshot_hash = hashlib.sha256(identity.encode()).hexdigest()
    output = runtime.artifact_path("snapshots", snapshot_hash)
    if output.exists():
        raise FileExistsError(f"snapshot already exists: {output}")
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        raise RuntimeError(f"incomplete snapshot requires explicit repair: {staging}")
    required = sum(item.size for located in artifacts for group in located.files.values() for item in group) * 6
    require_disk(runtime.artifact_path(), required, runtime.disk_reserve_gb)
    ontology = CanonicalBatchWriter(staging / "raw" / ONTOLOGY_PARTITION)
    try:
        ontology.write(biolink_ontology())
    finally:
        ontology.close()
    workers = max(1, min(runtime.cpu_workers, len(artifacts)))
    threads = max(1, runtime.cpu_workers // workers)
    memory = divide_memory(runtime.memory_limit, workers)
    canonicalize_sources(
        [
            CanonicalizationJob(
                located=located.model_copy(
                    update={"source": bounded_source(located.source, threads=threads, memory=memory)}
                ),
                raw_root=staging / "raw",
                spill_root=staging / "spill",
                threads=threads,
                memory=memory,
            )
            for located in artifacts
        ],
        workers=workers,
    )
    counts = merge_graph(staging, threads=runtime.cpu_workers, memory=runtime.memory_limit)
    manifest = SnapshotManifest(snapshot_hash=snapshot_hash, sources=lineages, counts=counts, parameters=parameters)
    atomic_write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
    atomic_write_json(staging / "license-report.json", [item.model_dump(mode="json") for item in lineages])
    atomic_write_json(staging / "build-report.json", {"counts": counts})
    write_reports(staging, threads=runtime.cpu_workers, memory=runtime.memory_limit)
    shutil.rmtree(staging / "spill", ignore_errors=True)
    checksums = {
        str(path.relative_to(staging)): sha256_file(path) for path in sorted(staging.rglob("*")) if path.is_file()
    }
    atomic_write_json(staging / "checksums.json", checksums)
    staging.rename(output)
    return output
