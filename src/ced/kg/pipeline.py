from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from ced.acquisition.base import FetchedArtifact, atomic_write_json, sha256_file
from ced.acquisition.pipeline import SourceArtifacts, locate_source
from ced.config.models import RuntimeConfig
from ced.config.sources import load_sources, select_sources
from ced.kg.adapters.amalgam_compact import AmalgamCompactAdapter
from ced.kg.adapters.kgx_ndjson import KgxNdjsonAdapter
from ced.kg.adapters.tablassert import TablassertAdapter
from ced.kg.audit import write_reports
from ced.kg.merge import merge_graph
from ced.kg.ontology import amalgam_ontology, biolink_ontology
from ced.kg.records import SnapshotManifest, SourceLineage, canonical_json
from ced.kg.writer import CanonicalBatchWriter
from ced.preflight import require_cpu_worker, require_disk


def build_graph(runtime: RuntimeConfig, names: list[str], *, fullmap_path: Path | None = None) -> Path:
    require_cpu_worker(runtime)
    require_disk(runtime.artifact_path(), 0, runtime.disk_reserve_gb)
    selected = select_sources(load_sources(runtime.source_directory), names, all_enabled=not names)
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
    writer = CanonicalBatchWriter(staging / "raw")
    try:
        writer.write(biolink_ontology())
        for located in artifacts:
            if located.source.adapter == "amalgam_compact":
                for metadata in located.paths("metadata"):
                    if metadata.name == "ontology.yaml":
                        writer.write(amalgam_ontology(metadata))
            adapter = (
                AmalgamCompactAdapter(located, staging / "spill")
                if located.source.adapter == "amalgam_compact"
                else KgxNdjsonAdapter(located)
            )
            try:
                for batch in adapter.batches():
                    writer.write(batch)
            finally:
                if isinstance(adapter, AmalgamCompactAdapter):
                    adapter.close()
    finally:
        writer.close()
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
