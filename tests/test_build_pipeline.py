from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb
import pytest

from ced.acquisition.pipeline import SourceArtifacts
from ced.config.models import RuntimeConfig
from ced.config.sources import AdapterMapping, ArtifactSpec, LocalAcquisition, ProvenanceConfig, SourceConfig
from ced.kg import pipeline
from ced.kg.pipeline import bounded_source, divide_memory, memory_bytes

SOURCES: dict[str, dict[str, list[dict[str, Any]]]] = {
    "alpha": {
        "nodes": [
            {"id": "X:1", "name": "One", "category": ["biolink:Gene"]},
            {"id": "X:2", "name": "Two", "category": ["biolink:Disease"]},
        ],
        "edges": [{"id": "e1", "subject": "X:1", "predicate": "biolink:related_to", "object": "X:2"}],
    },
    "beta": {
        "nodes": [
            {"id": "X:1", "name": "Uno", "category": ["biolink:Gene"]},
            {"id": "Y:3", "name": "Three", "category": ["biolink:Drug"]},
        ],
        "edges": [{"id": "e2", "subject": "X:1", "predicate": "biolink:related_to", "object": "Y:3"}],
    },
}


def kgx_source(
    name: str, root: Path, *, adapter: str = "kgx_ndjson", options: Mapping[str, Any] | None = None
) -> SourceConfig:
    return SourceConfig(
        name=name,
        adapter=adapter,
        artifacts=[ArtifactSpec(path="nodes.ndjson", role="nodes"), ArtifactSpec(path="edges.ndjson", role="edges")],
        acquisition=LocalAcquisition(backend="local", root=root),
        provenance=ProvenanceConfig(title=name, license_name="CC0", provided_by=f"infores:{name}"),
        mapping=AdapterMapping(options=dict(options or {})),
    )


def write_source(root: Path, name: str, records: Mapping[str, list[dict[str, Any]]]) -> SourceArtifacts:
    members: dict[str, list[Path]] = {}
    for role, rows in records.items():
        path = root / name / f"{role}.ndjson"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        members[role] = [path]
    return SourceArtifacts(source=kgx_source(name, root), files={}, members=members)


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RuntimeConfig:
    monkeypatch.setattr(pipeline, "require_cpu_worker", lambda config: None)
    monkeypatch.setattr(pipeline, "require_disk", lambda path, required, reserve: None)
    return RuntimeConfig(remote_project=tmp_path, artifact_root=Path("artifacts"), disk_reserve_gb=0)


def build(
    monkeypatch: pytest.MonkeyPatch,
    runtime: RuntimeConfig,
    located: Mapping[str, SourceArtifacts],
    names: list[str],
    *,
    workers: int,
) -> Path:
    """Run a real build over fixture artifacts, bypassing source discovery and acquisition."""
    ordered = [located[name] for name in names]
    monkeypatch.setattr(pipeline, "load_sources", lambda directory: {})
    monkeypatch.setattr(pipeline, "select_sources", lambda *args, **kwargs: [item.source for item in ordered])
    monkeypatch.setattr(pipeline, "locate_source", lambda config, source: located[source.name])
    return pipeline.build_graph(runtime.model_copy(update={"cpu_workers": workers}), names)


def read_table(snapshot: Path, name: str) -> list[tuple[Any, ...]]:
    connection = duckdb.connect()
    try:
        return connection.execute(f"SELECT * FROM read_parquet('{snapshot / name}/*.parquet')").fetchall()
    finally:
        connection.close()


def test_memory_limits_are_parsed_and_divided() -> None:
    assert memory_bytes("16GB") == 16_000_000_000
    assert memory_bytes("4096MB") == 4_096_000_000
    assert memory_bytes("1GiB") == 1_073_741_824
    assert memory_bytes("2048") == 2048
    for invalid in ("lots", "", "GB", "16 parsec"):
        with pytest.raises(ValueError, match="memory"):
            memory_bytes(invalid)
    assert divide_memory("16GB", 1) == "16000MB"
    assert divide_memory("16GB", 4) == "4000MB"
    assert divide_memory("100MB", 4) == "256MB"
    assert pipeline.duckdb_memory("16GB") == "12000MB"
    assert pipeline.duckdb_memory("16GiB") == "12884MB"
    assert pipeline.duckdb_memory("128MB") == "96MB"
    with pytest.raises(ValueError, match="positive"):
        pipeline.duckdb_memory("0GB")
    with pytest.raises(ValueError, match="positive"):
        divide_memory("16GB", 0)


def test_build_reserves_memory_for_process_overhead(
    tmp_path: Path, runtime: RuntimeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DuckDB can allocate outside its buffer manager, so the total process budget needs headroom."""
    located = {name: write_source(tmp_path / "downloads", name, records) for name, records in SOURCES.items()}
    observed: list[tuple[str, str]] = []
    real_merge = pipeline.merge_graph
    real_reports = pipeline.write_reports

    def record_merge(*args: Any, **kwargs: Any) -> dict[str, int]:
        observed.append(("merge", kwargs["memory"]))
        return real_merge(*args, **kwargs)

    def record_reports(*args: Any, **kwargs: Any) -> None:
        observed.append(("reports", kwargs["memory"]))
        real_reports(*args, **kwargs)

    monkeypatch.setattr(pipeline, "merge_graph", record_merge)
    monkeypatch.setattr(pipeline, "write_reports", record_reports)
    build(monkeypatch, runtime, located, ["alpha"], workers=1)
    assert observed == [("merge", "12000MB"), ("reports", "12000MB")]


def test_only_the_duckdb_adapter_gets_a_worker_budget(tmp_path: Path) -> None:
    """The pool multiplies per-process DuckDB budgets, so each worker is capped; sources without
    a DuckDB connection must be passed through untouched."""
    plain = kgx_source("plain", tmp_path)
    assert bounded_source(plain, threads=1, memory="4000MB") is plain

    amalgam = kgx_source("amalgam", tmp_path, adapter="amalgam_compact", options={"threads": 4, "memory_limit": "16GB"})
    bounded = bounded_source(amalgam, threads=1, memory="4000MB")
    assert bounded.mapping.options["threads"] == 1
    assert bounded.mapping.options["memory_limit"] == "4000MB"
    assert amalgam.mapping.options["threads"] == 4

    modest = kgx_source("modest", tmp_path, adapter="amalgam_compact", options={"threads": 1, "memory_limit": "1GB"})
    kept = bounded_source(modest, threads=4, memory="16000MB")
    assert kept.mapping.options["threads"] == 1
    assert kept.mapping.options["memory_limit"] == "1000MB"


def test_parallel_and_serial_builds_agree(
    tmp_path: Path, runtime: RuntimeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both worker counts must produce the same merged graph, one raw partition per source."""
    located = {name: write_source(tmp_path / "downloads", name, records) for name, records in SOURCES.items()}
    parallel = build(monkeypatch, runtime, located, ["alpha", "beta"], workers=2)
    serial = build(
        monkeypatch,
        runtime.model_copy(update={"remote_project": tmp_path / "serial"}),
        located,
        ["alpha", "beta"],
        workers=1,
    )

    for snapshot in (parallel, serial):
        assert snapshot.is_dir()
        assert not snapshot.with_name(snapshot.name + ".partial").exists()
        assert sorted(path.name for path in (snapshot / "raw").iterdir()) == ["_ontology", "alpha", "beta"]
        assert sorted(path.name for path in (snapshot / "raw/alpha").iterdir()) == [
            "attributes",
            "conflicts",
            "node_identities",
            "nodes",
            "ontology_terms",
            "predicates",
            "statements",
        ]

    counts = json.loads((parallel / "manifest.json").read_text(encoding="utf-8"))["counts"]
    assert counts == json.loads((serial / "manifest.json").read_text(encoding="utf-8"))["counts"]
    assert counts["nodes"] == 3
    manifest = json.loads((parallel / "manifest.json").read_text(encoding="utf-8"))
    assert {source["source_id"] for source in manifest["sources"]} == {"alpha", "beta"}

    nodes = read_table(parallel, "nodes")
    assert sorted(row[0] for row in nodes) == ["X:1", "X:2", "Y:3"]
    shared = next(row for row in nodes if row[0] == "X:1")
    assert sorted(shared[1]) == ["One", "Uno"]
    assert sorted(shared[4]) == ["infores:alpha", "infores:beta"]
    assert sorted({row[1] for row in read_table(parallel, "statements")}) == ["alpha", "beta"]


def test_a_failing_source_names_itself_and_promotes_nothing(
    tmp_path: Path, runtime: RuntimeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One broken source must fail loudly with its own name and leave no promoted snapshot."""
    located = {name: write_source(tmp_path / "downloads", name, records) for name, records in SOURCES.items()}
    broken = tmp_path / "downloads/broken"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "nodes.ndjson").write_text("[not an object]\n", encoding="utf-8")
    (broken / "edges.ndjson").write_text("", encoding="utf-8")
    located["broken"] = SourceArtifacts(
        source=kgx_source("broken", broken),
        files={},
        members={"nodes": [broken / "nodes.ndjson"], "edges": [broken / "edges.ndjson"]},
    )

    with pytest.raises(RuntimeError, match="canonicalization failed for source broken"):
        build(monkeypatch, runtime, located, ["alpha", "broken"], workers=2)
    promoted = [path for path in runtime.artifact_path("snapshots").glob("*") if not path.name.endswith(".partial")]
    assert promoted == []
