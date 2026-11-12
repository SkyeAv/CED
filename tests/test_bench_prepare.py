"""Contract tests for the committed preparation benchmark harness.

The harness is the only mechanical evidence that preparation stays bounded in
memory, so it must not silently rot when writer schemas or `prepare_dataset`
change. These tests are offline and hermetic.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "bench_prepare.py"

requires_wenceslaus = pytest.mark.skipif(
    not socket.gethostname().startswith("wenceslaus"),
    reason="prepare_dataset refuses to run outside the wenceslaus CPU worker",
)


def load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_prepare", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_prepare_script_exists_and_is_documented() -> None:
    """The harness must stay in-repo so benchmark commands are reproducible.

    Why: numbers quoted in a report are worthless if the generator that produced
    them lived only in /tmp on one machine.
    """
    assert SCRIPT.is_file()
    harness = load_harness()
    assert harness.__doc__ is not None
    assert "wenceslaus" in harness.__doc__
    assert callable(harness.write_snapshot)
    assert callable(harness.main)


def test_bench_prepare_snapshot_matches_writer_schemas(tmp_path: Path) -> None:
    """The synthetic snapshot must satisfy the real writer schemas and checksums.

    Why: `prepare_dataset` validates both. Silent schema drift here would make every
    future benchmark fail at load time, or worse, exercise a shape the pipeline never
    sees in production.
    """
    from ced.kg.writer import ATTRIBUTE_SCHEMA, NODE_SCHEMA, STATEMENT_SCHEMA

    harness = load_harness()
    root = tmp_path / "snapshot"
    harness.write_snapshot(root, nodes=64, statements=256, attributes=512, seed=7)

    expected = {
        "nodes": NODE_SCHEMA,
        "statements": STATEMENT_SCHEMA,
        "attributes": ATTRIBUTE_SCHEMA,
    }
    for name, schema in expected.items():
        table = pq.read_table(root / name / "data_0.parquet")
        assert table.schema == schema, name
        assert table.num_rows > 0, name

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"nodes": 64, "statements": 256, "attributes": 512}
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    on_disk = {str(path.relative_to(root)) for path in sorted(root.rglob("*.parquet"))}
    assert set(checksums) == on_disk
    assert all(len(digest) == 64 for digest in checksums.values())


@requires_wenceslaus
def test_bench_prepare_reports_wall_time_and_peak_rss(tmp_path: Path) -> None:
    """An end-to-end harness run must report positive wall time, peak RSS and bytes.

    Why: the acceptance evidence for bounded-memory preparation is exactly these three
    numbers. If the child crashes, silently skips a stage, or stops printing the
    `CED_BENCH_RESULT` line, the benchmark would report success with no measurement.
    """
    root = tmp_path / "bench"
    environment = os.environ | {"PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONUTF8": "1"}
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--statements",
            "2000",
            "--nodes",
            "500",
            "--attributes",
            "6000",
            "--disk-reserve-gb",
            "0.01",
            "--label",
            "smoke",
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]

    child_line = next(line for line in completed.stdout.splitlines() if line.startswith("CED_BENCH_RESULT "))
    child = json.loads(child_line.removeprefix("CED_BENCH_RESULT "))
    assert child["seconds"] > 0.0
    assert child["peak_rss_kb"] > 0
    assert child["output_bytes"] > 0

    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    assert summary["label"] == "smoke"
    assert summary["statements"] == 2000
    assert summary["parent_wall_seconds"] >= child["seconds"]
    assert summary["children_peak_rss_gb"] > 0.0

    datasets = list((root / "datasets").iterdir())
    assert len(datasets) == 1
    suffixes = {path.suffix for path in datasets[0].rglob("*") if path.is_file() and path.name != "manifest.json"}
    assert suffixes <= {".parquet", ".json", ".safetensors"}
