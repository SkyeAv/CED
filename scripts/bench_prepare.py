"""Benchmark CED dataset preparation serialization on wenceslaus.

Generates a synthetic canonical snapshot with the real writer schemas, then runs
``prepare_dataset`` in a subprocess and reports wall seconds plus peak RSS. The
child's own ``RUSAGE_SELF`` maximum is printed as ``CED_BENCH_RESULT``; the parent
prints one JSON summary line using ``RUSAGE_CHILDREN``.

Shard sampling is excluded by default because it is O(chunks x nodes x edges) and
identical across serialization strategies; including it would hide the change under
minutes of sampling. Pass ``--with-shards`` on a small workload to measure the whole
call.

Offline and hermetic: no downloads, no fullmap, no Modal. The child sets
``CED_REMOTE_WORKER`` exactly as the ``ced`` remote path does, but
``require_cpu_worker`` still refuses any host whose name does not start with
``wenceslaus``, so this script cannot be used to run heavy work on a laptop.

Usage (from the repository root on wenceslaus):

    uv run python scripts/bench_prepare.py --statements 50000 --nodes 20000 \
        --attributes 150000 --label parquet-50k-150k

Compare arms by checking out the revision under test and reusing one ``--root``;
snapshots are cached per (statements, attributes) and ``--skip-generate`` reuses them.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CHILD_ENV_MARK = "CED_BENCH_CHILD"
LONG_TEXT = "bench attribute text payload " * 8


def _strings(values: np.ndarray) -> pa.Array:
    return pa.array(values, type=pa.string())


def _empty_lists(rows: int) -> pa.Array:
    return pa.array([[]] * rows, type=pa.list_(pa.string()))


def _sha256(path: Path) -> str:
    from ced.acquisition.base import sha256_file

    return sha256_file(path)


def write_snapshot(root: Path, *, nodes: int, statements: int, attributes: int, seed: int) -> None:
    """Emit a synthetic canonical snapshot that prepare_dataset accepts unchanged."""
    rng = np.random.default_rng(seed)
    if root.exists():
        shutil.rmtree(root)
    for name in ("nodes", "statements", "attributes"):
        (root / name).mkdir(parents=True)

    node_ids = np.array([f"BENCH:N{i}" for i in range(nodes)])
    pq.write_table(
        pa.table(
            {
                "id": _strings(node_ids),
                "names": _empty_lists(nodes),
                "categories": pa.array([f"bench:c{i % 16}" for i in range(nodes)], pa.list_(pa.string())),
                "in_taxon": _empty_lists(nodes),
                "provided_by": _empty_lists(nodes),
                "source_ids": _empty_lists(nodes),
                "mapping_status": _strings(np.full(nodes, "mapped")),
                "original_label": _strings(np.full(nodes, "")),
                "native_ids_json": _strings(np.full(nodes, "{}")),
                "attributes_json": _strings(np.full(nodes, "{}")),
            }
        ),
        root / "nodes" / "data_0.parquet",
        compression="zstd",
    )

    subject = node_ids[rng.integers(0, nodes, statements)]
    tail = node_ids[rng.integers(0, nodes, statements)]
    roles = np.where(rng.integers(0, 100, statements) < 2, "schema", "asserted")
    predicates = np.char.mod("bench:p%d", np.arange(statements) % 32)
    statement_ids = np.char.mod("BENCH:S%d", np.arange(statements))
    pq.write_table(
        pa.table(
            {
                "id": _strings(statement_ids),
                "source_id": _strings(np.full(statements, "bench")),
                "native_id": _strings(np.char.mod("S%d", np.arange(statements))),
                "subject": _strings(subject),
                "predicate": _strings(predicates),
                "object": _strings(tail),
                "relation_role": _strings(roles),
                "target_eligible": pa.array(np.ones(statements, dtype=bool)),
                "categories": _empty_lists(statements),
                "qualifiers_json": _strings(np.full(statements, "{}")),
                "attributes_json": _strings(np.full(statements, "{}")),
                "publications": _empty_lists(statements),
                "sources_json": _strings(np.full(statements, "[]")),
                "original_record_json": _strings(np.full(statements, "{}")),
            }
        ),
        root / "statements" / "data_0.parquet",
        compression="zstd",
    )

    is_node = rng.integers(0, 2, attributes) == 0
    owner_id = np.where(
        is_node,
        node_ids[rng.integers(0, nodes, attributes)],
        statement_ids[rng.integers(0, statements, attributes)],
    )
    path_index = rng.integers(0, 8, attributes)
    kind_roll = rng.integers(0, 10, attributes)
    is_numeric = kind_roll < 4
    is_text = (kind_roll >= 4) & (kind_roll < 7)
    numbers = rng.normal(0.0, 5.0, attributes)
    raw_json = np.where(is_numeric, np.char.mod("%.6f", numbers), np.char.mod('"cat%d"', path_index))
    pq.write_table(
        pa.table(
            {
                "owner_kind": _strings(np.where(is_node, "node", "statement")),
                "owner_id": _strings(owner_id),
                "source_id": _strings(np.full(attributes, "bench")),
                "path": _strings(np.char.mod("attr%d", path_index)),
                "value_type": _strings(np.where(is_numeric, "numeric", "string")),
                "ordinal": pa.array(np.zeros(attributes, dtype=np.int32)),
                "raw_json": _strings(raw_json),
                "text": pa.array(np.full(attributes, LONG_TEXT, dtype=object), type=pa.string(), mask=~is_text),
                "numeric": pa.array(numbers, type=pa.float64(), mask=~is_numeric),
                "provenance_json": _strings(np.full(attributes, "{}")),
            }
        ),
        root / "attributes" / "data_0.parquet",
        compression="zstd",
    )

    from ced.kg.writer import ATTRIBUTE_SCHEMA, NODE_SCHEMA, STATEMENT_SCHEMA

    for name, schema in (
        ("nodes", NODE_SCHEMA),
        ("statements", STATEMENT_SCHEMA),
        ("attributes", ATTRIBUTE_SCHEMA),
    ):
        path = root / name / "data_0.parquet"
        table = pq.read_table(path)
        if table.schema != schema:
            raise ValueError(f"snapshot schema drift for {name}: {table.schema}")

    counts = {"nodes": nodes, "statements": statements, "attributes": attributes}
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "snapshot_hash": "bench", "counts": counts}, sort_keys=True),
        encoding="utf-8",
    )
    checksums = {str(path.relative_to(root)): _sha256(path) for path in sorted(root.rglob("*.parquet"))}
    (root / "checksums.json").write_text(json.dumps(checksums, sort_keys=True), encoding="utf-8")


def run_child(payload: str) -> None:
    spec: dict[str, Any] = json.loads(payload)
    os.environ["CED_REMOTE_WORKER"] = "1"
    from ced.config.models import RuntimeConfig
    from ced.model import prepare as prepare_module

    if not spec["with_shards"]:
        prepare_module.write_batch_shards = lambda root, arrays, runtime, **kwargs: []
    runtime = RuntimeConfig.model_validate(spec["runtime"])
    started = time.perf_counter()
    target = prepare_module.prepare_dataset(runtime, spec["snapshot"])
    seconds = time.perf_counter() - started
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    total = sum(path.stat().st_size for path in Path(target).rglob("*") if path.is_file())
    print("CED_BENCH_RESULT " + json.dumps({"seconds": seconds, "peak_rss_kb": peak_kb, "output_bytes": total}))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CED dataset preparation serialization on wenceslaus.")
    parser.add_argument("--root", default=".ced/bench")
    parser.add_argument("--nodes", type=int, default=200_000)
    parser.add_argument("--statements", type=int, default=2_000_000)
    parser.add_argument("--attributes", type=int, default=6_000_000)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--disk-reserve-gb", type=float, default=4.0)
    parser.add_argument("--with-shards", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--label", default="run")
    arguments = parser.parse_args()

    if os.environ.get(CHILD_ENV_MARK):
        run_child(os.environ[CHILD_ENV_MARK])
        return 0

    root = Path(arguments.root).resolve()
    snapshot_name = f"bench-{arguments.statements}-{arguments.attributes}"
    snapshot = root / "snapshots" / snapshot_name
    if not arguments.skip_generate:
        started = time.perf_counter()
        write_snapshot(
            snapshot,
            nodes=arguments.nodes,
            statements=arguments.statements,
            attributes=arguments.attributes,
            seed=arguments.seed,
        )
        print(f"generated snapshot in {time.perf_counter() - started:.1f}s at {snapshot}", flush=True)

    datasets = root / "datasets"
    shutil.rmtree(datasets, ignore_errors=True)
    datasets.mkdir(parents=True)

    payload = {
        "snapshot": snapshot_name,
        "with_shards": arguments.with_shards,
        "runtime": {
            "remote_project": str(root.parent.parent),
            "artifact_root": str(root),
            "cpu_workers": 4,
            "memory_limit": "16GB",
            "disk_reserve_gb": arguments.disk_reserve_gb,
        },
    }
    environment = os.environ | {CHILD_ENV_MARK: json.dumps(payload), "CED_REMOTE_WORKER": "1"}
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    started = time.perf_counter()
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve())], env=environment, check=False)
    wall = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if completed.returncode != 0:
        print(f"child failed with {completed.returncode}")
        return completed.returncode
    print(
        json.dumps(
            {
                "label": arguments.label,
                "statements": arguments.statements,
                "nodes": arguments.nodes,
                "attributes": arguments.attributes,
                "with_shards": arguments.with_shards,
                "parent_wall_seconds": round(wall, 2),
                "children_peak_rss_gb": round(max(after, before) / 1024**2, 3),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
