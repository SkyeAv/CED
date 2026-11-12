"""Shared helpers for perf-marked benchmarks: synthetic graphs, timing, JSON records.

These benchmarks are the before/after evidence for the performance stories in
.ralph/prd.json. They are excluded from the default suite (marker `perf`) and
must be run on wenceslaus so timings are comparable across stories.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def synthetic_graph(
    *,
    nodes: int,
    edges: int,
    relations: int,
    type_sets: int,
    seed: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> dict[str, np.ndarray]:
    """Deterministic random graph with the same array shapes the pipeline uses.

    Splits follow a seeded uniform draw (not the production hash split) because
    only the shape and composition of the workload matter for benchmarking.
    """
    random = np.random.default_rng(seed)
    heads = random.integers(0, nodes, size=edges)
    tails = random.integers(0, nodes, size=edges)
    edge_index = np.stack((heads, tails)).astype(np.int64)
    relation_id = random.integers(0, relations, size=edges).astype(np.int64)
    draw = random.random(edges)
    split = np.where(draw < train_fraction, 0, np.where(draw < train_fraction + validation_fraction, 1, 2))
    type_set_id = random.integers(0, type_sets, size=nodes).astype(np.int64)
    edge_type = (type_set_id[heads] * relations + relation_id) * type_sets + type_set_id[tails]
    _, edge_type_id = np.unique(edge_type, return_inverse=True)
    return {
        "edge_index": edge_index,
        "relation_id": relation_id,
        "split": split.astype(np.int8),
        "type_set_id": type_set_id,
        "edge_type_id": edge_type_id.astype(np.int64),
    }


def timed(fn: Callable[[], Any], *, iterations: int, warmup: int = 1) -> dict[str, float]:
    """Run fn warmup+iterations times, return per-call seconds (min/mean/total)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {
        "min": min(samples),
        "mean": sum(samples) / len(samples),
        "max": max(samples),
        "total": sum(samples),
    }


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_record(label: str, *, seconds: Mapping[str, float], iterations: int, workload: Mapping[str, Any]) -> Path:
    """Append one JSON benchmark record under .ced/perf/ (gitignored scratch)."""
    directory = REPO_ROOT / ".ced" / "perf"
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "label": label,
        "commit": _git_commit(),
        "host": platform.node(),
        "cpu_workers": os.environ.get("CED_CPU_WORKERS", "unset"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iterations": iterations,
        "workload": dict(workload),
        "seconds": dict(seconds),
    }
    path = directory / f"{label}.json"
    existing = []
    if path.is_file():
        existing = json.loads(path.read_text())
    existing.append(record)
    path.write_text(json.dumps(existing, indent=2))
    return path
