from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ced.acquisition.base import atomic_write_json
from ced.config.models import RuntimeConfig
from ced.kg.audit import validate_checksums

SHARED_TRAINING_FILES = (
    "manifest.json",
    "fitted_fields.parquet",
    "fitted_categories.parquet",
)
SHARED_EVALUATION_FILES = (
    "manifest.json",
    "entities.parquet",
    "relations.parquet",
    "types.parquet",
    "fitted_fields.parquet",
    "fitted_categories.parquet",
)


class VolumeClient(Protocol):
    def upload(self, files: Mapping[str, Path]) -> None: ...

    def download(self, remote: str, local: Path) -> None: ...

    def remove(self, remote: str, *, recursive: bool = False) -> None: ...

    def list(self, prefix: str) -> list[str]: ...


class FunctionRunner(Protocol):
    def call(self, function: str, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class BatchGroup:
    indices: list[int]
    files: dict[str, Path]
    bytes: int


@dataclass
class TransferPlan:
    mode: str
    root: Path
    shared: dict[str, Path]
    groups: list[BatchGroup] = field(default_factory=list)
    shared_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.shared_bytes + sum(group.bytes for group in self.groups)


def _tensor_paths(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        if "path" in value and "sha256" in value and isinstance(value["path"], str):
            yield value["path"]
        for item in value.values():
            yield from _tensor_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _tensor_paths(item)


def _retry(settings: RuntimeConfig, action: Callable[[], Any]) -> Any:
    delay = settings.modal.retry_base_seconds
    for attempt in range(settings.modal.retries):
        try:
            return action()
        except Exception:
            if attempt + 1 == settings.modal.retries:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable retry state")


def prepare_run_directory(runtime: RuntimeConfig, dataset: str, mode: str) -> Path:
    root = runtime.artifact_path("datasets", dataset)
    manifest = json.loads((root / "manifest.json").read_text())
    validate_checksums(root, manifest["checksums"])
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    identity = hashlib.sha256(f"{dataset}\0{mode}\0{stamp}\0{runtime.modal.model_dump_json()}".encode()).hexdigest()[
        :16
    ]
    run = runtime.artifact_path("runs", f"{mode}-{dataset[:12]}-{identity}")
    run.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        run / "run-config.json",
        {"dataset": dataset, "mode": mode, "created_utc": stamp, "modal": runtime.modal.model_dump(mode="json")},
    )
    return run


def dataset_transfer_plan(root: Path, runtime: RuntimeConfig, mode: str) -> TransferPlan:
    if mode not in {"train", "evaluate"}:
        raise ValueError("mode must be train or evaluate")
    manifest = json.loads((root / "manifest.json").read_text())
    validate_checksums(root, manifest["checksums"])
    shared_names: set[str] = set(SHARED_TRAINING_FILES if mode == "train" else SHARED_EVALUATION_FILES)
    shared: dict[str, Path] = {}
    if mode == "train":
        for item in _tensor_paths(manifest.get("features", {})):
            shared_names.add(item)
    else:
        for item in _tensor_paths(manifest.get("tensors", {})):
            shared_names.add(item)
        for item in _tensor_paths(manifest.get("features", {})):
            shared_names.add(item)
    for item in _tensor_paths(manifest.get("semantic", {})):
        shared_names.add(item)
    for name in shared_names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required dataset artifact is missing: {name}")
        shared[name] = path
    shared_bytes = sum(path.stat().st_size for path in shared.values())
    if shared_bytes > runtime.modal.max_shared_bytes:
        raise ValueError(
            f"shared transfer size {shared_bytes} exceeds {runtime.modal.max_shared_bytes}; reduce feature payload or rotate it explicitly"
        )
    groups: list[BatchGroup] = []
    if mode == "train":
        current = BatchGroup(indices=[], files={}, bytes=0)
        for index, example in enumerate(manifest["batches"]):
            files = {path: root / path for path in _tensor_paths(example)}
            size = sum(path.stat().st_size for path in files.values())
            if size > runtime.modal.max_batch_group_bytes:
                raise ValueError(
                    f"batch group {index} exceeds the Modal transfer bound; reduce batch size before training"
                )
            if current.files and current.bytes + size > runtime.modal.max_batch_group_bytes:
                groups.append(current)
                current = BatchGroup(indices=[], files={}, bytes=0)
            current.indices.append(index)
            current.files.update(files)
            current.bytes += size
        if current.files:
            groups.append(current)
    return TransferPlan(mode=mode, root=root, shared=shared, groups=groups, shared_bytes=shared_bytes)


def _upload(volume: VolumeClient, runtime: RuntimeConfig, files: Mapping[str, Path], prefix: str) -> None:
    _retry(runtime, lambda: volume.upload({f"{prefix}/{name}": path for name, path in files.items()}))


def download_outputs(
    volume: VolumeClient, runtime: RuntimeConfig, remote_prefix: str, local: Path, names: Iterable[str]
) -> None:
    local.mkdir(parents=True, exist_ok=True)
    for name in names:
        _retry(runtime, lambda name=name: volume.download(f"{remote_prefix}/{name}", local / name))


def execute_plan(
    plan: TransferPlan,
    runtime: RuntimeConfig,
    volume: VolumeClient,
    runner: FunctionRunner,
    run_root: Path,
    *,
    epochs: int = 1,
    seed: int = 20261110,
    payload_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = run_root / "checkpoint.pt"
    progress_path = run_root / "progress.json"
    _upload(volume, runtime, plan.shared, "dataset")
    if checkpoint.is_file():
        files = {"checkpoint.pt": checkpoint}
        for name in ("checkpoint.sha256", "checkpoint.json"):
            path = run_root / name
            if path.is_file():
                files[name] = path
        _retry(runtime, lambda: volume.upload(files))
    metrics: dict[str, Any] = {"mode": plan.mode, "groups": 0}
    cursor = int(json.loads(progress_path.read_text()).get("cursor", 0)) if progress_path.is_file() else 0
    try:
        if plan.mode == "encode":
            remote = "outputs/encode"
            result = _retry(
                runtime,
                lambda: runner.call(
                    "encode_text",
                    {
                        "root": "/scratch/text-encoding",
                        "output_prefix": f"/data/{remote}",
                        **(payload_overrides or {}),
                    },
                ),
            )
            download_outputs(volume, runtime, remote, run_root, result.pop("files"))
            metrics.update(result)
            return metrics
        if plan.mode == "evaluate":
            remote = "outputs/evaluate"
            result = _retry(
                runtime,
                lambda: runner.call(
                    "evaluate",
                    {
                        "root": "/scratch/dataset",
                        "output_prefix": f"/data/{remote}",
                        "seed": seed,
                        "checkpoint_path": "/scratch/checkpoint.pt",
                        **(payload_overrides or {}),
                    },
                ),
            )
            download_outputs(volume, runtime, remote, run_root, ["metrics.json"])
            metrics.update(result)
            return metrics
        total = sum(len(group.indices) for group in plan.groups)
        for number, group in enumerate(plan.groups):
            group_start = min(group.indices)
            if cursor >= group_start + len(group.indices):
                metrics["groups"] = number
                continue
            local_start = max(0, cursor - group_start)
            remaining = len(group.indices) - local_start
            _upload(volume, runtime, group.files, "dataset")
            remote = f"outputs/group-{number:05d}"
            payload = {
                "root": "/scratch/dataset",
                "selected_indices": group.indices,
                "start_cursor": local_start,
                "max_batches": remaining,
                "epochs": epochs,
                "seed": seed,
                "checkpoint_path": "/scratch/checkpoint.pt",
                "output_prefix": f"/data/{remote}",
                "device": "cuda",
                **(payload_overrides or {}),
            }
            result = _retry(runtime, lambda payload=payload: runner.call("train_group", payload))
            download_outputs(
                volume,
                runtime,
                remote,
                run_root,
                ["metrics.json", "checkpoint.pt", "checkpoint.sha256", "checkpoint.json"],
            )
            for name in group.files:
                _retry(runtime, lambda name=name: volume.remove(f"dataset/{name}"))
            _retry(runtime, lambda remote=remote: volume.remove(remote, recursive=True))
            cursor = group_start + int(result.get("cursor", local_start + remaining))
            metrics.update(result)
            metrics["groups"] = number + 1
            metrics["cursor"] = cursor
            metrics["total_batches"] = total
            atomic_write_json(progress_path, metrics)
        if metrics["groups"] == 0 and progress_path.is_file():
            metrics.update(json.loads(progress_path.read_text()))
        return metrics
    finally:
        _retry(runtime, lambda: volume.remove("dataset", recursive=True))
        _retry(runtime, lambda: volume.remove("outputs", recursive=True))
        _retry(runtime, lambda: volume.remove("checkpoint.pt"))
