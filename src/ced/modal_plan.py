from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from ced.acquisition.base import atomic_write_json, sha256_file
from ced.config.models import RuntimeConfig
from ced.kg.audit import validate_checksums
from ced.model.schemas import TrainingExample

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

    def stream(self, function: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]: ...


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
    edge_type_count: int | None = None

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
        {
            "dataset": dataset,
            "mode": mode,
            "run_id": run.name,
            "created_utc": stamp,
            "modal": runtime.modal.model_dump(mode="json"),
        },
    )
    return run


def dataset_transfer_plan(root: Path, runtime: RuntimeConfig, mode: str) -> TransferPlan:
    if mode not in {"train", "evaluate"}:
        raise ValueError("mode must be train or evaluate")
    manifest = json.loads((root / "manifest.json").read_text())
    validate_checksums(root, manifest["checksums"])
    shared_names: set[str] = set(SHARED_TRAINING_FILES if mode == "train" else SHARED_EVALUATION_FILES)
    shared: dict[str, Path] = {}
    for item in _tensor_paths(manifest.get("statement_attributes", {})):
        shared_names.add(item)
    if mode == "train":
        shared_names.update(manifest["tensors"][name]["path"] for name in ("edge_type_id", "split"))
        for item in _tensor_paths(manifest.get("features", {})):
            shared_names.add(item)
    else:
        for item in _tensor_paths(manifest.get("tensors", {})):
            shared_names.add(item)
        for item in _tensor_paths(manifest.get("features", {})):
            shared_names.add(item)
    for item in _tensor_paths(manifest.get("semantic", {})):
        shared_names.add(item)
    for item in _tensor_paths(manifest.get("publications", {})):
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
    edge_type_count = int(manifest["edge_type_count"]) if mode == "train" else None
    if mode == "train":
        current = BatchGroup(indices=[], files={}, bytes=0)
        for index, example in enumerate(manifest["batches"]):
            TrainingExample.model_validate(example)
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
    return TransferPlan(
        mode=mode,
        root=root,
        shared=shared,
        groups=groups,
        shared_bytes=shared_bytes,
        edge_type_count=edge_type_count,
    )


def _upload(volume: VolumeClient, runtime: RuntimeConfig, files: Mapping[str, Path], prefix: str) -> None:
    _retry(runtime, lambda: volume.upload({f"{prefix}/{name}": path for name, path in files.items()}))


def download_outputs(
    volume: VolumeClient, runtime: RuntimeConfig, remote_prefix: str, local: Path, names: Iterable[str]
) -> None:
    local.mkdir(parents=True, exist_ok=True)
    for name in names:
        _retry(runtime, lambda name=name: volume.download(f"{remote_prefix}/{name}", local / name))


class TrainingProgressRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    group_index: int = Field(ge=0)
    batch_start: int = Field(ge=0)
    batch_end: int = Field(gt=0)
    epoch: int = Field(ge=0)
    cursor: int = Field(ge=0)
    config_sha256: str = Field(pattern="[0-9a-f]{64}")
    checkpoint_path: str
    checkpoint_sha256: str = Field(pattern="[0-9a-f]{64}")


TRAINING_PROGRESS_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("group_index", pa.int64()),
        ("batch_start", pa.int64()),
        ("batch_end", pa.int64()),
        ("epoch", pa.int64()),
        ("cursor", pa.int64()),
        ("config_sha256", pa.string()),
        ("checkpoint_path", pa.string()),
        ("checkpoint_sha256", pa.string()),
    ]
)


def _training_identity(plan: TransferPlan, seed: int, overrides: Mapping[str, Any]) -> str:
    from ced.model.training import config_hash

    document = json.loads((plan.root / "manifest.json").read_text())
    return config_hash(
        document,
        learning_rate=float(overrides.get("learning_rate", 1e-3)),
        seed=seed,
        auxiliary_weight=float(overrides.get("auxiliary_weight", 0.1)),
        attribute_weight=float(overrides.get("attribute_weight", 0.1)),
    )


def validate_checkpoint(
    root: Path, expected_config_sha256: str, expected_metrics: Mapping[str, Any] | None = None
) -> str:
    """Validate one complete checkpoint directory and return its content digest."""
    checkpoint = root / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
    recorded = (root / "checkpoint.sha256").read_text().split()
    actual = sha256_file(checkpoint)
    if not recorded or recorded[0] != actual:
        raise ValueError(f"checkpoint checksum mismatch: {checkpoint}")
    metadata = json.loads((root / "checkpoint.json").read_text())
    if metadata.get("config_hash") != expected_config_sha256:
        raise ValueError(f"checkpoint configuration mismatch: {root}")
    if expected_metrics is not None and metadata.get("metrics") != dict(expected_metrics):
        raise ValueError(f"checkpoint metrics mismatch: {root}")
    return actual


def _checkpoint_group_dir(epoch: int, group: int) -> str:
    prefix = f"epoch-{epoch:05d}/" if epoch else ""
    return f"checkpoints/{prefix}group-{group:05d}"


def _checkpoint_relative_path(epoch: int, group: int) -> str:
    return f"{_checkpoint_group_dir(epoch, group)}/checkpoint.pt"


def _read_training_progress(plan: TransferPlan, run_root: Path) -> list[TrainingProgressRow]:
    path = run_root / "progress.parquet"
    if not path.is_file():
        return []
    table = pq.read_table(path)
    if table.schema != TRAINING_PROGRESS_SCHEMA:
        raise ValueError(f"training progress schema mismatch: {path}")
    rows = [TrainingProgressRow.model_validate(row) for row in table.to_pylist()]
    for position, row in enumerate(rows):
        if not plan.groups:
            raise ValueError("training progress exists for an empty batch plan")
        epoch, group_index = divmod(position, len(plan.groups))
        if row.group_index != group_index or row.epoch != epoch:
            raise ValueError("training progress groups are missing, duplicated, or out of order")
        expected = plan.groups[group_index]
        expected_path = _checkpoint_relative_path(epoch, group_index)
        if (
            row.run_id != run_root.name
            or row.batch_start != min(expected.indices)
            or row.batch_end != max(expected.indices) + 1
            or row.cursor != row.batch_end
            or row.checkpoint_path != expected_path
        ):
            raise ValueError(f"training progress row {position} does not match its plan or checkpoint")
    return rows


def _write_training_progress(run_root: Path, rows: list[TrainingProgressRow]) -> None:
    path = run_root / "progress.parquet"
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    table = pa.Table.from_pylist([row.model_dump(mode="python") for row in rows], schema=TRAINING_PROGRESS_SCHEMA)
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def _publish_latest_checkpoint(run_root: Path, group_root: Path) -> None:
    for name in ("checkpoint.pt", "checkpoint.sha256", "checkpoint.json"):
        temporary = run_root / f".{name}.tmp-{os.getpid()}"
        temporary.unlink(missing_ok=True)
        os.link(group_root / name, temporary)
        temporary.replace(run_root / name)


def _upload_checkpoint(volume: VolumeClient, runtime: RuntimeConfig, root: Path) -> None:
    files = {name: root / name for name in ("checkpoint.pt", "checkpoint.sha256", "checkpoint.json")}
    _retry(runtime, lambda: volume.upload(files))


def validate_training_resume(plan: TransferPlan, run_root: Path, *, seed: int) -> None:
    """Validate the durable resume boundary before paid orchestration starts."""
    identity = _training_identity(plan, seed, {})
    rows = _read_training_progress(plan, run_root)
    if not rows:
        if (run_root / "checkpoint.pt").exists():
            raise ValueError("training checkpoint has no relational progress")
        return
    if any(row.config_sha256 != identity for row in rows):
        raise ValueError("training resume configuration mismatch")
    latest = rows[-1]
    digest = validate_checkpoint((run_root / latest.checkpoint_path).parent, identity)
    if digest != latest.checkpoint_sha256:
        raise ValueError("training progress checkpoint hash does not match its ledger")


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
    adopt_encoded: Callable[[Path, dict[str, Any]], int] | None = None,
) -> dict[str, Any]:
    if plan.mode == "encode" and adopt_encoded is None:
        raise ValueError("text encoding requires a per-part adoption callback")
    checkpoint = run_root / "checkpoint.pt"
    progress_rows: list[TrainingProgressRow] = []
    identity = ""
    if plan.mode == "train":
        if epochs < 1:
            raise ValueError("epochs must be positive")
        protected = {
            "root",
            "selected_indices",
            "start_epoch",
            "start_cursor",
            "edge_type_count",
            "max_batches",
            "global_cursor",
            "config_sha256",
            "epochs",
            "seed",
            "checkpoint_path",
            "output_prefix",
        }
        if protected.intersection(payload_overrides or {}):
            raise ValueError("training payload overrides contain protected orchestration fields")
        identity = _training_identity(plan, seed, payload_overrides or {})
        progress_rows = _read_training_progress(plan, run_root)
        if any(row.config_sha256 != identity for row in progress_rows):
            raise ValueError("training resume configuration mismatch")
        if progress_rows:
            latest = progress_rows[-1]
            latest_root = (run_root / latest.checkpoint_path).parent
            digest = validate_checkpoint(latest_root, identity)
            if digest != latest.checkpoint_sha256:
                raise ValueError("training progress checkpoint hash does not match its ledger")
            _publish_latest_checkpoint(run_root, latest_root)
        elif checkpoint.exists():
            raise ValueError("training checkpoint has no relational progress")
    _upload(volume, runtime, plan.shared, "dataset")
    if checkpoint.is_file():
        files = {"checkpoint.pt": checkpoint}
        for name in ("checkpoint.sha256", "checkpoint.json"):
            path = run_root / name
            if path.is_file():
                files[name] = path
        _retry(runtime, lambda: volume.upload(files))
    metrics: dict[str, Any] = {"mode": plan.mode, "groups": 0}
    cursor = progress_rows[-1].cursor if progress_rows else 0
    try:
        if plan.mode == "encode":
            assert adopt_encoded is not None
            remote = "outputs/encode"
            payload = {
                "root": "/data/dataset",
                "output_prefix": f"/data/{remote}",
                **(payload_overrides or {}),
            }
            metrics.update({"rows": 0, "parts": 0})
            for streamed in runner.stream("encode_text", payload):
                descriptor = dict(streamed)
                files = descriptor.pop("files", None)
                checksums = descriptor.pop("checksums", None)
                part = int(descriptor.get("part", -1))
                expected_part = int((payload_overrides or {}).get("start_part", 0)) + metrics["parts"]
                if part != expected_part:
                    raise ValueError(f"unexpected encoded part {part}; expected {expected_part}")
                rows = int(descriptor.get("rows", 0))
                input_row_start = int(descriptor.get("input_row_start", -1))
                if rows <= 0 or input_row_start != metrics["rows"]:
                    raise ValueError(
                        f"invalid encoded input slice {input_row_start}:{input_row_start + rows} "
                        f"at stream offset {metrics['rows']}"
                    )
                name = f"part-{part:06d}.safetensors"
                expected_files = {f"vectors/{name}", f"index/{Path(name).with_suffix('.parquet').name}"}
                if not isinstance(files, list) or len(files) != 2 or set(files) != expected_files:
                    raise ValueError(f"encoded part {part} did not contain exactly one vector/index pair")
                if not isinstance(checksums, dict) or set(checksums) != expected_files:
                    raise ValueError(f"encoded part {part} checksums do not match its vector/index pair")
                dimension = int(descriptor.get("dimension", 0))
                if dimension <= 0 or metrics.get("dimension", dimension) != dimension:
                    raise ValueError(f"encoded part {part} has an inconsistent dimension")
                download_outputs(volume, runtime, remote, run_root, files)
                validate_checksums(run_root, checksums)
                adopted = adopt_encoded(run_root, dict(streamed))
                if adopted != rows:
                    raise ValueError(f"encoded part {part} adopted {adopted} rows, expected {rows}")
                for name in files:
                    _retry(runtime, lambda name=name: volume.remove(f"{remote}/{name}"))
                metrics["rows"] += adopted
                metrics["parts"] += 1
                metrics["dimension"] = dimension
            return metrics
        if plan.mode == "evaluate":
            remote = "outputs/evaluate"
            result = _retry(
                runtime,
                lambda: runner.call(
                    "evaluate",
                    {
                        "root": "/data/dataset",
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
        if not plan.groups:
            raise ValueError("dataset contains no training batches")
        if plan.edge_type_count is None:
            raise ValueError("training plan has no edge type metadata")
        metrics.update(
            {
                "cursor": cursor,
                "groups": len(progress_rows),
                "total_batches": total,
                "epoch": len(progress_rows) // len(plan.groups),
                "total_epochs": epochs,
            }
        )
        if progress_rows:
            latest_root = (run_root / progress_rows[-1].checkpoint_path).parent
            metrics.update(json.loads((latest_root / "metrics.json").read_text()))
            metrics.update(
                {
                    "groups": len(progress_rows),
                    "total_batches": total,
                    "epoch": len(progress_rows) // len(plan.groups),
                    "total_epochs": epochs,
                }
            )
        total_steps = epochs * len(plan.groups)
        if len(progress_rows) > total_steps:
            raise ValueError("training progress exceeds the requested epochs")
        for step in range(len(progress_rows), total_steps):
            epoch, number = divmod(step, len(plan.groups))
            group = plan.groups[number]
            epoch_prefix = f"epoch-{epoch:05d}/" if epoch else ""
            group_start = min(group.indices)
            local_start = 0
            remaining = len(group.indices)
            _upload(volume, runtime, group.files, "dataset")
            remote = f"outputs/{epoch_prefix}group-{number:05d}"
            payload = {
                "root": "/data/dataset",
                "selected_indices": group.indices,
                "start_epoch": epoch,
                "start_cursor": local_start,
                "edge_type_count": plan.edge_type_count,
                "max_batches": remaining,
                "global_cursor": group_start,
                "config_sha256": identity,
                "epochs": epochs,
                "seed": seed,
                "checkpoint_path": "/scratch/checkpoint.pt",
                "output_prefix": f"/data/{remote}",
                "device": "cuda",
                **(payload_overrides or {}),
            }
            result = _retry(runtime, lambda payload=payload: runner.call("train_group", payload))
            group_root = run_root / _checkpoint_group_dir(epoch, number)
            staging = group_root.with_name(f"{group_root.name}.partial")
            if staging.exists():
                shutil.rmtree(staging)
            download_outputs(
                volume,
                runtime,
                remote,
                staging,
                ["metrics.json", "checkpoint.pt", "checkpoint.sha256", "checkpoint.json"],
            )
            expected_cursor = max(group.indices) + 1
            if (
                int(result.get("cursor", -1)) != expected_cursor
                or int(result.get("consumed", -1)) != remaining
                or int(result.get("epoch", -1)) != epoch
            ):
                raise ValueError(f"training group {number} did not complete its expected epoch and batch range")
            if json.loads((staging / "metrics.json").read_text()) != result:
                raise ValueError(f"training group {number} returned mismatched metrics")
            digest = validate_checkpoint(staging, identity, result)
            if group_root.exists():
                shutil.rmtree(group_root)
            staging.rename(group_root)
            progress_rows.append(
                TrainingProgressRow(
                    run_id=run_root.name,
                    group_index=number,
                    batch_start=group_start,
                    batch_end=expected_cursor,
                    epoch=epoch,
                    cursor=expected_cursor,
                    config_sha256=identity,
                    checkpoint_path=_checkpoint_relative_path(epoch, number),
                    checkpoint_sha256=digest,
                )
            )
            _write_training_progress(run_root, progress_rows)
            _publish_latest_checkpoint(run_root, group_root)
            _upload_checkpoint(volume, runtime, group_root)
            for name in group.files:
                _retry(runtime, lambda name=name: volume.remove(f"dataset/{name}"))
            _retry(runtime, lambda remote=remote: volume.remove(remote, recursive=True))
            cursor = expected_cursor
            metrics.update(result)
            metrics["groups"] = len(progress_rows)
            metrics["cursor"] = cursor
            metrics["total_batches"] = total
            metrics["epoch"] = len(progress_rows) // len(plan.groups)
            metrics["total_epochs"] = epochs
        return metrics
    finally:
        _retry(runtime, lambda: volume.remove("dataset", recursive=True))
        _retry(runtime, lambda: volume.remove("outputs", recursive=True))
        for name in ("checkpoint.pt", "checkpoint.sha256", "checkpoint.json"):
            _retry(runtime, lambda name=name: volume.remove(name))
