from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ced.acquisition.base import atomic_write_json
from ced.config.models import ModalSettings, RuntimeConfig
from ced.modal_plan import TransferPlan, _retry, dataset_transfer_plan, execute_plan, prepare_run_directory


class FakeVolume:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.removed: list[str] = []

    def upload(self, files: Mapping[str, Path]) -> None:
        for remote, local in files.items():
            self.files[remote] = local.read_bytes()

    def download(self, remote: str, local: Path) -> None:
        if remote not in self.files:
            raise FileNotFoundError(remote)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self.files[remote])

    def remove(self, remote: str, *, recursive: bool = False) -> None:
        keys = [key for key in self.files if key == remote or (recursive and key.startswith(f"{remote}/"))]
        if not keys:
            return
        for key in keys:
            del self.files[key]
        self.removed.extend(keys)

    def list(self, prefix: str) -> list[str]:
        return sorted(key for key in self.files if key.startswith(prefix))


class FakeRunner:
    def __init__(self, volume: FakeVolume) -> None:
        self.volume = volume
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, function: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((function, payload))
        prefix = str(payload["output_prefix"]).removeprefix("/data/")
        selected = payload.get("selected_indices") or []
        start = int(payload.get("start_cursor", 0))
        metrics = {
            "cursor": start + len(selected),
            "batches": float(len(selected)),
            "loss": 0.5,
            "ranking": payload.get("ranking", "exact"),
        }
        self.volume.files[f"{prefix}/metrics.json"] = json.dumps(metrics).encode()
        self.volume.files[f"{prefix}/checkpoint.pt"] = b"checkpoint"
        self.volume.files[f"{prefix}/checkpoint.sha256"] = b"digest"
        self.volume.files[f"{prefix}/checkpoint.json"] = json.dumps({"cursor": metrics["cursor"]}).encode()
        return metrics


def runtime(tmp_path: Path, **modal: Any) -> RuntimeConfig:
    settings = ModalSettings(**({"gpu": "L40S", "retries": 1} | modal))
    return RuntimeConfig(
        remote_project=tmp_path,
        artifact_root=tmp_path / ".ced",
        disk_reserve_gb=0,
        modal=settings,
    )


def test_modal_plan_rotates_bounded_batches(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    root = tmp_path / "dataset"
    batch_bytes = synthetic_dataset(root, batches=3)
    config = runtime(tmp_path, max_shared_bytes=1024**3, max_batch_group_bytes=batch_bytes * 2 + 16)
    plan = dataset_transfer_plan(root, config, "train")
    assert [group.indices for group in plan.groups] == [[0, 1], [2]]
    assert plan.shared_bytes <= config.modal.max_shared_bytes

    run_root = tmp_path / "run"
    run_root.mkdir()
    volume = FakeVolume()
    runner = FakeRunner(volume)
    metrics = execute_plan(plan, config, volume, runner, run_root, epochs=1, seed=7)
    assert metrics["cursor"] == 3
    assert metrics["groups"] == 2
    assert [payload["start_cursor"] for _, payload in runner.calls] == [0, 0]
    assert (run_root / "checkpoint.pt").read_bytes() == b"checkpoint"
    assert json.loads((run_root / "progress.json").read_text())["cursor"] == 3
    assert not any(key.startswith("dataset/batches") for key in volume.files)
    assert not any(key.startswith("outputs") for key in volume.files)


def test_modal_plan_resumes_from_progress(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    root = tmp_path / "dataset"
    batch_bytes = synthetic_dataset(root, batches=3)
    config = runtime(tmp_path, max_shared_bytes=1024**3, max_batch_group_bytes=batch_bytes * 2 + 16)
    plan = dataset_transfer_plan(root, config, "train")
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "checkpoint.pt").write_bytes(b"old")
    atomic_write_json(run_root / "progress.json", {"cursor": 2, "groups": 1})
    volume = FakeVolume()
    runner = FakeRunner(volume)
    metrics = execute_plan(plan, config, volume, runner, run_root, epochs=1, seed=7)
    assert len(runner.calls) == 1
    payload = runner.calls[0][1]
    assert payload["selected_indices"] == [2]
    assert payload["start_cursor"] == 0
    assert metrics["cursor"] == 3


def test_modal_plan_rejects_oversized_batch(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    root = tmp_path / "dataset"
    batch_bytes = synthetic_dataset(root, batches=1)
    config = runtime(tmp_path, max_batch_group_bytes=batch_bytes - 1)
    with pytest.raises(ValueError, match="exceeds the Modal transfer bound"):
        dataset_transfer_plan(root, config, "train")


def test_transfer_plan_rejects_unknown_mode_and_missing_shared_files(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"checksums": {}}))
    with pytest.raises(ValueError, match="mode must be train or evaluate"):
        dataset_transfer_plan(root, runtime(tmp_path), "finetune")
    with pytest.raises(FileNotFoundError, match="required dataset artifact is missing"):
        dataset_transfer_plan(root, runtime(tmp_path), "evaluate")


def test_transfer_plan_rejects_oversized_shared_payload(tmp_path: Path) -> None:
    """The volume rotates per-group files but shares one immutable payload; a shared
    payload past the Modal bound must fail at planning time, not mid-upload."""
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "entities.parquet").write_bytes(b"x" * 8)
    (root / "relations.parquet").write_bytes(b"x" * 8)
    (root / "types.parquet").write_bytes(b"x" * 8)
    (root / "manifest.json").write_text(json.dumps({"checksums": {}}))
    config = runtime(tmp_path, max_shared_bytes=8)
    with pytest.raises(ValueError, match="shared transfer size"):
        dataset_transfer_plan(root, config, "evaluate")


def test_prepare_run_directory_stamps_config_and_is_unique(
    tmp_path: Path, synthetic_dataset: Callable[..., int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run directory must capture the paid-job configuration and never collide: the
    identity hash makes reruns land in fresh directories instead of mixing metrics."""
    synthetic_dataset(runtime(tmp_path).artifact_path("datasets", "synthetic"))
    config = runtime(tmp_path)
    frozen = datetime(2026, 1, 1, tzinfo=UTC)
    monkeypatch.setattr("ced.modal_plan.datetime", SimpleNamespace(now=lambda tz: frozen))
    run = prepare_run_directory(config, "synthetic", "train")
    recorded = json.loads((run / "run-config.json").read_text())
    assert recorded["dataset"] == "synthetic"
    assert recorded["mode"] == "train"
    assert recorded["modal"]["gpu"] == "L40S"
    with pytest.raises(FileExistsError):
        prepare_run_directory(config, "synthetic", "train")


def synthetic_dataset_root(tmp_path: Path, builder: Callable[..., int]) -> Path:
    root = tmp_path / "dataset"
    builder(root, batches=1)
    return root


def test_execute_plan_evaluates_without_group_rotation(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    root = synthetic_dataset_root(tmp_path, synthetic_dataset)
    config = runtime(tmp_path, max_shared_bytes=1024**3)
    plan = dataset_transfer_plan(root, config, "evaluate")
    run_root = tmp_path / "run"
    run_root.mkdir()
    volume = FakeVolume()
    runner = FakeRunner(volume)
    metrics = execute_plan(plan, config, volume, runner, run_root, seed=7)
    function, payload = runner.calls[0]
    assert function == "evaluate"
    assert payload["checkpoint_path"] == "/scratch/checkpoint.pt"
    assert metrics["mode"] == "evaluate"
    assert json.loads((run_root / "metrics.json").read_text())  # downloaded into the run directory


def test_retry_does_not_swallow_persistent_failures(tmp_path: Path) -> None:
    """Retries are for transient volume faults; a failure on the final attempt must
    propagate unchanged so callers keep matching on the exception type."""
    config = runtime(tmp_path, retries=3, retry_base_seconds=0.001)
    attempts = 0

    def failing() -> None:
        nonlocal attempts
        attempts += 1
        raise FileNotFoundError("volume gone")

    with pytest.raises(FileNotFoundError, match="volume gone"):
        _retry(config, failing)
    assert attempts == 3


def test_paid_modal_confirmation_requires_explicit_approval(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("modal")
    from ced.modal_app import confirm_paid

    config = runtime(tmp_path)
    plan = TransferPlan(mode="train", root=tmp_path, shared={}, groups=[])
    confirm_paid(plan, config, assume_yes=True)
    assert "Approved by --yes" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="interactive terminal"):
        confirm_paid(plan, config, assume_yes=False)
