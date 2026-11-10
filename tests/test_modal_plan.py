from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from ced.acquisition.base import atomic_write_json
from ced.config.models import ModalSettings, RuntimeConfig
from ced.modal_plan import TransferPlan, dataset_transfer_plan, execute_plan


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


def test_paid_modal_confirmation_requires_explicit_approval(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("modal")
    from ced.modal_app import confirm_paid

    config = runtime(tmp_path)
    plan = TransferPlan(mode="train", root=tmp_path, shared={}, groups=[])
    confirm_paid(plan, config, assume_yes=True)
    assert "Approved by --yes" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="interactive terminal"):
        confirm_paid(plan, config, assume_yes=False)
