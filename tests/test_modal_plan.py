from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from ced.acquisition.base import sha256_file
from ced.config.models import ModalSettings, RuntimeConfig
from ced.modal_plan import TransferPlan, _retry, dataset_transfer_plan, execute_plan, prepare_run_directory
from ced.model.training import train_batches


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
        self.checkpoint_inputs: list[bytes | None] = []

    def stream(self, function: str, payload: dict[str, Any]):
        raise AssertionError("stream is only used by encoding test runners")

    def call(self, function: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((function, payload))
        prefix = str(payload["output_prefix"]).removeprefix("/data/")
        selected = payload.get("selected_indices") or []
        if function == "train_group":
            self.checkpoint_inputs.append(self.volume.files.get("checkpoint.pt"))
        start = int(payload.get("start_cursor", 0))
        global_start = int(payload.get("global_cursor", (min(selected) if selected else 0) + start))
        consumed = len(selected) - start
        metrics = {
            "epoch": float(payload.get("start_epoch", 0)),
            "cursor": global_start + consumed,
            "consumed": float(consumed),
            "batches": float(consumed),
            "loss": 0.5,
            "ranking": payload.get("ranking", "exact"),
        }
        checkpoint = f"checkpoint-{metrics['cursor']}".encode()
        metadata = {
            "epoch": int(payload.get("start_epoch", 0)),
            "cursor": metrics["cursor"],
            "config_hash": payload.get("config_sha256", "test-config"),
            "metrics": metrics,
            "device": "cpu",
        }
        self.volume.files[f"{prefix}/metrics.json"] = json.dumps(metrics).encode()
        self.volume.files[f"{prefix}/checkpoint.pt"] = checkpoint
        self.volume.files[f"{prefix}/checkpoint.sha256"] = (
            f"{hashlib.sha256(checkpoint).hexdigest()}  checkpoint.pt\n".encode()
        )
        self.volume.files[f"{prefix}/checkpoint.json"] = json.dumps(metadata).encode()
        return metrics


class TrainingRunner:
    """Execute real training with only the files visible in one Modal group."""

    def __init__(self, volume: FakeVolume, root: Path) -> None:
        self.volume = volume
        self.root = root
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def stream(self, function: str, payload: dict[str, Any]):
        raise AssertionError("stream is only used by encoding test runners")

    def call(self, function: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert function == "train_group"
        call_root = self.root / f"call-{len(self.calls)}"
        self.calls.append((function, payload))
        root = call_root / "dataset"
        for name, value in self.volume.files.items():
            if name.startswith("dataset/"):
                path = call_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
        checkpoint = call_root / "checkpoint.pt"
        if "checkpoint.pt" in self.volume.files:
            checkpoint.write_bytes(self.volume.files["checkpoint.pt"])
        options = {key: payload[key] for key in ("start_epoch", "edge_type_count") if key in payload}
        metrics = train_batches(
            root,
            epochs=payload["epochs"],
            seed=payload["seed"],
            checkpoint_path=checkpoint,
            device="cpu",
            selected_indices=payload["selected_indices"],
            start_cursor=payload["start_cursor"],
            global_cursor=payload["global_cursor"],
            max_batches=payload["max_batches"],
            **options,
        )
        prefix = str(payload["output_prefix"]).removeprefix("/data/")
        self.volume.files[f"{prefix}/metrics.json"] = json.dumps(metrics).encode()
        for name in ("checkpoint.pt", "checkpoint.sha256", "checkpoint.json"):
            self.volume.files[f"{prefix}/{name}"] = (call_root / name).read_bytes()
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
    import pyarrow.parquet as pq

    root = tmp_path / "dataset"
    batch_bytes = synthetic_dataset(root, batches=3)
    config = runtime(tmp_path, max_shared_bytes=1024**3, max_batch_group_bytes=batch_bytes * 2 + 16)
    plan = dataset_transfer_plan(root, config, "train")
    assert [group.indices for group in plan.groups] == [[0, 1], [2]]
    assert plan.shared_bytes <= config.modal.max_shared_bytes
    assert plan.edge_type_count == 2
    assert config.modal.single_use_containers is False

    run_root = tmp_path / "run"
    run_root.mkdir()
    volume = FakeVolume()
    runner = FakeRunner(volume)
    metrics = execute_plan(plan, config, volume, runner, run_root, epochs=1, seed=7)
    assert metrics["cursor"] == 3
    assert metrics["groups"] == 2
    assert [payload["start_cursor"] for _, payload in runner.calls] == [0, 0]
    assert [payload["global_cursor"] for _, payload in runner.calls] == [0, 2]
    assert runner.checkpoint_inputs == [None, b"checkpoint-2"]
    assert (run_root / "checkpoint.pt").read_bytes() == b"checkpoint-3"
    assert sha256_file(run_root / "checkpoint.pt") == (run_root / "checkpoint.sha256").read_text().split()[0]
    progress = pq.read_table(run_root / "progress.parquet")
    assert progress.column_names == [
        "run_id",
        "group_index",
        "batch_start",
        "batch_end",
        "epoch",
        "cursor",
        "config_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
    ]
    assert progress.column("group_index").to_pylist() == [0, 1]
    assert progress.column("batch_end").to_pylist() == [2, 3]
    assert progress.column("checkpoint_path").to_pylist() == [
        "checkpoints/group-00000/checkpoint.pt",
        "checkpoints/group-00001/checkpoint.pt",
    ]
    assert not (run_root / "progress.json").exists()
    assert not any(key.startswith("dataset/batches") for key in volume.files)
    assert not any(key.startswith("outputs") for key in volume.files)


def test_training_resume_uses_relational_completed_groups(
    tmp_path: Path, synthetic_dataset: Callable[..., int]
) -> None:
    """Only ledger-complete groups are skipped, and each group gets the latest checkpoint.

    Why: JSON cursor-only resume mixed global and group-local offsets and could
    upload the same stale checkpoint to every group. A failed second group must
    leave the first immutable checkpoint and relational record as the sole resume
    boundary, with no repeated GPU work on the completed prefix.
    """
    import pyarrow.parquet as pq

    root = tmp_path / "dataset"
    batch_bytes = synthetic_dataset(root, batches=3)
    config = runtime(tmp_path, max_shared_bytes=1024**3, max_batch_group_bytes=batch_bytes * 2 + 16)
    plan = dataset_transfer_plan(root, config, "train")
    run_root = tmp_path / "run"
    run_root.mkdir()

    class FailSecondGroup(FakeRunner):
        def call(self, function: str, payload: dict[str, Any]) -> dict[str, Any]:
            if payload.get("selected_indices") == [2]:
                raise RuntimeError("injected second-group failure")
            return super().call(function, payload)

    failing_volume = FakeVolume()
    with pytest.raises(RuntimeError, match="injected second-group failure"):
        execute_plan(plan, config, failing_volume, FailSecondGroup(failing_volume), run_root, epochs=1, seed=7)
    progress = pq.read_table(run_root / "progress.parquet")
    assert progress.column("group_index").to_pylist() == [0]
    assert (run_root / "checkpoint.pt").read_bytes() == b"checkpoint-2"

    volume = FakeVolume()
    runner = FakeRunner(volume)
    metrics = execute_plan(plan, config, volume, runner, run_root, epochs=1, seed=7)
    assert len(runner.calls) == 1
    payload = runner.calls[0][1]
    assert payload["selected_indices"] == [2]
    assert payload["start_cursor"] == 0
    assert payload["global_cursor"] == 2
    assert runner.checkpoint_inputs == [b"checkpoint-2"]
    assert metrics["cursor"] == 3
    assert metrics["groups"] == 2
    assert pq.read_table(run_root / "progress.parquet").num_rows == 2


@pytest.mark.parametrize(
    ("mutation", "match"), [("config", "configuration mismatch"), ("checksum", "checksum mismatch")]
)
def test_training_resume_rejects_invalid_checkpoint(
    tmp_path: Path, synthetic_dataset: Callable[..., int], mutation: str, match: str
) -> None:
    """A ledger cannot make a corrupt or incompatible checkpoint resumable.

    Why: paid resume validates both content and configuration identity before any
    new group call. Otherwise corrupted state could silently poison later groups.
    """
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    config = runtime(tmp_path)
    plan = dataset_transfer_plan(root, config, "train")
    run_root = tmp_path / "run"
    run_root.mkdir()
    volume = FakeVolume()
    execute_plan(plan, config, volume, FakeRunner(volume), run_root, seed=7)
    group = run_root / "checkpoints" / "group-00000"
    if mutation == "config":
        metadata = json.loads((group / "checkpoint.json").read_text())
        metadata["config_hash"] = "0" * 64
        (group / "checkpoint.json").write_text(json.dumps(metadata))
    else:
        (group / "checkpoint.pt").write_bytes(b"corrupt")

    runner = FakeRunner(FakeVolume())
    with pytest.raises(ValueError, match=match):
        execute_plan(plan, config, runner.volume, runner, run_root, seed=7)
    assert runner.calls == []


@pytest.mark.parametrize("field", ["config_sha256", "global_cursor", "selected_indices", "seed"])
def test_training_rejects_protected_payload_overrides(
    tmp_path: Path, synthetic_dataset: Callable[..., int], field: str
) -> None:
    """Protected orchestration overrides fail before uploads or GPU calls."""
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    config = runtime(tmp_path)
    plan = dataset_transfer_plan(root, config, "train")
    run_root = tmp_path / "run"
    run_root.mkdir()
    volume = FakeVolume()
    runner = FakeRunner(volume)
    with pytest.raises(ValueError, match="protected orchestration fields"):
        execute_plan(plan, config, volume, runner, run_root, payload_overrides={field: "invalid"})
    assert not volume.files
    assert not runner.calls


def test_training_resume_rejects_changed_seed(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    """A changed seed cannot resume a previously completed checkpoint lineage."""
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    config = runtime(tmp_path)
    plan = dataset_transfer_plan(root, config, "train")
    run_root = tmp_path / "run"
    run_root.mkdir()
    volume = FakeVolume()
    execute_plan(plan, config, volume, FakeRunner(volume), run_root, seed=7)
    runner = FakeRunner(FakeVolume())
    with pytest.raises(ValueError, match="configuration mismatch"):
        execute_plan(plan, config, runner.volume, runner, run_root, seed=8)
    assert not runner.calls
    assert not runner.volume.files


def test_training_partial_download_preserves_last_complete_group(
    tmp_path: Path, synthetic_dataset: Callable[..., int]
) -> None:
    """A partial next checkpoint cannot replace the last durable group.

    Why: download failures happen after GPU computation. Staging, checksum
    validation, and the completed-row ledger must prevent that partial file from
    becoming the next resume input, while preserving the previous GPU result.
    """
    import pyarrow.parquet as pq

    root = tmp_path / "dataset"
    batch_bytes = synthetic_dataset(root, batches=3)
    config = runtime(tmp_path, max_shared_bytes=1024**3, max_batch_group_bytes=batch_bytes * 2 + 16)
    plan = dataset_transfer_plan(root, config, "train")
    run_root = tmp_path / "run"
    run_root.mkdir()

    class PartialSecondDownload(FakeVolume):
        def download(self, remote: str, local: Path) -> None:
            if remote == "outputs/group-00001/checkpoint.pt":
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(b"partial")
                raise RuntimeError("injected partial checkpoint download")
            super().download(remote, local)

    volume = PartialSecondDownload()
    with pytest.raises(RuntimeError, match="injected partial checkpoint download"):
        execute_plan(plan, config, volume, FakeRunner(volume), run_root, seed=7)
    assert pq.read_table(run_root / "progress.parquet").column("group_index").to_pylist() == [0]
    assert (run_root / "checkpoint.pt").read_bytes() == b"checkpoint-2"
    assert (run_root / "checkpoints" / "group-00001.partial" / "checkpoint.pt").read_bytes() == b"partial"

    resumed_volume = FakeVolume()
    runner = FakeRunner(resumed_volume)
    execute_plan(plan, config, resumed_volume, runner, run_root, seed=7)
    assert len(runner.calls) == 1
    assert runner.calls[0][1]["selected_indices"] == [2]
    assert runner.checkpoint_inputs == [b"checkpoint-2"]
    assert pq.read_table(run_root / "progress.parquet").num_rows == 2


def test_training_resume_rejects_non_contiguous_progress(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    """Missing or reordered ledger groups fail instead of selecting wrong work."""
    import pyarrow.parquet as pq

    root = tmp_path / "dataset"
    batch_bytes = synthetic_dataset(root, batches=3)
    config = runtime(tmp_path, max_shared_bytes=1024**3, max_batch_group_bytes=batch_bytes * 2 + 16)
    plan = dataset_transfer_plan(root, config, "train")
    run_root = tmp_path / "run"
    run_root.mkdir()
    volume = FakeVolume()
    execute_plan(plan, config, volume, FakeRunner(volume), run_root, seed=7)
    progress = pq.read_table(run_root / "progress.parquet")
    pq.write_table(progress.slice(1), run_root / "progress.parquet")

    runner = FakeRunner(FakeVolume())
    with pytest.raises(ValueError, match="missing, duplicated, or out of order"):
        execute_plan(plan, config, runner.volume, runner, run_root, seed=7)
    assert runner.calls == []


def test_modal_grouped_training_matches_full_epochs_and_resume(
    tmp_path: Path, synthetic_dataset: Callable[..., int]
) -> None:
    import pyarrow.parquet as pq

    root = tmp_path / "dataset"
    batch_bytes = synthetic_dataset(root, batches=3)
    expected_checkpoint = tmp_path / "expected.pt"
    expected = train_batches(root, epochs=2, seed=7, checkpoint_path=expected_checkpoint, device="cpu")
    config = runtime(tmp_path, max_shared_bytes=1024**3, max_batch_group_bytes=batch_bytes * 2 + 16)
    plan = dataset_transfer_plan(root, config, "train")
    run_root = tmp_path / "run"
    run_root.mkdir()

    first_volume = FakeVolume()
    first_runner = TrainingRunner(first_volume, tmp_path / "first-remote")
    first = execute_plan(plan, config, first_volume, first_runner, run_root, epochs=1, seed=7)
    assert first["batches"] == 3
    assert first["epoch"] == 1
    assert [payload["start_epoch"] for _, payload in first_runner.calls] == [0, 0]

    class FailLaterSecondGroup(TrainingRunner):
        def call(self, function: str, payload: dict[str, Any]) -> dict[str, Any]:
            if payload["start_epoch"] == 1 and payload["selected_indices"] == [2]:
                raise RuntimeError("injected later-epoch failure")
            return super().call(function, payload)

    failing_volume = FakeVolume()
    failing_runner = FailLaterSecondGroup(failing_volume, tmp_path / "failing-remote")
    with pytest.raises(RuntimeError, match="injected later-epoch failure"):
        execute_plan(plan, config, failing_volume, failing_runner, run_root, epochs=2, seed=7)
    assert pq.read_table(run_root / "progress.parquet").num_rows == 3

    resumed_volume = FakeVolume()
    resumed_runner = TrainingRunner(resumed_volume, tmp_path / "resumed-remote")
    resumed = execute_plan(plan, config, resumed_volume, resumed_runner, run_root, epochs=2, seed=7)
    assert resumed["batches"] == expected["batches"] == 6
    assert resumed["loss"] == expected["loss"]
    assert resumed["epoch"] == 2
    assert resumed["groups"] == 4
    assert [payload["start_epoch"] for _, payload in resumed_runner.calls] == [1]
    assert [payload["global_cursor"] for _, payload in resumed_runner.calls] == [2]
    progress = pq.read_table(run_root / "progress.parquet")
    assert progress.column("epoch").to_pylist() == [0, 0, 1, 1]
    assert progress.column("checkpoint_path").to_pylist()[-2:] == [
        "checkpoints/epoch-00001/group-00000/checkpoint.pt",
        "checkpoints/epoch-00001/group-00001/checkpoint.pt",
    ]
    actual_state = torch.load(run_root / "checkpoint.pt", map_location="cpu", weights_only=False)["state"]
    expected_state = torch.load(expected_checkpoint, map_location="cpu", weights_only=False)["state"]
    assert actual_state.keys() == expected_state.keys()
    for name in expected_state:
        assert torch.equal(actual_state[name], expected_state[name]), name

    rejected_runner = FakeRunner(FakeVolume())
    with pytest.raises(ValueError, match="progress exceeds the requested epochs"):
        execute_plan(plan, config, rejected_runner.volume, rejected_runner, run_root, epochs=1, seed=7)
    assert rejected_runner.calls == []


def test_training_rejects_non_positive_epochs(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    config = runtime(tmp_path)
    plan = dataset_transfer_plan(root, config, "train")
    run_root = tmp_path / "run"
    run_root.mkdir()
    volume = FakeVolume()
    runner = FakeRunner(volume)
    with pytest.raises(ValueError, match="epochs must be positive"):
        execute_plan(plan, config, volume, runner, run_root, epochs=0, seed=7)
    assert not volume.files
    assert runner.calls == []


def test_transfer_plan_requires_shard_metadata(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tensor_path = manifest["batches"][0]["graph"]["edge_type_id"]["path"]
    (root / tensor_path).unlink()
    del manifest["checksums"][tensor_path]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError):
        dataset_transfer_plan(root, runtime(tmp_path), "train")


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
    (root / "fitted_fields.parquet").write_bytes(b"x" * 8)
    (root / "fitted_categories.parquet").write_bytes(b"x" * 8)
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
    assert recorded["run_id"] == run.name
    assert recorded["modal"]["gpu"] == "L40S"
    with pytest.raises(FileExistsError):
        prepare_run_directory(config, "synthetic", "train")


def test_train_remote_resume_reuses_run_and_validates_before_paid(
    tmp_path: Path, synthetic_dataset: Callable[..., int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume preflight precedes paid approval and never allocates a new run ID.

    Why: the old path copied sidecars from `.ced/runs` instead of the selected run
    and confirmed payment before validating resume state. Continuing the original
    directory keeps its relational ledger and checkpoint lineage authoritative.
    """
    import ced.modal_app as modal_app

    synthetic_dataset(runtime(tmp_path).artifact_path("datasets", "synthetic"))
    config = runtime(tmp_path)
    source = prepare_run_directory(config, "synthetic", "train")
    (source / "checkpoint.pt").write_bytes(b"checkpoint")
    (source / "progress.parquet").write_bytes(b"ledger")
    events: list[str] = []
    monkeypatch.setattr(modal_app, "require_cpu_worker", lambda config: None)
    monkeypatch.setattr(
        modal_app,
        "validate_training_resume",
        lambda plan, run_root, *, seed: events.append("validate"),
    )
    monkeypatch.setattr(modal_app, "confirm_paid", lambda *args, **kwargs: events.append("confirm"))

    def unexpected_new_run(*args: object, **kwargs: object) -> Path:
        raise AssertionError("resume allocated a new run directory")

    monkeypatch.setattr(modal_app, "prepare_run_directory", unexpected_new_run)
    volume = FakeVolume()
    monkeypatch.setattr(modal_app.modal.Volume, "ephemeral", lambda **kwargs: nullcontext(volume))
    monkeypatch.setattr(
        modal_app, "create_app", lambda *args, **kwargs: (SimpleNamespace(run=lambda: nullcontext()), {})
    )
    executed: list[Path] = []

    def fake_execute(
        _plan: TransferPlan,
        _runtime: RuntimeConfig,
        _adapter: object,
        _runner: object,
        run_root: Path,
        **kwargs: object,
    ) -> dict[str, Any]:
        executed.append(run_root)
        return {"mode": "train", "groups": 0, "cursor": 0}

    monkeypatch.setattr(modal_app, "execute_plan", fake_execute)
    result = modal_app.train_remote(config, "synthetic", assume_yes=True, seed=7, resume=source.name)
    assert result == source
    assert executed == [source]
    assert events == ["validate", "confirm"]
    assert (source / "metrics.json").is_file()

    recorded = json.loads((source / "run-config.json").read_text())
    recorded["run_id"] = "different-run"
    (source / "run-config.json").write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="resume run identity"):
        modal_app.train_remote(config, "synthetic", assume_yes=True, seed=7, resume=source.name)
    assert events == ["validate", "confirm"]


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


def test_modal_encoding_transfers_only_missing_text_and_encoded_parts(tmp_path: Path) -> None:
    pytest.importorskip("modal")
    from ced.modal_app import ModalFunctionRunner

    missing = tmp_path / "missing.parquet"
    missing.write_bytes(b"texts")
    plan = TransferPlan(mode="encode", root=tmp_path, shared={"missing.parquet": missing}, shared_bytes=5)
    volume = FakeVolume()
    payloads: list[dict[str, Any]] = []
    outputs = {"vectors/part-000000.safetensors": b"vectors", "index/part-000000.parquet": b"index"}

    def remote_gen(payload: dict[str, Any], _volume: Any):
        payloads.append(payload)
        assert volume.files["dataset/missing.parquet"] == b"texts"
        for name, content in outputs.items():
            volume.files[f"outputs/encode/{name}"] = content
        yield {
            "groups": 0,
            "input_row_start": 0,
            "part": 0,
            "rows": 1,
            "dimension": 4,
            "files": list(outputs),
            "checksums": {name: hashlib.sha256(content).hexdigest() for name, content in outputs.items()},
        }

    runner = ModalFunctionRunner({"encode_text": SimpleNamespace(remote_gen=remote_gen)}, volume)
    adopted: list[dict[str, Any]] = []

    def adopt(_root: Path, descriptor: dict[str, Any]) -> int:
        adopted.append(descriptor)
        return 1

    run_root = tmp_path / "run"
    run_root.mkdir()
    metrics = execute_plan(
        plan,
        runtime(tmp_path),
        volume,
        runner,
        run_root,
        payload_overrides={"ensemble": {"gpu": "L4"}, "start_row": 4, "start_part": 0},
        adopt_encoded=adopt,
    )
    assert metrics["rows"] == 1
    assert payloads[0]["start_row"] == 4
    assert payloads[0]["start_part"] == 0
    assert adopted[0]["input_row_start"] == 0
    assert (run_root / "vectors" / "part-000000.safetensors").read_bytes() == b"vectors"
    assert (run_root / "index" / "part-000000.parquet").read_bytes() == b"index"
    assert not any(key.startswith("outputs") for key in volume.files)


def test_modal_encoding_adopts_parts_before_failure_and_antijoins_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A yielded prefix survives stream failure and is absent from later work.

    Why: this exercises the cloud-cost boundary end to end with real Parquet and
    Safetensors files. The fake generator fails only after its first yield; the
    next request must contain just the unadopted suffix and must not be retried
    blindly by the generic cloud retry wrapper.
    """
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ced.config.models import TextEncodingSettings
    from ced.model.encoding import RandomTextEncoder, TextEmbeddingCache, iter_encode_missing_parts, text_key

    monkeypatch.setattr("ced.model.encoding.PART_ROWS", 2)
    requested = tmp_path / "requested.parquet"
    texts = ["one", "two", "three"]
    pq.write_table(pa.table({"text_sha256": [text_key(text) for text in texts], "text": texts}), requested)
    cache = TextEmbeddingCache(tmp_path / "cache", TextEncodingSettings(provider="random", dimension=4))
    config = runtime(tmp_path, retries=3, retry_base_seconds=0.001)
    attempts = 0
    stream_calls = 0

    def invoke(missing: Path, *, fail: bool):
        nonlocal attempts
        attempts += 1
        volume = FakeVolume()
        plan = TransferPlan(
            mode="encode",
            root=missing.parent,
            shared={"missing.parquet": missing},
            shared_bytes=missing.stat().st_size,
        )
        manifest = cache.manifest()
        staged = tmp_path / f"staged-{attempts}"

        class StreamingRunner:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def call(self, function: str, payload: dict[str, Any]) -> dict[str, Any]:
                raise AssertionError("encoding must use stream, not call")

            def stream(self, function: str, payload: dict[str, Any]):
                nonlocal stream_calls
                assert function == "encode_text"
                stream_calls += 1
                self.calls.append(payload)
                for descriptor in iter_encode_missing_parts(
                    missing,
                    staged,
                    RandomTextEncoder(4),
                    start_row=payload["start_row"],
                    start_part=payload["start_part"],
                ):
                    for name in descriptor["files"]:
                        volume.files[f"outputs/encode/{name}"] = (staged / name).read_bytes()
                    yield descriptor
                    if fail:
                        raise RuntimeError("injected stream failure")

        runner = StreamingRunner()
        run_root = tmp_path / f"run-{attempts}"
        run_root.mkdir()

        def adopt(root: Path, descriptor: dict[str, Any]) -> int:
            return cache.adopt_parts(
                root,
                missing,
                input_row_start=descriptor["input_row_start"],
                rows=descriptor["rows"],
            )

        metrics = execute_plan(
            plan,
            config,
            volume,
            runner,
            run_root,
            payload_overrides={"start_row": manifest["rows"], "start_part": len(manifest["parts"])},
            adopt_encoded=adopt,
        )
        assert not any(key.startswith("outputs") for key in volume.files)
        return runner, metrics

    with pytest.raises(RuntimeError, match="injected stream failure"):
        invoke(requested, fail=True)
    assert stream_calls == 1
    assert cache.manifest()["rows"] == 2

    resumed = tmp_path / "resumed.parquet"
    connection = duckdb.connect()
    try:
        cache.create_index(connection)
        connection.execute(
            f"""COPY (SELECT r.text_sha256, r.text FROM read_parquet('{requested}') r
                    WHERE NOT EXISTS (SELECT 1 FROM cache_index c WHERE c.text_sha256=r.text_sha256)
                    ORDER BY r.text) TO '{resumed}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
        )
    finally:
        connection.close()
    assert pq.read_table(resumed).column("text").to_pylist() == ["three"]

    runner, metrics = invoke(resumed, fail=False)
    assert len(runner.calls) == 1
    assert runner.calls[0]["start_row"] == 2
    assert runner.calls[0]["start_part"] == 1
    assert metrics["rows"] == 1
    assert cache.manifest()["rows"] == 3
    assert pq.read_table(cache.index / "part-000001.parquet").column("row").to_pylist() == [2]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("part", "unexpected encoded part"),
        ("offset", "invalid encoded input slice"),
        ("files", "exactly one vector/index pair"),
        ("checksum_keys", "checksums do not match"),
        ("checksum", "checksum mismatch"),
        ("dimension", "inconsistent dimension"),
        ("adoption", "adopted 0 rows, expected 1"),
    ],
)
def test_modal_encoding_rejects_invalid_part_descriptors(tmp_path: Path, mutation: str, match: str) -> None:
    """Unverified or inconsistent stream metadata cannot trigger adoption.

    Why: descriptors cross a cloud/local trust boundary. Exact pair names prevent
    traversal, offsets and dimensions prevent row corruption, and hashes prevent
    partial downloads from becoming completed cache rows.
    """
    missing = tmp_path / "missing.parquet"
    missing.write_bytes(b"texts")
    plan = TransferPlan(mode="encode", root=tmp_path, shared={"missing.parquet": missing}, shared_bytes=5)
    volume = FakeVolume()
    outputs = {"vectors/part-000000.safetensors": b"vectors", "index/part-000000.parquet": b"index"}
    for name, content in outputs.items():
        volume.files[f"outputs/encode/{name}"] = content
    descriptor: dict[str, Any] = {
        "groups": 0,
        "input_row_start": 0,
        "part": 0,
        "rows": 1,
        "dimension": 4,
        "files": list(outputs),
        "checksums": {name: hashlib.sha256(content).hexdigest() for name, content in outputs.items()},
    }
    if mutation == "part":
        descriptor["part"] = 1
    elif mutation == "offset":
        descriptor["input_row_start"] = 1
    elif mutation == "files":
        descriptor["files"] = ["../escape", "index/part-000000.parquet"]
    elif mutation == "checksum_keys":
        descriptor["checksums"] = {}
    elif mutation == "checksum":
        descriptor["checksums"]["vectors/part-000000.safetensors"] = "0" * 64
    elif mutation == "dimension":
        descriptor["dimension"] = 0

    class Runner:
        def call(self, function: str, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("encoding must stream")

        def stream(self, function: str, payload: dict[str, Any]):
            yield descriptor

    adopted = []

    def adopt(_root: Path, _descriptor: dict[str, Any]) -> int:
        adopted.append(descriptor)
        return 0 if mutation == "adoption" else 1

    run_root = tmp_path / "run"
    run_root.mkdir()
    with pytest.raises(ValueError, match=match):
        execute_plan(
            plan,
            runtime(tmp_path),
            volume,
            Runner(),
            run_root,
            payload_overrides={"start_part": 0},
            adopt_encoded=adopt,
        )
    if mutation != "adoption":
        assert not adopted


def test_modal_encoding_requires_adoption_before_transfer(tmp_path: Path) -> None:
    """Encoding without a durable adopter fails before consuming GPU output."""
    missing = tmp_path / "missing.parquet"
    missing.write_bytes(b"texts")
    plan = TransferPlan(mode="encode", root=tmp_path, shared={"missing.parquet": missing}, shared_bytes=5)
    volume = FakeVolume()
    with pytest.raises(ValueError, match="requires a per-part adoption callback"):
        execute_plan(plan, runtime(tmp_path), volume, FakeRunner(volume), tmp_path)
    assert volume.files == {}


def test_empty_text_encoding_returns_before_modal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The direct remote API rejects paid work for an empty missing relation.

    Why: callers should be safe even if they bypass the higher-level semantic
    workflow. Reading only Parquet metadata is enough to avoid confirmation,
    ephemeral volumes, model downloads, and GPU container startup.
    """
    pytest.importorskip("modal")
    import pyarrow as pa
    import pyarrow.parquet as pq

    import ced.modal_app as modal_app
    from ced.config.models import TextEncodingSettings
    from ced.model.encoding import TextEmbeddingCache

    missing = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"text_sha256": pa.array([], pa.string()), "text": pa.array([], pa.string())}), missing)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("empty encoding reached a paid or ephemeral Modal boundary")

    monkeypatch.setattr(modal_app, "require_cpu_worker", lambda config: None)
    monkeypatch.setattr(modal_app, "confirm_paid", forbidden)
    monkeypatch.setattr(modal_app.modal.Volume, "ephemeral", forbidden)
    cache = TextEmbeddingCache(tmp_path / "cache", TextEncodingSettings(provider="random", dimension=4))
    modal_app.encode_text_remote(runtime(tmp_path), missing, cache)
    assert not (tmp_path / "text-modal.partial").exists()


def test_paid_modal_confirmation_requires_explicit_approval(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("modal")
    from ced.modal_app import confirm_paid

    config = runtime(tmp_path)
    plan = TransferPlan(mode="train", root=tmp_path, shared={}, groups=[])
    confirm_paid(plan, config, assume_yes=True)
    assert "Approved by --yes" in capsys.readouterr().out
    confirm_paid(plan, config, assume_yes=True, gpu="L4")
    assert "using L4" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="interactive terminal"):
        confirm_paid(plan, config, assume_yes=False)
