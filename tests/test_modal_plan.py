from __future__ import annotations

import hashlib
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

    def stream(self, function: str, payload: dict[str, Any]):
        raise AssertionError("stream is only used by encoding test runners")

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
