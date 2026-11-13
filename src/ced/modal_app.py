from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import modal
import pyarrow.parquet as pq

from ced.acquisition.base import atomic_write_json
from ced.config.models import RuntimeConfig
from ced.modal_plan import (
    FunctionRunner,
    TransferPlan,
    VolumeClient,
    dataset_transfer_plan,
    execute_plan,
    prepare_run_directory,
)
from ced.preflight import require_cpu_worker


class TextCache(Protocol):
    def manifest(self) -> dict[str, Any]: ...

    def adopt_parts(
        self, encoded: Path, missing: Path, *, input_row_start: int = 0, rows: int | None = None
    ) -> int: ...


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


class ModalVolumeAdapter(VolumeClient):
    def __init__(self, volume: Any) -> None:
        self.volume = volume

    def upload(self, files: Mapping[str, Path]) -> None:
        with self.volume.batch_upload(force=True) as batch:
            for remote, local in files.items():
                batch.put_file(str(local), remote)
        self.volume.commit()

    def download(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("wb") as stream:
            self.volume.read_file_into_fileobj(remote, stream)

    def remove(self, remote: str, *, recursive: bool = False) -> None:
        try:
            self.volume.remove_file(remote, recursive=recursive)
            self.volume.commit()
        except FileNotFoundError:
            pass

    def list(self, prefix: str) -> list[str]:
        return [entry.path for entry in self.volume.listdir(prefix, recursive=True)]


class ModalFunctionRunner(FunctionRunner):
    def __init__(self, functions: dict[str, Any], volume: Any) -> None:
        self.functions = functions
        self.volume = volume

    def call(self, function: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.functions[function].remote(payload, self.volume)

    def stream(self, function: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        yield from self.functions[function].remote_gen(payload, self.volume)


def build_image(runtime: RuntimeConfig, *, text: bool = False) -> Any:
    image = (
        modal.Image.from_registry(runtime.modal.base_image, add_python=runtime.modal.python_version)
        .pip_install("uv")
        .pip_install("torch>=2.6", index_url=runtime.modal.torch_index_url)
        .add_local_file("pyproject.toml", "/ced/pyproject.toml", copy=True)
        .add_local_file("README.md", "/ced/README.md", copy=True)
        .add_local_dir("src", "/ced/src", copy=True)
        .pip_install_from_pyproject("pyproject.toml")
        .env({"PYTHONPATH": "/ced/src", "PYTHONUTF8": "1"})
    )
    if text:
        image = image.pip_install("sentence-transformers>=6.1,<7", "transformers>=5.18,<6").env(
            {"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"}
        )
    return image


def create_app(
    runtime: RuntimeConfig, volume: Any, run_name: str, *, include_encoding: bool = False
) -> tuple[Any, dict[str, Any]]:
    app = modal.App(run_name)
    common = {
        "image": build_image(runtime),
        "gpu": runtime.modal.gpu,
        "cpu": runtime.modal.cpu,
        "memory": runtime.modal.memory_mb,
        "timeout": runtime.modal.timeout_seconds,
        "single_use_containers": runtime.modal.single_use_containers,
        "serialized": True,
        "volumes": {"/data": volume},
    }

    def _prepare(payload: dict[str, Any]) -> Path:
        scratch = Path(payload["root"])
        if scratch.exists():
            shutil.rmtree(scratch)
        shutil.copytree("/data/dataset", scratch)
        checkpoint = Path(payload.get("checkpoint_path", "/scratch/checkpoint.pt"))
        source = Path("/data/checkpoint.pt")
        if source.exists():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, checkpoint)
        output = Path(payload["output_prefix"])
        output.mkdir(parents=True, exist_ok=True)
        return scratch

    @app.function(**common)
    def train_group(payload: dict[str, Any], volume: Any) -> dict[str, Any]:
        from ced.model.training import train_batches

        root = _prepare(payload)
        metrics = train_batches(
            root,
            epochs=payload["epochs"],
            seed=payload["seed"],
            checkpoint_path=Path(payload["checkpoint_path"]),
            device=payload.get("device", "cuda"),
            selected_indices=payload["selected_indices"],
            start_cursor=payload["start_cursor"],
            max_batches=payload["max_batches"],
        )
        output = Path(payload["output_prefix"])
        atomic_write_json(output / "metrics.json", metrics)
        checkpoint = Path(payload["checkpoint_path"])
        shutil.copy2(checkpoint, output / "checkpoint.pt")
        for name in ("checkpoint.sha256", "checkpoint.json"):
            source = checkpoint.parent / name
            if source.exists():
                shutil.copy2(source, output / name)
        volume.commit()
        return metrics

    @app.function(**common)
    def evaluate(payload: dict[str, Any], volume: Any) -> dict[str, Any]:
        from ced.model.evaluation import evaluate_dataset

        root = _prepare(payload)
        report = evaluate_dataset(
            root,
            split=payload.get("split", "test"),
            checkpoint=Path(payload["checkpoint_path"]) if payload.get("checkpoint_path") else None,
            seed=payload.get("seed", 20261110),
            device=payload.get("device", "cuda"),
            ranking=payload.get("ranking", "exact"),
        )
        output = Path(payload["output_prefix"])
        atomic_write_json(output / "metrics.json", report)
        volume.commit()
        return report

    functions: dict[str, Any] = {"train_group": train_group, "evaluate": evaluate}
    if include_encoding:
        model_volume = modal.Volume.from_name("ced-text-models-v1", create_if_missing=True)
        encoding_common = common | {
            "image": build_image(runtime, text=True),
            "gpu": runtime.text_encoding.ensemble.gpu,
            "volumes": {"/data": volume, "/hf": model_volume},
        }

        @app.function(**encoding_common)
        def encode_text(payload: dict[str, Any], volume: Any) -> Iterator[dict[str, Any]]:
            from ced.config.models import EnsembleTextEncodingSettings
            from ced.model.encoding import iter_encode_missing_parts
            from ced.model.text_models import create_ensemble_encoder

            root = _prepare(payload)
            settings = EnsembleTextEncodingSettings.model_validate(payload["ensemble"])
            encoder = create_ensemble_encoder(settings, device="cuda")
            model_volume.commit()
            staged = root / "encoded"
            output = Path(payload["output_prefix"])
            for descriptor in iter_encode_missing_parts(
                root / "missing.parquet",
                staged,
                encoder,
                start_row=payload["start_row"],
                start_part=payload["start_part"],
            ):
                for name in descriptor["files"]:
                    source = staged / name
                    destination = output / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    source.unlink()
                volume.commit()
                yield descriptor

        functions["encode_text"] = encode_text
    return app, functions


def confirm_paid(plan: TransferPlan, runtime: RuntimeConfig, *, assume_yes: bool, gpu: str | None = None) -> None:
    message = (
        f"Start a paid Modal GPU {plan.mode} job using {gpu or runtime.modal.gpu}, "
        f"{len(plan.groups)} batch group(s), and {human_size(plan.total_bytes)} of ephemeral transfer storage?"
    )
    if assume_yes:
        print(f"Approved by --yes: {message}")
        return
    if not sys.stdin.isatty():
        raise RuntimeError("paid Modal jobs require an interactive terminal or explicit --yes")
    if input(f"{message}\nType RUN to continue: ").strip() != "RUN":
        raise RuntimeError("Modal job was not approved")


def encode_text_remote(runtime: RuntimeConfig, missing: Path, cache: TextCache, *, assume_yes: bool = False) -> None:
    require_cpu_worker(runtime)
    if pq.ParquetFile(missing).metadata.num_rows == 0:
        return
    settings = runtime.text_encoding.ensemble
    size = missing.stat().st_size
    if size > runtime.modal.max_shared_bytes:
        raise ValueError(f"missing-text input is {size} bytes, exceeding the Modal shared transfer bound")
    plan = TransferPlan(
        mode="encode", root=missing.parent, shared={"missing.parquet": missing}, groups=[], shared_bytes=size
    )
    confirm_paid(plan, runtime, assume_yes=assume_yes, gpu=settings.gpu)
    manifest = cache.manifest()
    run_root = missing.parent / "text-modal.partial"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir()
    try:
        with modal.Volume.ephemeral(version=2) as volume:
            app, functions = create_app(runtime, volume, f"ced-encode-{uuid4().hex[:12]}", include_encoding=True)
            adapter = ModalVolumeAdapter(volume)
            with app.run():
                execute_plan(
                    plan,
                    runtime,
                    adapter,
                    ModalFunctionRunner(functions, volume),
                    run_root,
                    payload_overrides={
                        "ensemble": settings.model_dump(mode="json"),
                        "start_row": manifest["rows"],
                        "start_part": len(manifest["parts"]),
                    },
                    adopt_encoded=lambda root, descriptor: cache.adopt_parts(
                        root,
                        missing,
                        input_row_start=descriptor["input_row_start"],
                        rows=descriptor["rows"],
                    ),
                )
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def train_remote(
    runtime: RuntimeConfig,
    dataset: str,
    *,
    assume_yes: bool = False,
    epochs: int = 1,
    seed: int = 20261110,
    resume: str | None = None,
) -> Path:
    require_cpu_worker(runtime)
    root = runtime.artifact_path("datasets", dataset)
    plan = dataset_transfer_plan(root, runtime, "train")
    confirm_paid(plan, runtime, assume_yes=assume_yes)
    run_root = prepare_run_directory(runtime, dataset, "train")
    if resume:
        source = runtime.artifact_path("runs", resume)
        checkpoint = source / "checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint is missing: {checkpoint}")
        shutil.copy2(checkpoint, run_root / "checkpoint.pt")
        for name in ("checkpoint.sha256", "checkpoint.json", "progress.json"):
            sidecar = source.parent / name
            if sidecar.is_file():
                shutil.copy2(sidecar, run_root / name)
    with modal.Volume.ephemeral(version=2) as volume:
        app, functions = create_app(runtime, volume, f"ced-train-{run_root.name}")
        adapter = ModalVolumeAdapter(volume)
        with app.run():
            metrics = execute_plan(
                plan, runtime, adapter, ModalFunctionRunner(functions, volume), run_root, epochs=epochs, seed=seed
            )
    atomic_write_json(run_root / "metrics.json", metrics)
    return run_root


def evaluate_remote(
    runtime: RuntimeConfig,
    run: str,
    *,
    assume_yes: bool = False,
    split: str = "test",
    seed: int = 20261110,
    ranking: str = "exact",
) -> Path:
    require_cpu_worker(runtime)
    run_root = runtime.artifact_path("runs", run)
    config = json.loads((run_root / "run-config.json").read_text())
    checkpoint = run_root / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"run checkpoint is missing: {checkpoint}")
    root = runtime.artifact_path("datasets", config["dataset"])
    plan = dataset_transfer_plan(root, runtime, "evaluate")
    confirm_paid(plan, runtime, assume_yes=assume_yes)
    with modal.Volume.ephemeral(version=2) as volume:
        app, functions = create_app(runtime, volume, f"ced-evaluate-{run_root.name}")
        adapter = ModalVolumeAdapter(volume)
        with app.run():
            execute_plan(
                plan,
                runtime,
                adapter,
                ModalFunctionRunner(functions, volume),
                run_root,
                seed=seed,
                payload_overrides={"split": split, "ranking": ranking},
            )
    report = json.loads((run_root / "metrics.json").read_text())
    report.update({"split": split, "ranking": ranking})
    atomic_write_json(run_root / "evaluation.json", report)
    return run_root
