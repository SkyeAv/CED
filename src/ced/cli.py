from __future__ import annotations

import logging
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import cyclopts

from ced.config import is_remote_worker, load_runtime

app = cyclopts.App(name="ced", help="Build biomedical KGs and train relational graph autoencoders on wenceslaus/Modal.")
sources_app = cyclopts.App(name="sources", help="Inspect declarative dataset sources.")
config_app = cyclopts.App(name="config", help="Validate runtime and source configuration.")
app.command(sources_app, name="sources")
app.command(config_app, name="config")


def _remote_or_run(arguments: list[str], action: Callable[[], Any]) -> Any:
    if is_remote_worker():
        return action()
    _remote_or_run_executor(arguments)
    return None


def _remote_or_run_executor(arguments: list[str]) -> None:
    from ced.remote import RemoteExecutor

    RemoteExecutor(load_runtime()).run(arguments)


def _many(flag: str, values: list[str] | None) -> list[str]:
    return [item for value in values or [] for item in (flag, value)]


@app.command
def sync(install: bool = True) -> None:
    """Synchronize source files and optionally install the remote environment."""
    if is_remote_worker():
        raise RuntimeError("ced sync is a local orchestration command")
    from ced.remote import sync as sync_remote

    sync_remote(load_runtime(), install=install)


@app.command
def check() -> None:
    """Run every test and quality check on wenceslaus."""

    def action() -> None:
        from ced.remote import run_quality_gate

        run_quality_gate(load_runtime())

    _remote_or_run(["check"], action)


@config_app.command(name="validate")
def config_validate() -> None:
    """Validate runtime and every enabled source on wenceslaus."""

    def action() -> None:
        from ced.config.sources import load_sources
        from ced.preflight import require_cpu_worker

        runtime = load_runtime()
        require_cpu_worker(runtime)
        sources = load_sources(runtime.source_directory)
        print(f"validated runtime and {len(sources)} source configs")

    _remote_or_run(["config", "validate"], action)


@sources_app.command(name="list")
def sources_list() -> None:
    """List configured sources without fetching artifacts."""
    from ced.config.sources import load_sources

    for source in load_sources(load_runtime().source_directory).values():
        print(
            f"{source.name}\t{source.adapter}\t{source.acquisition.backend}\t{'enabled' if source.enabled else 'disabled'}"
        )


@app.command
def fetch(source: list[str] | None = None, all_enabled: bool = False, yes: bool = False) -> None:
    """Fetch only explicitly selected, pinned artifacts on wenceslaus."""
    arguments = [
        "fetch",
        *_many("--source", source),
        *(["--all-enabled"] if all_enabled else []),
        *(["--yes"] if yes else []),
    ]

    def action() -> None:
        from ced.acquisition.pipeline import fetch_sources

        fetch_sources(load_runtime(), source or [], all_enabled=all_enabled, assume_yes=yes)

    _remote_or_run(arguments, action)


@app.command
def build(source: list[str] | None = None, fullmap_path: Path | None = None) -> None:
    """Build and merge selected source graphs without implicit downloads."""
    arguments = [
        "build",
        *_many("--source", source),
        *_many("--fullmap-path", [str(fullmap_path)] if fullmap_path else []),
    ]

    def action() -> None:
        from ced.kg.pipeline import build_graph

        build_graph(load_runtime(), source or [], fullmap_path=fullmap_path)

    _remote_or_run(arguments, action)


@app.command
def prepare(snapshot: str) -> None:
    """Prepare split and tensor shards on wenceslaus."""
    arguments = ["prepare", snapshot]

    def action() -> None:
        import_module("ced.model.prepare").prepare_dataset(load_runtime(), snapshot)

    _remote_or_run(arguments, action)


@app.command
def encode_text(
    dataset: str, provider: str | None = None, model: str | None = None, dimension: int | None = None
) -> None:
    """Encode prepared text features with the configured Sentence Transformers backend."""
    arguments = [
        "encode-text",
        dataset,
        *(["--provider", provider] if provider else []),
        *(["--model", model] if model else []),
        *(["--dimension", str(dimension)] if dimension else []),
    ]

    def action() -> None:
        from ced.config.models import TextEncodingSettings

        runtime = load_runtime()
        updates = {
            name: value
            for name, value in (("provider", provider), ("model", model), ("dimension", dimension))
            if value is not None
        }
        if updates:
            runtime.text_encoding = TextEncodingSettings.model_validate(runtime.text_encoding.model_dump() | updates)
        import_module("ced.model.encoding").encode_dataset_text(runtime, dataset)

    _remote_or_run(arguments, action)


@app.command
def train(dataset: str, epochs: int = 1, seed: int = 20261110, resume: str | None = None, yes: bool = False) -> None:
    """Run approved GPU training on Modal from prepared wenceslaus shards."""
    arguments = [
        "train",
        dataset,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        *(["--resume", resume] if resume else []),
        *(["--yes"] if yes else []),
    ]

    def action() -> None:
        import_module("ced.modal_app").train_remote(
            load_runtime(), dataset, assume_yes=yes, epochs=epochs, seed=seed, resume=resume
        )

    _remote_or_run(arguments, action)


@app.command
def evaluate(
    run: str,
    split: str = "test",
    ranking: Literal["exact", "sampled"] = "exact",
    seed: int = 20261110,
    yes: bool = False,
) -> None:
    """Evaluate a recovered run with CPU preparation and GPU inference."""
    arguments = [
        "evaluate",
        run,
        "--split",
        split,
        "--ranking",
        ranking,
        "--seed",
        str(seed),
        *(["--yes"] if yes else []),
    ]

    def action() -> None:
        import_module("ced.modal_app").evaluate_remote(
            load_runtime(), run, assume_yes=yes, split=split, seed=seed, ranking=ranking
        )

    _remote_or_run(arguments, action)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app()
