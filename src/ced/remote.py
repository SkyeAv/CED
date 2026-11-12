from __future__ import annotations

import logging
import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from ced.config import RuntimeConfig
from ced.preflight import require_cpu_worker

LOGGER = logging.getLogger(__name__)
_UV_ENV = (
    "PATH=/home/sgoetz/.local/bin:/home/sgoetz/.nix-profile/bin:$PATH "
    "XDG_DATA_HOME=/local_raid1/sgoetz/.local/share "
    "XDG_CACHE_HOME=/local_raid1/sgoetz/.cache "
    "UV_CACHE_DIR=/local_raid1/sgoetz/.cache/uv"
)
# Machine configuration never syncs between checkouts; each machine keeps its own untracked .envrc.
_SYNC_EXCLUDES = (
    ".git",
    ".venv",
    ".ced",
    ".env",
    ".envrc",
    ".tablassert",
    "plans",
    "__pycache__",
    ".pytest_cache",
)


class RemoteExecutor:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def run(self, arguments: Sequence[str]) -> None:
        environment = " ".join(f"{key}={shlex.quote(value)}" for key, value in self.config.worker_environment.items())
        inner = f"exec uv run --no-sync ced {shlex.join(arguments)}"
        project = shlex.quote(str(self.config.remote_project))
        command = (
            f"cd {project} && export {_UV_ENV} {environment} && exec direnv exec {project} bash -c {shlex.quote(inner)}"
        )
        subprocess.run(["ssh", self.config.remote_host, command], check=True)


def sync(config: RuntimeConfig, install: bool = True) -> None:
    destination = f"{config.remote_host}:{config.remote_project}/"
    subprocess.run(["ssh", config.remote_host, f"mkdir -p {shlex.quote(str(config.remote_project))}"], check=True)
    command = ["rsync", "-az"]
    command.extend(f"--exclude={name}" for name in _SYNC_EXCLUDES)
    command.extend(["./", destination])
    subprocess.run(command, check=True)
    if install:
        remote = (
            f"cd {shlex.quote(str(config.remote_project))} && "
            f"export {_UV_ENV} && uv sync --frozen --group dev --all-extras"
        )
        subprocess.run(["ssh", config.remote_host, remote], check=True)


def run_quality_gate(config: RuntimeConfig) -> None:
    require_cpu_worker(config)
    commands = [
        ["uv", "lock", "--check"],
        ["uv", "run", "--no-sync", "ruff", "check", "."],
        ["uv", "run", "--no-sync", "ruff", "format", "--check", "."],
        ["uv", "run", "--no-sync", "pyright"],
        ["uv", "run", "--no-sync", "pytest", "-n", str(config.cpu_workers)],
    ]
    yaml_files = sorted(
        str(path) for root in (Path("src"), Path("tests")) if root.exists() for path in root.rglob("*.yaml")
    )
    if yaml_files:
        commands.append(
            ["uv", "run", "--no-sync", "yamllint", "--strict", "--config-file", ".yamllint.yaml", *yaml_files]
        )
    environment = os.environ | config.worker_environment
    for command in commands:
        LOGGER.info("remote gate: %s", shlex.join(command))
        command_environment = environment
        if command[:4] == ["uv", "run", "--no-sync", "pytest"]:
            command_environment = environment | {
                name: "1" for name in config.worker_environment if name.endswith("THREADS")
            }
        subprocess.run(command, check=True, env=command_environment)
