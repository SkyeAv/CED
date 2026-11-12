from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from ced import cli
from ced.config.models import RuntimeConfig
from ced.preflight import require_cpu_worker, require_disk, require_fullmap, require_program
from ced.remote import RemoteExecutor, sync


def runtime(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(remote_project=tmp_path, disk_reserve_gb=0)


def test_remote_executor_builds_ssh_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every `ced` subcommand must reach wenceslaus through one ssh invocation whose
    environment pins the uv cache and worker variables; a drifting command line would
    silently run orchestration on the laptop instead of the CPU worker."""
    calls: list[list[str]] = []

    def record(command: list[str], check: bool = True, env: dict[str, str] | None = None) -> Any:
        calls.append(command)

    monkeypatch.setattr("ced.remote.subprocess.run", record)
    RemoteExecutor(runtime(tmp_path)).run(["check"])
    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "ssh"
    assert argv[1] == "wenceslaus"
    assert f"cd {tmp_path}" in argv[2]
    assert "exec uv run --no-sync ced check" in argv[2]


def test_sync_rsyncs_with_excludes_and_optionally_installs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mirror must exclude credentials, caches and the managed `.ced` artifact root,
    and `--frozen` keeps the remote environment pinned to the committed lockfile."""
    calls: list[list[str]] = []

    def record(command: list[str], check: bool = True, env: dict[str, str] | None = None) -> Any:
        calls.append(command)

    monkeypatch.setattr("ced.remote.subprocess.run", record)
    sync(runtime(tmp_path), install=False)
    assert calls[0][0] == "ssh"
    transfers = [command for command in calls if command[0] == "rsync"]
    assert len(transfers) == 1
    for name in (".git", ".venv", ".ced", ".env", ".tablassert"):
        assert f"--exclude={name}" in transfers[0]
    sync(runtime(tmp_path), install=True)
    installs = [command for command in calls if command[0] == "ssh" and "uv sync" in command[2]]
    assert len(installs) == 1
    assert "uv sync --frozen --group dev --all-extras" in installs[0][2]


def test_sync_refuses_to_run_as_remote_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote worker already sits on the mirror; syncing there would recurse."""
    monkeypatch.setattr(cli, "is_remote_worker", lambda: True)
    with pytest.raises(RuntimeError, match="local orchestration command"):
        cli.sync(install=True)


def test_cli_subcommands_delegate_exact_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI is a thin argument packer; pinning each argv keeps local and remote
    invocations from drifting apart when a flag is renamed."""
    recorded: list[list[str]] = []
    monkeypatch.setattr("ced.remote.RemoteExecutor.run", lambda self, arguments: recorded.append(list(arguments)))
    monkeypatch.setattr(cli, "is_remote_worker", lambda: False)

    cli.build(source=["chembl"], group=["translator"], fullmap_path=Path("/maps/fullmap"))
    cli.fetch(source=["chembl"], group=["translator"], all_enabled=True, yes=True)
    cli.prepare("snapshot-1")
    cli.encode_text("dataset-1", provider="random", dimension=8)
    cli.encode_attributes("dataset-1", provider="random", dimension=8)
    cli.encode_publications("dataset-1", download=True, yes=True)
    cli.train("dataset-1", epochs=2, seed=7, resume="run-9", yes=True)
    cli.evaluate("run-9", split="validation", ranking="sampled", seed=7, yes=True)
    cli.config_validate()
    cli.check()

    assert recorded == [
        ["build", "--source", "chembl", "--group", "translator", "--fullmap-path", "/maps/fullmap"],
        ["fetch", "--source", "chembl", "--group", "translator", "--all-enabled", "--yes"],
        ["prepare", "snapshot-1"],
        ["encode-text", "dataset-1", "--provider", "random", "--dimension", "8"],
        ["encode-attributes", "dataset-1", "--provider", "random", "--dimension", "8"],
        ["encode-publications", "dataset-1", "--download", "--yes"],
        ["train", "dataset-1", "--epochs", "2", "--seed", "7", "--resume", "run-9", "--yes"],
        ["evaluate", "run-9", "--split", "validation", "--ranking", "sampled", "--seed", "7", "--yes"],
        ["config", "validate"],
        ["check"],
    ]


def test_remote_worker_executes_quality_gate_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """As a remote worker, `ced check` must call the gate directly instead of ssh-ing
    back into the machine it is already running on."""
    quality: list[RuntimeConfig] = []
    monkeypatch.setattr(cli, "is_remote_worker", lambda: True)
    monkeypatch.setattr("ced.remote.run_quality_gate", lambda runtime: quality.append(runtime))
    cli.check()
    assert len(quality) == 1


def test_sources_list_prints_declarative_rows(capsys: pytest.CaptureFixture[str]) -> None:
    cli.sources_list()
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines
    for line in lines:
        fields = line.split("\t")
        assert len(fields) == 5
        assert fields[-1] in {"enabled", "disabled"}
        assert fields[1]
        assert fields[2]


def test_require_cpu_worker_rejects_foreign_hosts_and_worker_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime(Path("/tmp"))
    monkeypatch.setattr(socket, "gethostname", lambda: "laptop")
    with pytest.raises(RuntimeError, match="CPU stages must run on wenceslaus"):
        require_cpu_worker(config)
    monkeypatch.setattr(socket, "gethostname", lambda: "wenceslaus")
    with pytest.raises(RuntimeError, match="may not exceed four"):
        require_cpu_worker(config.model_copy(update={"cpu_workers": 5}))


def test_require_program_disk_and_fullmap_guards(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="required program is unavailable"):
        require_program("ced-definitely-missing-program")
    require_disk(tmp_path / "artifacts", 1, 0)
    with pytest.raises(RuntimeError, match="insufficient disk"):
        require_disk(tmp_path, 10**18, 1)
    with pytest.raises(RuntimeError, match="fullmap does not exist"):
        require_fullmap(tmp_path / "absent-fullmap")
