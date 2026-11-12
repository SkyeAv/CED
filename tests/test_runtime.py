import shlex
import tomllib
from pathlib import Path

import pytest

from ced.config.loader import _expand_env
from ced.config.models import RuntimeConfig, SplitSettings
from ced.preflight import require_cpu_worker
from ced.remote import RemoteExecutor, run_quality_gate


def runtime(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(remote_project=tmp_path / "remote", artifact_root=Path(".ced"))


def test_environment_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CED_TEST_VALUE", "configured")
    assert _expand_env("${CED_TEST_VALUE}") == "configured"
    assert _expand_env("${CED_MISSING_VALUE:-fallback}") == "fallback"
    with pytest.raises(ValueError, match="CED_MISSING_VALUE"):
        _expand_env("${CED_MISSING_VALUE}")


def test_worker_environment_caps_all_thread_pools(tmp_path: Path) -> None:
    environment = runtime(tmp_path).worker_environment
    assert environment["CED_REMOTE_WORKER"] == "1"
    assert {key for key in environment if key.endswith("THREADS") or key == "POLARS_MAX_THREADS"}
    assert set(environment.values()) >= {"1", "4"}


def test_remote_command_never_executes_pipeline_locally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr("subprocess.run", lambda command, check: calls.append(command))
    RemoteExecutor(runtime(tmp_path)).run(["fetch", "--source", "chembl"])
    assert len(calls) == 1
    assert calls[0][0:2] == ["ssh", "wenceslaus"]
    assert "uv run --no-sync ced fetch --source chembl" in calls[0][2]
    assert "CED_REMOTE_WORKER=1" in calls[0][2]


def test_cpu_worker_requires_wenceslaus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CED_REMOTE_WORKER", raising=False)
    with pytest.raises(RuntimeError, match="wenceslaus"):
        require_cpu_worker(runtime(tmp_path))


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        SplitSettings(train=0.9, validation=0.2, test=0.1)


def test_pytest_defaults_to_bounded_xdist(tmp_path: Path) -> None:
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    options = shlex.split(pyproject["tool"]["pytest"]["ini_options"]["addopts"])
    workers = runtime(tmp_path).cpu_workers
    assert workers == 4
    assert options == [
        "-m",
        "not network and not gpu and not fullmap and not perf",
        "-n",
        str(workers),
        "--dist",
        "loadfile",
    ]


def test_pre_push_runs_wenceslaus_quality_gate() -> None:
    config = tomllib.loads((Path(__file__).parents[1] / "prek.toml").read_text())
    assert config["default_install_hook_types"] == ["pre-commit", "pre-push"]
    hooks = [hook for repo in config["repos"] for hook in repo.get("hooks", [])]
    gate = next((hook for hook in hooks if hook["id"] == "wenceslaus-check"), None)
    assert gate is not None
    assert gate["entry"] == "env CED_REMOTE_WORKER=1 uv run --no-sync ced check"
    assert gate["pass_filenames"] is False
    assert gate["always_run"] is True
    assert gate["stages"] == ["pre-push"]


def test_quality_gate_uses_pyright_and_bounded_xdist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime(tmp_path).model_copy(update={"cpu_workers": 2})
    calls: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr("ced.remote.require_cpu_worker", lambda runtime: None)

    def record(command: list[str], check: bool, env: dict[str, str]) -> None:
        assert check is True
        calls.append((command, env))

    monkeypatch.setattr("ced.remote.subprocess.run", record)
    run_quality_gate(config)
    commands = [command for command, _ in calls]
    assert ["uv", "run", "--no-sync", "pyright"] in commands
    pytest_command = next(command for command in commands if "pytest" in command)
    assert pytest_command[:6] == ["uv", "run", "--no-sync", "pytest", "-n", "2"]
    assert "--cov=ced" in pytest_command
    assert "--cov-report=term-missing" in pytest_command
    pytest_environment = next(env for command, env in calls if "pytest" in command)
    ruff_environment = next(env for command, env in calls if "ruff" in command)
    assert ruff_environment["RAYON_NUM_THREADS"] == "2"
    assert pytest_environment["CED_CPU_WORKERS"] == "2"
    thread_limits = {name: pytest_environment[name] for name in config.worker_environment if name.endswith("THREADS")}
    assert set(thread_limits.values()) == {"1"}
