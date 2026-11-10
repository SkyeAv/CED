from pathlib import Path

import pytest

from ced.config.loader import _expand_env
from ced.config.models import RuntimeConfig, SplitSettings
from ced.preflight import require_cpu_worker
from ced.remote import RemoteExecutor


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
