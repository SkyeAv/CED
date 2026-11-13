"""Contracts for the shared transient-failure classification behind every download retry."""

from __future__ import annotations

import logging
from http.client import IncompleteRead

import httpx
import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
)

from ced.acquisition.retry import HttpDownloadError, is_transient, is_transient_huggingface, retry_transient


def _hub_error(status: int) -> HfHubHTTPError:
    return HfHubHTTPError(
        f"HTTP {status}", response=httpx.Response(status_code=status, request=httpx.Request("GET", "https://hub"))
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (HttpDownloadError("unreachable"), True),
        (HttpDownloadError("overloaded", status=503), True),
        (HttpDownloadError("throttled", status=429), True),
        (HttpDownloadError("slow", status=408), True),
        (HttpDownloadError("moved", status=404), False),
        (HttpDownloadError("forbidden", status=403), False),
        (ConnectionResetError("reset"), True),
        (TimeoutError("timed out"), True),
        (IncompleteRead(b"truncated", 1024), True),
        (ValueError("bad content"), False),
    ],
)
def test_is_transient(error: Exception, expected: bool) -> None:
    assert is_transient(error) is expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (EntryNotFoundError("absent"), False),
        (LocalEntryNotFoundError("not cached"), False),
        (OfflineModeIsEnabled("offline"), False),
        (_hub_error(404), False),
        (_hub_error(403), False),
        (_hub_error(503), True),
        (_hub_error(429), True),
        (httpx.ConnectError("refused"), True),
        (httpx.ReadTimeout("slow"), True),
    ],
)
def test_is_transient_huggingface(error: Exception, expected: bool) -> None:
    assert is_transient_huggingface(error) is expected


def test_retry_transient_backs_off_and_succeeds(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    waits: list[float] = []
    monkeypatch.setattr("ced.acquisition.retry.time.sleep", waits.append)
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise HttpDownloadError("dropped", status=502)
        return "payload"

    with caplog.at_level(logging.WARNING, logger="ced.acquisition.retry"):
        assert retry_transient(flaky, attempts=4, base_wait=2.0, label="fixture download") == "payload"
    assert len(attempts) == 3
    assert waits == [2.0, 4.0]
    assert len(caplog.records) == 2


def test_retry_transient_stops_immediately_on_permanent_failure() -> None:
    """Retrying a 404 or an auth refusal cannot help, so the error must surface on the first attempt."""
    attempts: list[int] = []

    def permanent() -> str:
        attempts.append(1)
        raise HttpDownloadError("gone", status=404)

    with pytest.raises(HttpDownloadError):
        retry_transient(permanent, attempts=5, base_wait=1.0, label="fixture download")
    assert len(attempts) == 1


def test_retry_transient_exhausts_the_budget_before_raising() -> None:
    attempts: list[int] = []

    def always_transient() -> str:
        attempts.append(1)
        raise HttpDownloadError("dropped", status=503)

    with pytest.raises(HttpDownloadError, match="dropped"):
        retry_transient(always_transient, attempts=3, base_wait=0.0, label="fixture download")
    assert len(attempts) == 3
