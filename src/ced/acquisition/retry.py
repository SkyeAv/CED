"""Shared retry classification for every network-bound transfer.

Unreliable hosts drop connections, truncate bodies, and answer 5xx for stretches at a
time. Every download boundary classifies failures through the same predicates and
retries the transient ones with exponential backoff; permanent failures (missing
entries, auth, other 4xx answers) raise immediately because retrying cannot help.
Backoff bounds come from ``RuntimeConfig.download_retries`` and
``RuntimeConfig.download_retry_wait_seconds``, overridable per source through the
``max_tries`` and ``retry_wait_seconds`` acquisition fields.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from http.client import HTTPException

LOGGER = logging.getLogger(__name__)

# Answers worth another attempt: the origin is overloaded or rate-limiting, not gone.
_TRANSIENT_STATUS = frozenset({408, 429})


class HttpDownloadError(RuntimeError):
    """A transfer cannot complete; the ``.part`` file and its state are retained for resume."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def is_transient(error: Exception) -> bool:
    """Connection-level failures and 408/429/5xx answers deserve another attempt.

    A missing status means the request never completed, which is the common
    unreliable-host failure. Truncated HTTP framing is indistinguishable from a dropped
    connection, and the checksum verification catches either way.
    """
    if isinstance(error, HttpDownloadError):
        return error.status is None or error.status >= 500 or error.status in _TRANSIENT_STATUS
    return isinstance(error, (OSError, HTTPException))


def is_transient_huggingface(error: Exception) -> bool:
    """Classify ``huggingface_hub`` failures: missing entries and offline mode are permanent."""
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, OfflineModeIsEnabled

    if isinstance(error, (EntryNotFoundError, OfflineModeIsEnabled)):
        return False
    if isinstance(error, HfHubHTTPError):
        status = error.response.status_code if error.response is not None else None
        return status is None or status >= 500 or status in _TRANSIENT_STATUS
    from httpx import HTTPError

    if isinstance(error, HTTPError):
        return True
    return is_transient(error)


def retry_transient[T](
    action: Callable[[], T],
    *,
    attempts: int,
    base_wait: float,
    label: str,
    transient: Callable[[Exception], bool] = is_transient,
) -> T:
    """Run ``action`` until it succeeds; transient failures back off exponentially, permanent ones raise at once."""
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as error:
            if attempt == attempts or not transient(error):
                raise
            wait = base_wait * 2 ** (attempt - 1)
            LOGGER.warning("%s: attempt %d/%d failed (%s); retrying in %.1fs", label, attempt, attempts, error, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")
