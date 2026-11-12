from __future__ import annotations

import itertools
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.client import HTTPResponse
from pathlib import Path

from ced.acquisition.base import (
    AcquisitionBackend,
    FetchedArtifact,
    atomic_write_json,
    cached_or_verify,
    verify_artifact,
)
from ced.config.models import RuntimeConfig
from ced.config.sources import ArtifactSpec, HttpAcquisition, SourceConfig

LOGGER = logging.getLogger(__name__)
_PART_SUFFIX = ".part"
_STATE_SUFFIX = ".part.state.json"
_STATE_VERSION = 1
_BLOCK_BYTES = 4 * 1024 * 1024
_DEFAULT_SEGMENT_BYTES = 16 * 1024 * 1024
_TIMEOUT_SECONDS = 60.0
_USER_AGENT = "ced-http-backend/1"


def redact_url(url: str) -> str:
    """Strip the query string: pre-signed redirect targets carry credentials that must never reach a log."""
    return url.split("?", 1)[0]


class HttpDownloadError(RuntimeError):
    """A transfer cannot complete; the ``.part`` file and its state are retained for resume."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(slots=True)
class _Segment:
    start: int
    end: int
    done: bool = False

    @property
    def size(self) -> int:
        return self.end - self.start + 1


@dataclass(slots=True)
class _Transfer:
    url: str
    fd: int
    partial: Path
    total: int
    last_modified: str | None
    segments: list[_Segment]


class HttpBackend(AcquisitionBackend):
    """Dependency-free segmented downloader.

    The release host answers ranged requests with a redirect to a short-lived pre-signed S3
    URL, so every request resolves its own redirect, each segment writes straight into one
    preallocated ``.part`` file with ``os.pwrite``, and a sidecar state file records completed
    segments for resume. ``aria2c`` is not installed on the CPU worker, which is why this
    backend exists at all; it keeps the SourceBackend contract of the aria2 backend, so
    verification, cache hits and the cold-cache instruction behave identically. Backends are
    shared across the acquisition thread pool, so mutable state is guarded by a lock.
    """

    def __init__(self, runtime: RuntimeConfig, source: SourceConfig) -> None:
        self.runtime = runtime
        self.source = source
        acquisition = source.acquisition
        if not isinstance(acquisition, HttpAcquisition):
            raise TypeError("HttpBackend requires http acquisition")
        self.acquisition = acquisition
        self.lock = threading.Lock()

    def destination(self, artifact: ArtifactSpec) -> Path:
        return self.runtime.artifact_path("downloads", self.source.name, artifact.path)

    def origin(self, artifact: ArtifactSpec) -> str:
        return redact_url(artifact.urls[0]) if artifact.urls else "local-cache"

    def fetch(self, artifact: ArtifactSpec, *, allow_download: bool) -> FetchedArtifact:
        destination = self.destination(artifact)
        origin = self.origin(artifact)
        cached = cached_or_verify(destination, self.source, artifact, origin)
        if cached:
            return cached
        if not allow_download:
            raise FileNotFoundError(f"{destination}; run: ced fetch --source {self.source.name}")
        if not artifact.urls:
            raise ValueError(f"http artifact has no URLs: {artifact.path}")
        partial = destination.with_name(f"{destination.name}{_PART_SUFFIX}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = ""
        for attempt in (1, 2):
            self._download(artifact, partial)
            try:
                digest = verify_artifact(partial, artifact)
            except ValueError as error:
                # A stale segment state can survive an upstream rewrite with an unchanged
                # Last-Modified; verify everything once from scratch before giving up.
                if attempt == 2:
                    raise ValueError(f"downloaded artifact failed verification twice: {error}") from error
                LOGGER.warning("download failed verification (%s); restarting from a clean partial: %s", error, origin)
                self._discard(partial)
                continue
            break
        partial.replace(destination)
        self._unlink_state(destination)
        return FetchedArtifact(
            source=self.source.name,
            role=artifact.role,
            path=destination,
            sha256=digest,
            size=destination.stat().st_size,
            origin=origin,
        )

    def _download(self, artifact: ArtifactSpec, partial: Path) -> None:
        url = artifact.urls[0]
        response = self._probe(url)
        try:
            if response is None:
                total, ranged, last_modified = 0, True, None
            else:
                total, ranged, last_modified = self._describe(response, url)
                if ranged:
                    response.close()
            if artifact.size is not None and artifact.size != total:
                raise ValueError(
                    f"size mismatch for {redact_url(url)}: pinned {artifact.size}, server reported {total}"
                )
            segments = self._plan(total, ranged)
            self._resume(partial, url, total, last_modified, segments)
            with os.fdopen(os.open(partial, os.O_RDWR | os.O_CREAT, 0o644), "rb+") as handle:
                handle.truncate(total)
                if ranged:
                    self._write_state(partial, total, last_modified, segments)
                    if segments:
                        transfer = _Transfer(url, handle.fileno(), partial, total, last_modified, segments)
                        with ThreadPoolExecutor(max_workers=min(len(segments), self._connections())) as pool:
                            futures = [
                                pool.submit(self._fetch_segment, transfer, index) for index in range(len(segments))
                            ]
                            for future in futures:
                                future.result()
                elif total and response is not None:
                    self._copy_stream(response, handle.fileno(), 0, total, label="stream")
                os.fsync(handle.fileno())
        finally:
            if response is not None:
                response.close()

    def _probe(self, url: str) -> HTTPResponse | None:
        """Open the transfer probe with the same retry budget as a segment.

        A 416 answer to `bytes=0-0` means the object is empty. Permanent 4xx answers fail
        immediately; 408/429 and 5xx answers are transient and retried with backoff.
        """
        attempts = self.acquisition.max_tries
        for attempt in range(1, attempts + 1):
            try:
                return self._open_probe(url)
            except HttpDownloadError as error:
                if attempt == attempts or (
                    error.status is not None and 400 <= error.status < 500 and error.status not in (408, 429)
                ):
                    raise
                wait = self.acquisition.retry_wait_seconds * 2 ** (attempt - 1)
                LOGGER.warning("probe attempt %d/%d failed (%s); retrying in %.1fs", attempt, attempts, error, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def _open_probe(self, url: str) -> HTTPResponse | None:
        try:
            return self._open(url, "bytes=0-0")
        except HttpDownloadError as error:
            if error.status == 416:
                return None
            raise

    def _open(self, url: str, range_header: str | None = None) -> HTTPResponse:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **self.acquisition.headers})
        if range_header:
            request.add_header("Range", range_header)
        try:
            response = urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as error:
            raise HttpDownloadError(
                f"origin {redact_url(url)} refused the request with HTTP {error.code}: {error.reason}",
                status=error.code,
            ) from error
        except urllib.error.URLError as error:
            raise HttpDownloadError(f"origin {redact_url(url)} is unreachable: {error.reason}") from error
        assert isinstance(response, HTTPResponse)
        return response

    def _describe(self, response: HTTPResponse, url: str) -> tuple[int, bool, str | None]:
        """Read one probe response into a transfer plan; non-ranged bodies stay unread."""
        last_modified = response.headers.get("Last-Modified")
        if response.status == 206:
            header = response.headers.get("Content-Range", "")
            marker = header.rsplit("/", 1)[-1].strip()
            if not marker.isdigit():
                raise HttpDownloadError(f"origin {redact_url(url)} sent an unusable Content-Range: {header}")
            response.read(1)
            return int(marker), True, last_modified
        if response.status != 200:
            raise HttpDownloadError(f"origin {redact_url(url)} answered HTTP {response.status}", status=response.status)
        length = response.headers.get("Content-Length", "")
        if not length.isdigit():
            raise HttpDownloadError(f"origin {redact_url(url)} reported no usable Content-Length: {length!r}")
        return int(length), False, last_modified

    def _connections(self) -> int:
        return self.acquisition.connections or self.runtime.download_connections

    def _segment_bytes(self) -> int:
        return self.acquisition.segment_bytes or _DEFAULT_SEGMENT_BYTES

    def _plan(self, total: int, ranged: bool) -> list[_Segment]:
        """Split into at most `connections` segments, each no smaller than the segment floor."""
        if not total or not ranged:
            return []
        count = max(1, min(self._connections(), total // self._segment_bytes()))
        bounds = [round(total * index / count) for index in range(count + 1)]
        return [_Segment(start, end - 1) for start, end in itertools.pairwise(bounds) if end > start]

    def _state_path(self, partial: Path) -> Path:
        return partial.with_name(f"{partial.name}.state.json")

    def _discard(self, partial: Path) -> None:
        self._state_path(partial).unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    def _unlink_state(self, destination: Path) -> None:
        destination.with_name(f"{destination.name}{_STATE_SUFFIX}").unlink(missing_ok=True)

    def _resume(self, partial: Path, url: str, total: int, last_modified: str | None, segments: list[_Segment]) -> None:
        state_path = self._state_path(partial)
        if not state_path.is_file():
            return
        try:
            document = json.loads(state_path.read_text(encoding="utf-8"))
            compatible = (
                document.get("version") == _STATE_VERSION
                and document.get("size") == total
                and document.get("last_modified") == last_modified
                and len(document.get("segments", [])) == len(segments)
            )
            records = iter(document.get("segments", []))
            if compatible:
                for segment, record in zip(segments, records, strict=True):
                    segment.done = (
                        bool(record.get("done"))
                        and record.get("start") == segment.start
                        and record.get("end") == segment.end
                    )
            if not compatible or any(segment.done for segment in segments):
                LOGGER.info(
                    "resuming %s with %d of %d segments complete",
                    redact_url(url),
                    sum(segment.done for segment in segments),
                    len(segments),
                )
            if not compatible:
                self._discard(partial)
        except (ValueError, AttributeError, TypeError):
            LOGGER.warning("discarding unreadable download state for %s", redact_url(url))
            self._discard(partial)

    def _fetch_segment(self, transfer: _Transfer, index: int) -> None:
        segment = transfer.segments[index]
        if segment.done:
            return
        attempts = self.acquisition.max_tries
        error: Exception | None = None
        for attempt in range(1, attempts + 1):
            response: HTTPResponse | None = None
            try:
                response = self._open(transfer.url, f"bytes={segment.start}-{segment.end}")
                if response.status != 206:
                    raise HttpDownloadError(
                        f"segment {index} lost range support: HTTP {response.status}", status=response.status
                    )
                self._copy_stream(response, transfer.fd, segment.start, segment.size, label=f"segment {index}")
            except (HttpDownloadError, OSError) as failure:
                error = failure
                if response is not None:
                    response.close()
                if attempt == attempts:
                    break
                wait = self.acquisition.retry_wait_seconds * 2 ** (attempt - 1)
                LOGGER.warning(
                    "segment %d attempt %d/%d failed (%s); retrying in %.1fs", index, attempt, attempts, failure, wait
                )
                time.sleep(wait)
                continue
            response.close()
            with self.lock:
                segment.done = True
                self._write_state(transfer.partial, transfer.total, transfer.last_modified, transfer.segments)
            return
        raise HttpDownloadError(
            f"segment {index} of {redact_url(transfer.url)} failed after {attempts} attempts: {error}; "
            f"resumable partial retained at {transfer.partial}"
        )

    def _write_state(self, partial: Path, total: int, last_modified: str | None, segments: list[_Segment]) -> None:
        atomic_write_json(
            self._state_path(partial),
            {
                "version": _STATE_VERSION,
                "size": total,
                "last_modified": last_modified,
                "segments": [
                    {"start": segment.start, "end": segment.end, "done": segment.done} for segment in segments
                ],
            },
        )

    @staticmethod
    def _copy_stream(response: HTTPResponse, fd: int, offset: int, expected: int, *, label: str) -> None:
        """Write one HTTP body at `offset` with bounded memory; a short or long body fails loudly."""
        written = 0
        while block := response.read(_BLOCK_BYTES):
            position = offset + written
            data = memoryview(block)
            while data:
                count = os.pwrite(fd, data, position)
                position += count
                data = data[count:]
            written += len(block)
        if written != expected:
            raise HttpDownloadError(f"{label} transferred {written} bytes but expected {expected}")
