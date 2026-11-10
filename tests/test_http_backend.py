from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ced.acquisition.http import HttpBackend
from ced.config.models import RuntimeConfig
from ced.config.sources import ArtifactSpec, HttpAcquisition, ProvenanceConfig, SourceConfig

LAST_MODIFIED = "Sat, 18 Oct 2026 06:34:51 GMT"
PAYLOAD = bytes(range(256)) * 1024  # 262144 bytes; 4 segments of 64 KiB with segment_bytes=65536


@dataclass
class Fixture:
    payload: bytes = PAYLOAD
    failures: dict[str | None, int] = field(default_factory=dict)  # remaining 503s per Range header
    fail_query: bool = False  # fail requests whose URL carries a query string
    ignore_range: bool = False  # answer every Range with the full body
    redirect_with_query: bool = False
    require_token: bool = False
    requests: list[tuple[str | None, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    @property
    def fixture(self) -> Fixture:
        return self.server.fixture  # pyright: ignore[reportAttributeAccessIssue]

    def _reply(self, status: int, body: bytes, extra: dict[str, str] = {}) -> None:  # noqa: B006 - read-only helper
        self.send_response(status)
        for key, value in extra.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        fixture = self.fixture
        range_header = self.headers.get("Range")
        path = self.path.split("?", 1)[0]
        if fixture.require_token and "token=KEEP" not in self.path:
            self._reply(403, b"missing token")
            return
        with fixture.lock:
            fixture.requests.append((range_header, self.path))
            remaining = fixture.failures.get(range_header, 0)
            if remaining:
                fixture.failures[range_header] = remaining - 1
        if remaining:
            self._reply(503, b"transient")
            return
        if fixture.fail_query and "?" in self.path:
            self._reply(500, b"boom")
            return
        if fixture.redirect_with_query and "?" not in self.path:
            self._reply(302, b"", {"Location": f"{path}?AWSAccessKeyId=AKIAEXAMPLE&Signature=SECRET&Expires=1"})
            return
        payload = fixture.payload
        if range_header and not fixture.ignore_range:
            start, _, end = range_header.removeprefix("bytes=").partition("-")
            first, last = int(start), int(end)
            if first >= len(payload):
                self._reply(416, b"", {"Content-Range": f"bytes */{len(payload)}"})
                return
            chunk = payload[first : min(last, len(payload) - 1) + 1]
            self._reply(
                206,
                chunk,
                {
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes {first}-{first + len(chunk) - 1}/{len(payload)}",
                    "Last-Modified": LAST_MODIFIED,
                },
            )
            return
        self._reply(200, payload, {"Accept-Ranges": "bytes", "Last-Modified": LAST_MODIFIED})


@pytest.fixture
def origin() -> Iterator[tuple[Fixture, str]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.fixture = Fixture()  # pyright: ignore[reportAttributeAccessIssue]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.fixture, f"http://127.0.0.1:{server.server_address[1]}/data"  # pyright: ignore[reportAttributeAccessIssue]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def http_backend(tmp_path: Path, spec: ArtifactSpec, *, connections: int | None = 4) -> HttpBackend:
    runtime = RuntimeConfig(remote_project=tmp_path, artifact_root=Path(".ced"), disk_reserve_gb=0)
    source = SourceConfig(
        name="fixture",
        adapter="kgx_ndjson",
        acquisition=HttpAcquisition(
            backend="http", connections=connections, segment_bytes=65536, retry_wait_seconds=0.01
        ),
        provenance=ProvenanceConfig(title="Fixture", license_name="CC0-1.0", provided_by="infores:fixture"),
        artifacts=[spec],
    )
    return HttpBackend(runtime, source)


def artifact(url: str, *, size: int | None = len(PAYLOAD), sha256: str | None = None) -> ArtifactSpec:
    return ArtifactSpec(path="payload.bin", role="nodes", urls=[url], size=size, sha256=sha256)


def test_segmented_download_reproduces_bytes_and_checksum(origin: tuple[Fixture, str], tmp_path: Path) -> None:
    """The whole point of the backend: parallel ranged transfers must still produce the exact
    pinned object, otherwise nothing downstream can trust the cache."""
    fixture, url = origin
    spec = artifact(url, sha256=hashlib.sha256(PAYLOAD).hexdigest())
    fetched = http_backend(tmp_path, spec).fetch(spec, allow_download=True)
    assert fetched.path.read_bytes() == PAYLOAD
    assert fetched.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert fetched.origin == url
    ranges = [request for request in fixture.requests if request[0] and request[0] != "bytes=0-0"]
    assert {request[0] for request in ranges} == {
        "bytes=0-65535",
        "bytes=65536-131071",
        "bytes=131072-196607",
        "bytes=196608-262143",
    }
    assert not fetched.path.with_name(f"{fetched.path.name}.part").exists()
    assert not fetched.path.with_name(f"{fetched.path.name}.part.state.json").exists()


def test_range_ignoring_server_downloads_in_one_stream(origin: tuple[Fixture, str], tmp_path: Path) -> None:
    """A host that answers 200 to the ranged probe must still yield a correct file: the probe
    response itself is streamed as the whole transfer, so no extra request is needed."""
    fixture, url = origin
    fixture.ignore_range = True
    spec = artifact(url, sha256=hashlib.sha256(PAYLOAD).hexdigest())
    fetched = http_backend(tmp_path, spec).fetch(spec, allow_download=True)
    assert fetched.path.read_bytes() == PAYLOAD
    assert fixture.requests == [("bytes=0-0", "/data")]


def test_pinned_size_mismatch_fails_before_any_transfer(origin: tuple[Fixture, str], tmp_path: Path) -> None:
    """A wrong pinned size means the release moved underneath the config; failing before
    transfer avoids downloading hundreds of megabytes that can never verify."""
    fixture, url = origin
    spec = artifact(url, size=7)
    with pytest.raises(ValueError, match="pinned 7, server reported"):
        http_backend(tmp_path, spec).fetch(spec, allow_download=True)
    assert fixture.requests == [("bytes=0-0", "/data")]


def test_transient_failures_retry_per_segment(origin: tuple[Fixture, str], tmp_path: Path) -> None:
    """Pre-signed S3 links and flaky links drop connections; one bad segment must not poison
    the artifact as long as retries can still complete it."""
    fixture, url = origin
    fixture.failures["bytes=0-65535"] = 2
    spec = artifact(url, sha256=hashlib.sha256(PAYLOAD).hexdigest())
    fetched = http_backend(tmp_path, spec).fetch(spec, allow_download=True)
    assert fetched.path.read_bytes() == PAYLOAD
    assert [request for request in fixture.requests if request[0] == "bytes=0-65535"].count(
        ("bytes=0-65535", "/data")
    ) == 3


def test_permanent_failure_retains_part_and_state(origin: tuple[Fixture, str], tmp_path: Path) -> None:
    """When retries run out the partial stays on disk with its plan, so a later run resumes
    instead of restarting a multi-hundred-megabyte transfer from byte zero."""
    fixture, url = origin
    fixture.failures[None] = 0
    for index in range(4):
        fixture.failures[f"bytes={index * 65536}-{index * 65536 + 65535}"] = 99
    spec = artifact(url)
    client = http_backend(tmp_path, spec)
    with pytest.raises(RuntimeError, match="resumable partial retained"):
        client.fetch(spec, allow_download=True)
    partial = client.destination(artifact(url)).with_name("payload.bin.part")
    assert partial.is_file()
    assert partial.stat().st_size == len(PAYLOAD)
    state = json.loads(partial.with_name("payload.bin.part.state.json").read_text(encoding="utf-8"))
    assert state["size"] == len(PAYLOAD)
    assert not any(record["done"] for record in state["segments"])


def test_resume_requests_only_missing_segments(origin: tuple[Fixture, str], tmp_path: Path) -> None:
    """Completed segments recorded in the state file must never be re-requested; that is the
    entire economy of resumable downloads."""
    fixture, url = origin
    spec = artifact(url, sha256=hashlib.sha256(PAYLOAD).hexdigest())
    client = http_backend(tmp_path, spec)
    partial = client.destination(spec).with_name("payload.bin.part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(PAYLOAD[:131072] + b"\x00" * 131072)
    partial.with_name("payload.bin.part.state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "size": len(PAYLOAD),
                "last_modified": LAST_MODIFIED,
                "segments": [
                    {"start": 0, "end": 65535, "done": True},
                    {"start": 65536, "end": 131071, "done": True},
                    {"start": 131072, "end": 196607, "done": False},
                    {"start": 196608, "end": 262143, "done": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    fetched = client.fetch(spec, allow_download=True)
    assert fetched.path.read_bytes() == PAYLOAD
    ranges = {request[0] for request in fixture.requests if request[0]}
    assert ranges == {"bytes=0-0", "bytes=131072-196607", "bytes=196608-262143"}


def test_signed_redirect_query_never_reaches_errors_or_origin(origin: tuple[Fixture, str], tmp_path: Path) -> None:
    """The host redirects to short-lived pre-signed S3 URLs; those query strings are
    credentials, so no log line, error text or manifest record may contain them."""
    fixture, url = origin
    fixture.redirect_with_query = True
    fixture.fail_query = True
    spec = artifact(url)
    with pytest.raises(RuntimeError) as failure:
        http_backend(tmp_path, spec).fetch(spec, allow_download=True)
    assert "AKIAEXAMPLE" not in str(failure.value)
    assert "Signature" not in str(failure.value)


def test_cold_cache_reports_offline_instruction(origin: tuple[Fixture, str], tmp_path: Path) -> None:
    """build must never download implicitly; a missing artifact has to surface the documented
    fetch instruction instead of quietly pulling it."""
    _, url = origin
    spec = artifact(url)
    with pytest.raises(FileNotFoundError, match="ced fetch --source fixture"):
        http_backend(tmp_path, spec).fetch(spec, allow_download=False)


def test_cached_artifact_never_touches_the_network(origin: tuple[Fixture, str], tmp_path: Path) -> None:
    """locate_source runs on every build; a cached artifact must verify locally without a
    single request to the host."""
    fixture, url = origin
    spec = artifact(url, sha256=hashlib.sha256(PAYLOAD).hexdigest())
    client = http_backend(tmp_path, spec)
    first = client.fetch(spec, allow_download=True)
    count = len(fixture.requests)
    second = client.fetch(spec, allow_download=False)
    assert len(fixture.requests) == count
    assert second.sha256 == first.sha256


def test_verification_failure_restarts_once_from_clean_partial(origin: tuple[Fixture, str], tmp_path: Path) -> None:
    """A stale state can survive an upstream rewrite with the same Last-Modified; one clean
    restart must heal it instead of failing forever or serving wrong bytes."""
    fixture, url = origin
    spec = artifact(url, sha256=hashlib.sha256(PAYLOAD).hexdigest())
    client = http_backend(tmp_path, spec)
    partial = client.destination(spec).with_name("payload.bin.part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"\x00" * len(PAYLOAD))
    partial.with_name("payload.bin.part.state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "size": len(PAYLOAD),
                "last_modified": LAST_MODIFIED,
                "segments": [
                    {"start": index * 65536, "end": index * 65536 + 65535, "done": True} for index in range(4)
                ],
            }
        ),
        encoding="utf-8",
    )
    fetched = client.fetch(spec, allow_download=True)
    assert fetched.path.read_bytes() == PAYLOAD
    assert {request[0] for request in fixture.requests if request[0]} == {
        "bytes=0-0",
        "bytes=0-65535",
        "bytes=65536-131071",
        "bytes=131072-196607",
        "bytes=196608-262143",
    }


def test_configured_query_string_is_sent_but_never_recorded(origin: tuple[Fixture, str], tmp_path: Path) -> None:
    """Some hosts require a token in the query string, so the configured URL must reach the
    server verbatim while the recorded origin stays redacted."""
    fixture, url = origin
    fixture.require_token = True
    spec = artifact(f"{url}?token=KEEP", sha256=hashlib.sha256(PAYLOAD).hexdigest())
    fetched = http_backend(tmp_path, spec).fetch(spec, allow_download=True)
    assert fetched.path.read_bytes() == PAYLOAD
    assert fetched.origin == url
    assert all(request[1].endswith("?token=KEEP") for request in fixture.requests)
