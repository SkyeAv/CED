from __future__ import annotations

import subprocess

from ced.acquisition.base import AcquisitionBackend, FetchedArtifact, cached_or_verify, verify_artifact
from ced.acquisition.retry import retry_transient
from ced.config.models import RuntimeConfig
from ced.config.sources import Aria2Acquisition, ArtifactSpec, SourceConfig
from ced.preflight import require_program


class Aria2Backend(AcquisitionBackend):
    def __init__(self, runtime: RuntimeConfig, source: SourceConfig) -> None:
        self.runtime = runtime
        self.source = source
        acquisition = source.acquisition
        if not isinstance(acquisition, Aria2Acquisition):
            raise TypeError("Aria2Backend requires aria2 acquisition")
        self.acquisition = acquisition

    def destination(self, artifact: ArtifactSpec):
        return self.runtime.artifact_path("downloads", self.source.name, artifact.path)

    def fetch(self, artifact: ArtifactSpec, *, allow_download: bool) -> FetchedArtifact:
        destination = self.destination(artifact)
        origin = artifact.urls[0] if artifact.urls else "local-cache"
        cached = cached_or_verify(destination, self.source, artifact, origin)
        if cached:
            return cached
        if not allow_download:
            raise FileNotFoundError(f"{destination}; run: ced fetch --source {self.source.name}")
        if not artifact.urls:
            raise ValueError(f"aria2 artifact has no URLs: {artifact.path}")
        aria2 = require_program("aria2c")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(f"{destination.suffix}.part")
        connections = self.acquisition.connections or self.runtime.download_connections
        attempts = self.acquisition.max_tries or self.runtime.download_retries
        wait = self.acquisition.retry_wait_seconds or self.runtime.download_retry_wait_seconds
        command = [
            aria2,
            "--continue=true",
            "--check-integrity=true",
            "--auto-file-renaming=false",
            "--allow-overwrite=false",
            "--file-allocation=none",
            f"--max-connection-per-server={connections}",
            f"--split={connections}",
            "--min-split-size=16M",
            f"--max-tries={attempts}",
            f"--retry-wait={round(wait)}",
            "--connect-timeout=30",
            "--timeout=60",
            f"--dir={partial.parent}",
            f"--out={partial.name}",
        ]
        command.extend(f"--header={name}: {value}" for name, value in self.acquisition.headers.items())
        command.extend(artifact.urls)
        try:
            # aria2 retries per connection; the outer pass rides out host outages that
            # exhaust those retries, resuming from aria2's own control file.
            retry_transient(
                lambda: subprocess.run(command, check=True),
                attempts=attempts,
                base_wait=wait,
                label=f"aria2 download of {artifact.path}",
                transient=lambda error: isinstance(error, subprocess.CalledProcessError),
            )
        except subprocess.CalledProcessError:
            raise RuntimeError(
                f"aria2 download failed after {attempts} passes; resumable partial retained at {partial}"
            ) from None
        digest = verify_artifact(partial, artifact)
        partial.replace(destination)
        partial.with_suffix(f"{partial.suffix}.aria2").unlink(missing_ok=True)
        return FetchedArtifact(
            source=self.source.name,
            role=artifact.role,
            path=destination,
            sha256=digest,
            size=destination.stat().st_size,
            origin=origin,
        )
