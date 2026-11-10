from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from ced.config.models import RuntimeConfig
from ced.config.sources import ArtifactSpec, SourceConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def verify_artifact(path: Path, artifact: ArtifactSpec) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    if artifact.size is not None and path.stat().st_size != artifact.size:
        raise ValueError(f"size mismatch for {path}: expected {artifact.size}, got {path.stat().st_size}")
    digest = sha256_file(path)
    if artifact.sha256 is not None and digest != artifact.sha256:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return digest


class FetchedArtifact(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    role: str
    path: Path
    sha256: str
    size: int
    origin: str

    @classmethod
    def from_path(cls, source: SourceConfig, artifact: ArtifactSpec, path: Path, origin: str) -> FetchedArtifact:
        digest = verify_artifact(path, artifact)
        return cls(
            source=source.name,
            role=artifact.role,
            path=path,
            sha256=digest,
            size=path.stat().st_size,
            origin=origin,
        )


class AcquisitionBackend(Protocol):
    runtime: RuntimeConfig

    def fetch(self, artifact: ArtifactSpec, *, allow_download: bool) -> FetchedArtifact: ...


def cached_or_verify(path: Path, source: SourceConfig, artifact: ArtifactSpec, origin: str) -> FetchedArtifact | None:
    if not path.is_file():
        return None
    try:
        return FetchedArtifact.from_path(source, artifact, path, origin)
    except ValueError as error:
        raise ValueError(f"cached artifact failed verification: {error}; repair or remove it explicitly") from error


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp-{os.getpid()}")
    with source.open("rb") as reader, temporary.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    temporary.replace(destination)
