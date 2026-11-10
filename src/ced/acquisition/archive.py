from __future__ import annotations

import fnmatch
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import zstandard

from ced.acquisition.base import atomic_write_json, sha256_file
from ced.config.sources import ArchiveSpec

_MANIFEST = ".ced-archive.json"


def _safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive member path: {name}")
    return path


def _existing(destination: Path) -> dict[str, list[Path]] | None:
    manifest = destination / _MANIFEST
    if not manifest.is_file():
        return None
    document = json.loads(manifest.read_text(encoding="utf-8"))
    result: dict[str, list[Path]] = {}
    for role, records in document["members"].items():
        for record in records:
            path = destination / _safe_name(record["path"])
            if not path.is_file() or path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
                raise ValueError(f"extracted archive member failed verification: {path}")
            result.setdefault(role, []).append(path)
    return result


def extract_archive(archive: Path, spec: ArchiveSpec, destination: Path) -> dict[str, list[Path]]:
    existing = _existing(destination)
    if existing is not None:
        return existing
    if destination.exists():
        raise ValueError(f"incomplete archive extraction requires explicit repair: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    members: dict[str, list[dict[str, object]]] = {}
    count = 0
    total = 0
    try:
        with (
            archive.open("rb") as raw,
            zstandard.ZstdDecompressor().stream_reader(raw) as reader,
            tarfile.open(fileobj=reader, mode="r|") as bundle,
        ):
            for member in bundle:
                count += 1
                if count > spec.max_members:
                    raise ValueError("archive member limit exceeded")
                relative = _safe_name(member.name)
                if member.isdir():
                    continue
                if not member.isreg():
                    raise ValueError(f"archive links/devices are forbidden: {member.name}")
                if member.size > spec.max_member_bytes:
                    raise ValueError(f"archive member is too large: {member.name}")
                roles = [
                    role
                    for role, patterns in spec.members.items()
                    if any(fnmatch.fnmatch(member.name, pattern) for pattern in patterns)
                ]
                if not roles:
                    continue
                total += member.size
                if total > spec.max_total_bytes:
                    raise ValueError("archive expanded-size limit exceeded")
                output = temporary / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member: {member.name}")
                with source, output.open("xb") as writer:
                    shutil.copyfileobj(source, writer, length=8 * 1024 * 1024)
                record = {"path": str(relative), "size": member.size, "sha256": sha256_file(output)}
                for role in roles:
                    members.setdefault(role, []).append(record)
        missing = set(spec.members) - set(members)
        if missing:
            raise ValueError(f"archive roles have no members: {', '.join(sorted(missing))}")
        atomic_write_json(temporary / _MANIFEST, {"members": members})
        temporary.replace(destination)
    except Exception:
        raise RuntimeError(f"safe archive extraction failed; temporary files retained at {temporary}") from None
    result = _existing(destination)
    if result is None:
        raise RuntimeError("archive manifest was not written")
    return result
