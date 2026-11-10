from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from ced.config.loader import _expand_env, read_yaml
from ced.config.models import ConfigBase


class ArtifactSpec(ConfigBase):
    path: str
    role: str
    urls: list[str] = Field(default_factory=list)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size: int | None = Field(default=None, ge=0)
    required: bool = True
    archive: ArchiveSpec | None = None


class ArchiveSpec(ConfigBase):
    format: Literal["tar.zst"]
    members: dict[str, list[str]]
    max_members: int = Field(default=1024, gt=0)
    max_member_bytes: int = Field(default=100 * 1024**3, gt=0)
    max_total_bytes: int = Field(default=500 * 1024**3, gt=0)


class ArtifactInventorySpec(ConfigBase):
    manifest: str
    format: Literal["amalgam_compact"]
    include: list[str]
    roles: dict[str, str]

    def materialize(self, document: dict[str, Any]) -> list[ArtifactSpec]:
        if self.format != "amalgam_compact":
            raise ValueError(f"unsupported inventory format: {self.format}")
        if document.get("schema_version") != 1 or document.get("encoding") != "global-int32-uuid-binary16-zstd9":
            raise ValueError("unsupported Amalgam compact manifest")
        if document.get("lossless") is not True:
            raise ValueError("Amalgam compact manifest is not validated as lossless")
        compact = document.get("compact", {})
        hashes = compact.get("sha256", {})
        counts = compact.get("row_counts", {})
        if compact.get("parquet_files") != len(hashes) or set(hashes) != set(counts):
            raise ValueError("Amalgam compact inventory hashes/counts disagree")
        artifacts: list[ArtifactSpec] = []
        for prefix in self.include:
            matches = sorted(path for path in hashes if path.startswith(prefix))
            if not matches:
                raise ValueError(f"Amalgam inventory has no artifacts for {prefix}")
            role = self.roles.get(prefix, prefix.rstrip("/"))
            artifacts.extend(ArtifactSpec(path=path, role=role, sha256=hashes[path]) for path in matches)
        return artifacts


class HuggingFaceAcquisition(ConfigBase):
    backend: Literal["huggingface"]
    repo_id: str
    revision: str = Field(min_length=40, max_length=40)
    fast_transfer: bool = True


class Aria2Acquisition(ConfigBase):
    backend: Literal["aria2"]
    connections: int | None = Field(default=None, ge=1, le=32)
    headers: dict[str, str] = Field(default_factory=dict)


class HttpAcquisition(ConfigBase):
    backend: Literal["http"]
    connections: int | None = Field(default=None, ge=1, le=32)
    headers: dict[str, str] = Field(default_factory=dict)
    segment_bytes: int | None = Field(default=None, ge=1024, le=1024**3)
    max_tries: int = Field(default=5, ge=1, le=10)
    retry_wait_seconds: float = Field(default=5, gt=0, le=300)


class LocalAcquisition(ConfigBase):
    backend: Literal["local"]
    root: Path


AcquisitionConfig = Annotated[
    HuggingFaceAcquisition | Aria2Acquisition | HttpAcquisition | LocalAcquisition, Field(discriminator="backend")
]


class ProvenanceConfig(ConfigBase):
    title: str
    description: str = ""
    homepage: str | None = None
    license_name: str
    license_url: str | None = None
    provided_by: str
    citations: list[str] = Field(default_factory=list)


class AdapterMapping(ConfigBase):
    predicate_overrides: dict[str, str] = Field(default_factory=dict)
    schema_relations: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class SourceConfig(ConfigBase):
    schema_version: int = 1
    name: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    enabled: bool = True
    adapter: str
    groups: list[str] = Field(default_factory=list)
    acquisition: AcquisitionConfig
    provenance: ProvenanceConfig
    graph: Path | None = None
    artifacts: list[ArtifactSpec] = Field(default_factory=list)
    inventory: ArtifactInventorySpec | None = None
    mapping: AdapterMapping = Field(default_factory=AdapterMapping)

    @model_validator(mode="after")
    def has_artifacts(self) -> SourceConfig:
        if not self.artifacts and self.inventory is None:
            raise ValueError("source requires artifacts or an inventory")
        if self.inventory and self.inventory.manifest not in {artifact.path for artifact in self.artifacts}:
            raise ValueError("inventory manifest must also be declared as an artifact")
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact paths must be unique")
        invalid_groups = [group for group in self.groups if not re.fullmatch(r"[a-z][a-z0-9-]*", group)]
        if invalid_groups:
            raise ValueError(f"invalid source groups: {', '.join(invalid_groups)}")
        return self


def load_sources(directory: Path) -> dict[str, SourceConfig]:
    sources: dict[str, SourceConfig] = {}
    for path in sorted(directory.glob("*.yaml")):
        source = SourceConfig.model_validate(_expand_env(read_yaml(path)))
        if source.name != path.stem:
            raise ValueError(f"source name {source.name} must match filename {path.name}")
        if source.name in sources:
            raise ValueError(f"duplicate source: {source.name}")
        sources[source.name] = source
    if not sources:
        raise ValueError(f"no source YAML files found in {directory}")
    return sources


def select_sources(
    sources: dict[str, SourceConfig],
    names: list[str] | None,
    *,
    all_enabled: bool = False,
    groups: list[str] | None = None,
    require_explicit: bool = False,
) -> list[SourceConfig]:
    """Resolve exactly one selection mode: explicit names, groups, or every enabled source."""
    modes = [bool(names), bool(groups), all_enabled]
    if sum(modes) > 1:
        raise ValueError("select sources by name, by group, or by --all-enabled, but not more than one")
    if require_explicit and not any(modes):
        raise ValueError("fetch requires --source, --group or --all-enabled")
    if names:
        unknown = sorted(set(names) - set(sources))
        if unknown:
            raise ValueError(f"unknown sources: {', '.join(unknown)}")
        selected = [sources[name] for name in names]
    elif groups:
        declared = {group for source in sources.values() for group in source.groups}
        unknown = sorted(set(groups) - declared)
        if unknown:
            raise ValueError(f"unknown groups: {', '.join(unknown)}")
        wanted = set(groups)
        selected = [source for source in sources.values() if source.enabled and wanted & set(source.groups)]
    else:
        selected = [source for source in sources.values() if source.enabled]
    if not selected:
        raise ValueError("no sources selected")
    disabled = [source.name for source in selected if not source.enabled]
    if disabled:
        raise ValueError(f"disabled sources: {', '.join(disabled)}")
    return selected
