from __future__ import annotations

from ced.acquisition.base import AcquisitionBackend, FetchedArtifact
from ced.config.models import RuntimeConfig
from ced.config.sources import ArtifactSpec, LocalAcquisition, SourceConfig


class LocalBackend(AcquisitionBackend):
    def __init__(self, runtime: RuntimeConfig, source: SourceConfig) -> None:
        self.runtime = runtime
        self.source = source
        acquisition = source.acquisition
        if not isinstance(acquisition, LocalAcquisition):
            raise TypeError("LocalBackend requires local acquisition")
        self.acquisition = acquisition

    def fetch(self, artifact: ArtifactSpec, *, allow_download: bool) -> FetchedArtifact:
        del allow_download
        path = self.acquisition.root / artifact.path
        if not path.is_file():
            raise FileNotFoundError(f"local artifact is missing: {path}")
        return FetchedArtifact.from_path(self.source, artifact, path, str(path))
