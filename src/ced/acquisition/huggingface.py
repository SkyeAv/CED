from __future__ import annotations

import os
from pathlib import Path

from ced.acquisition.base import AcquisitionBackend, FetchedArtifact
from ced.config.models import RuntimeConfig
from ced.config.sources import ArtifactSpec, HuggingFaceAcquisition, SourceConfig


class HuggingFaceBackend(AcquisitionBackend):
    def __init__(self, runtime: RuntimeConfig, source: SourceConfig) -> None:
        self.runtime = runtime
        self.source = source
        acquisition = source.acquisition
        if not isinstance(acquisition, HuggingFaceAcquisition):
            raise TypeError("HuggingFaceBackend requires huggingface acquisition")
        self.acquisition = acquisition
        self.cache = runtime.artifact_path("cache", "huggingface")

    def fetch(self, artifact: ArtifactSpec, *, allow_download: bool) -> FetchedArtifact:
        from huggingface_hub import hf_hub_download

        if self.acquisition.fast_transfer:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        path = hf_hub_download(
            repo_id=self.acquisition.repo_id,
            repo_type="dataset",
            revision=self.acquisition.revision,
            filename=artifact.path,
            cache_dir=self.cache,
            local_files_only=not allow_download,
        )
        origin = f"https://huggingface.co/datasets/{self.acquisition.repo_id}/resolve/{self.acquisition.revision}/{artifact.path}"
        return FetchedArtifact.from_path(self.source, artifact, path=Path(path), origin=origin)
