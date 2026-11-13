from __future__ import annotations

import os
from pathlib import Path

from ced.acquisition.base import AcquisitionBackend, FetchedArtifact
from ced.acquisition.retry import is_transient_huggingface, retry_transient
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
        # hf_hub_download retries transient network faults internally, but an unreliable
        # host or CDN outage outlasts that budget; the outer pass resumes from the hub's
        # own cache and backs off exponentially.
        path = retry_transient(
            lambda: hf_hub_download(
                repo_id=self.acquisition.repo_id,
                repo_type="dataset",
                revision=self.acquisition.revision,
                filename=artifact.path,
                cache_dir=self.cache,
                local_files_only=not allow_download,
            ),
            attempts=self.acquisition.max_tries or self.runtime.download_retries,
            base_wait=self.acquisition.retry_wait_seconds or self.runtime.download_retry_wait_seconds,
            label=f"download {artifact.path} from {self.acquisition.repo_id}",
            transient=is_transient_huggingface,
        )
        origin = f"https://huggingface.co/datasets/{self.acquisition.repo_id}/resolve/{self.acquisition.revision}/{artifact.path}"
        return FetchedArtifact.from_path(self.source, artifact, path=Path(path), origin=origin)
