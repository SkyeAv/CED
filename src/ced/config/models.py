from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModalSettings(ConfigBase):
    gpu: str = "L40S"
    timeout_seconds: int = Field(default=86_400, gt=0)
    single_use_containers: bool = True
    cpu: float = Field(default=8, gt=0)
    memory_mb: int = Field(default=65_536, gt=0)
    max_shared_bytes: int = Field(default=64 * 1024**3, gt=0)
    max_batch_group_bytes: int = Field(default=8 * 1024**3, gt=0)
    retries: int = Field(default=3, ge=1, le=10)
    retry_base_seconds: float = Field(default=2, gt=0)
    base_image: str = "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
    python_version: str = "3.13"
    torch_index_url: str = "https://download.pytorch.org/whl/cu124"


class PreprocessingSettings(ConfigBase):
    # Zero disables pruning. The default removes only singleton and two-node trees.
    isolated_tree_max_nodes: int = Field(default=2, ge=0)


class SplitSettings(ConfigBase):
    seed: int = 20261110
    train: float = Field(default=0.8, gt=0, lt=1)
    validation: float = Field(default=0.1, ge=0, lt=1)
    test: float = Field(default=0.1, ge=0, lt=1)
    negatives: int = Field(default=16, ge=1)

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> SplitSettings:
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to one")
        return self


class TextModelSettings(ConfigBase):
    model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dimension: int = Field(gt=0)


class EnsembleTextEncodingSettings(ConfigBase):
    sapbert: TextModelSettings = Field(
        default_factory=lambda: TextModelSettings(
            model="RenaissanceComputing/SapBERT-PubMedBERT-BabelSynonyms-v2025-01-23",
            revision="a3dfc167497278cd3190e556c4eb717b1cbe293c",
            dimension=768,
        )
    )
    gemma: TextModelSettings = Field(
        default_factory=lambda: TextModelSettings(
            model="google/embeddinggemma-2",
            revision="f9972ac22e696a105c482202646979e33e4b5a1e",
            dimension=768,
        )
    )
    qwen: TextModelSettings = Field(
        default_factory=lambda: TextModelSettings(
            model="Qwen/Qwen3-Embedding-0.6B",
            revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            dimension=1024,
        )
    )
    gpu: str = "L4"
    batch_size: int = Field(default=256, gt=0)
    max_length: int = Field(default=128, gt=0)
    local_files_only: bool = False

    @property
    def dimension(self) -> int:
        return self.sapbert.dimension + self.gemma.dimension + self.qwen.dimension

    @property
    def model_name(self) -> str:
        return "+".join(f"{spec.model}@{spec.revision}" for spec in (self.sapbert, self.gemma, self.qwen))


class TextEncodingSettings(ConfigBase):
    provider: Literal["none", "random", "sentence_transformers", "modal_ensemble"] = "none"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    revision: str | None = None
    dimension: int = Field(default=1024, gt=0)
    batch_size: int = Field(default=64, gt=0)
    max_length: int = Field(default=256, gt=0)
    normalize: bool = True
    local_files_only: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"
    cache_subdirectory: Path = Path("text-embeddings")
    ensemble: EnsembleTextEncodingSettings = Field(default_factory=EnsembleTextEncodingSettings)


class PublicationEmbeddingSettings(ConfigBase):
    repo_id: str = "glusman-team/medcpt-ae"
    revision: str = Field(default="3c923644c682a9664f1409ce07ae4e78b622bcf5", pattern=r"^[0-9a-f]{40}$")
    filename: str = "medcpt-embeddings-128.parquet"
    source_sha256: str = Field(
        default="ef11c7dbb164c7d0ddb86458f3926a3dc4580e26dc139d32dfc32763af0456e8", pattern=r"^[0-9a-f]{64}$"
    )
    size: int = Field(default=20_449_885_518, gt=0)
    dimension: int = Field(default=128, gt=0)
    batch_size: int = Field(default=65_536, gt=0)
    fast_transfer: bool = True


class BatchSettings(ConfigBase):
    node_limit: int = Field(default=4096, gt=0)
    neighbor_fanout: list[int] = Field(default=[16, 8])


class EvaluationSettings(ConfigBase):
    family_prefix: str = "CED:Family_"
    hits: list[int] = Field(default=[1, 3, 10])
    selection_metric: str = "ced_family.mrr"


class RuntimeConfig(ConfigBase):
    schema_version: int = 1
    remote_host: str = "wenceslaus"
    remote_project: Path
    artifact_root: Path = Path(".ced")
    fullmap: Path | None = None
    cpu_workers: int = Field(default=4, ge=1, le=4)
    memory_limit: str = "16GB"
    disk_reserve_gb: float = Field(default=32, ge=0)
    download_connections: int = Field(default=8, ge=1, le=32)
    download_jobs: int = Field(default=4, ge=1, le=4)
    source_directory: Path = Path("src/ced/config/sources")
    graph_config: Path = Path("src/ced/graphs/ced-graph.yaml")
    modal: ModalSettings = Field(default_factory=ModalSettings)
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    split: SplitSettings = Field(default_factory=SplitSettings)
    batch: BatchSettings = Field(default_factory=BatchSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    text_encoding: TextEncodingSettings = Field(default_factory=TextEncodingSettings)
    publication_embeddings: PublicationEmbeddingSettings = Field(default_factory=PublicationEmbeddingSettings)

    @model_validator(mode="before")
    @classmethod
    def empty_fullmap_is_none(cls, value: object) -> object:
        if isinstance(value, dict) and not value.get("fullmap"):
            value["fullmap"] = None
        return value

    @property
    def worker_environment(self) -> dict[str, str]:
        workers = str(self.cpu_workers)
        names = (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "POLARS_MAX_THREADS",
            "RAYON_NUM_THREADS",
        )
        return dict.fromkeys(names, workers) | {
            "CED_REMOTE_WORKER": "1",
            "CED_CPU_WORKERS": workers,
            "CED_DOWNLOAD_JOBS": str(self.download_jobs),
            "PYTHONUTF8": "1",
        }

    def artifact_path(self, *parts: str) -> Path:
        root = self.artifact_root
        if not root.is_absolute():
            root = self.remote_project / root
        return root.joinpath(*parts)


def is_remote_worker() -> bool:
    return os.environ.get("CED_REMOTE_WORKER") == "1"
