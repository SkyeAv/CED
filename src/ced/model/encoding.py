from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from ced.acquisition.base import atomic_write_json, sha256_file
from ced.config.models import RuntimeConfig, TextEncodingSettings
from ced.kg.audit import validate_checksums
from ced.model.schemas import DenseFeatureStore, TensorRef
from ced.preflight import require_cpu_worker


class TextEncoder(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class RandomTextEncoder(nn.Module):
    """Deterministic hashed projection for hermetic tests; it carries no learned semantics."""

    def __init__(self, dimension: int, normalize: bool = True, seed: int = 20261110) -> None:
        super().__init__()
        self.dimension = dimension
        self.normalize = normalize
        self.seed = seed

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        vectors = torch.empty((len(texts), self.dimension), dtype=torch.float32)
        for index, text in enumerate(texts):
            generator = torch.Generator().manual_seed(self.seed ^ int(text_key(text)[:16], 16) & 0x7FFF_FFFF)
            vectors[index] = torch.randn(self.dimension, generator=generator)
        return torch.nn.functional.normalize(vectors, dim=-1) if self.normalize else vectors


class SentenceTransformerTextEncoder(nn.Module):
    """Lazy wrapper around an explicit, user-selected Sentence Transformers model."""

    def __init__(self, settings: TextEncodingSettings) -> None:
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as exception:
            raise ModuleNotFoundError("install the optional text extra: uv sync --extra text") from exception
        device = settings.device if settings.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.settings = settings
        self.model = SentenceTransformer(settings.model, device=device, local_files_only=settings.local_files_only)
        self.model.max_seq_length = settings.max_length
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        if self.dimension != settings.dimension:
            raise ValueError(
                f"configured text dimension {settings.dimension} does not match model dimension {self.dimension}"
            )

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        return self.model.encode(
            list(texts),
            batch_size=self.settings.batch_size,
            normalize_embeddings=self.settings.normalize,
            convert_to_tensor=True,
            show_progress_bar=False,
        )


def create_text_encoder(settings: TextEncodingSettings) -> TextEncoder:
    if settings.provider == "random":
        return RandomTextEncoder(settings.dimension, settings.normalize)
    if settings.provider == "sentence_transformers":
        return SentenceTransformerTextEncoder(settings)
    raise ValueError("text encoding is disabled; choose random or sentence_transformers")


def encoder_key(settings: TextEncodingSettings) -> str:
    payload = settings.model_dump(mode="json", exclude={"cache_subdirectory"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


class TextEmbeddingCache:
    """Persistent content-addressed dense-text store with bounded streaming updates."""

    def __init__(self, root: Path, settings: TextEncodingSettings) -> None:
        self.root = root / encoder_key(settings)
        self.settings = settings
        self.manifest_path = self.root / "manifest.json"
        self.index_path = self.root / "text-index.parquet"
        self.embedding_path = self.root / "embeddings.npy"
        self.root.mkdir(parents=True, exist_ok=True)

    def load_entries(self) -> dict[str, int]:
        if not self.index_path.exists():
            return {}
        table = pq.read_table(self.index_path, columns=["text_sha256", "row"])
        return dict(zip(table["text_sha256"].to_pylist(), table["row"].to_pylist(), strict=True))

    def embed(self, texts: Sequence[str], encoder: TextEncoder) -> tuple[np.ndarray, dict[str, int]]:
        entries = self.load_entries()
        missing: dict[str, str] = {}
        for text in texts:
            digest = text_key(text)
            if digest not in entries:
                missing.setdefault(digest, text)
        if not missing:
            return np.array([entries[text_key(text)] for text in texts], dtype=np.int64), entries
        if self.embedding_path.exists():
            previous = np.load(self.embedding_path, mmap_mode="r", allow_pickle=False)
        else:
            previous = np.zeros((0, encoder.dimension), dtype=np.float32)
        total = int(previous.shape[0]) + len(missing)
        embeddings = np.lib.format.open_memmap(
            self.root / "embeddings.partial.npy", mode="w+", dtype="float32", shape=(total, encoder.dimension)
        )
        embeddings[: int(previous.shape[0])] = previous[:]
        position = int(previous.shape[0])
        batch: list[str] = []
        keys: list[str] = []
        for digest, text in missing.items():
            batch.append(text)
            keys.append(digest)
            if len(batch) >= self.settings.batch_size:
                embeddings[position : position + len(batch)] = encoder.encode(batch).detach().cpu().numpy()
                for offset, key in enumerate(keys):
                    entries[key] = position + offset
                position += len(batch)
                batch, keys = [], []
        if batch:
            embeddings[position : position + len(batch)] = encoder.encode(batch).detach().cpu().numpy()
            for offset, key in enumerate(keys):
                entries[key] = position + offset
        embeddings.flush()
        del previous, embeddings
        atomic_write_json(
            self.manifest_path,
            {
                "provider": self.settings.provider,
                "model": self.settings.model,
                "dimension": encoder.dimension,
                "rows": total,
                "normalize": self.settings.normalize,
            },
        )
        pq.write_table(
            pa.table(
                {"text_sha256": list(entries), "row": [entries[key] for key in entries]},
                schema=pa.schema([("text_sha256", pa.string()), ("row", pa.int64())]),
            ),
            self.index_path,
            compression="zstd",
        )
        (self.root / "embeddings.partial.npy").replace(self.embedding_path)
        return np.array([entries[text_key(text)] for text in texts], dtype=np.int64), entries


def encode_dataset_text(runtime: RuntimeConfig, dataset: str) -> DenseFeatureStore:
    """Encode dataset text features on wenceslaus without re-running graph preparation."""
    require_cpu_worker(runtime)
    root = runtime.artifact_path("datasets", dataset)
    settings = runtime.text_encoding
    manifest = json.loads((root / "manifest.json").read_text())
    validate_checksums(root, manifest["checksums"])
    encoder = create_text_encoder(settings)
    cache = TextEmbeddingCache(runtime.artifact_path(str(settings.cache_subdirectory)), settings)
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={runtime.cpu_workers}")
        connection.execute("SET memory_limit=?", [runtime.memory_limit])
        connection.execute("SET temp_directory=?", [str(root / "spill")])
        node_count = manifest["entities"]
        common = """SELECT owner_index, field_id, ordinal, role, language, text FROM ({}) ORDER BY owner_index, field_id, ordinal"""
        node_query = f"""SELECT e.idx AS owner_index, f.field_id, f.ordinal, 'node' AS role, NULL AS language, f.text
            FROM read_parquet('{root}/node-features.parquet') f
            JOIN read_parquet('{root}/entities.parquet') e ON f.owner_id=e.id
            WHERE f.kind='text'"""
        statement_query = f"""SELECT {node_count}+s.index AS owner_index, f.field_id, f.ordinal, 'statement' AS role, NULL AS language, f.text
            FROM read_parquet('{root}/statement-features.parquet') f
            JOIN read_json('{root}/split-records.ndjson') s ON f.owner_id=s.statement_id
            WHERE f.kind='text'"""
        query = common.format(f"{node_query} UNION ALL BY NAME {statement_query}")
        partial = root / "text-records.partial.parquet"
        connection.execute(f"COPY ({query}) TO '{partial}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        texts = pq.read_table(partial, columns=["text"])["text"].to_pylist()
        rows, _ = cache.embed(texts, encoder)
        records = pq.read_table(partial).append_column("row", pa.array(rows.tolist(), pa.int64()))
        index_path = root / "text-index.parquet"
        pq.write_table(records, index_path, compression="zstd")
        partial.unlink()
        embeddings = np.load(cache.embedding_path, mmap_mode="r", allow_pickle=False)
        store = DenseFeatureStore(
            provider=settings.provider,
            model=settings.model,
            encoder_key=encoder_key(settings),
            dimension=encoder.dimension,
            rows=records.num_rows,
            index_path=str(index_path),
            tensor=TensorRef(
                path=str(cache.embedding_path),
                dtype="float32",
                shape=list(embeddings.shape),
                sha256=sha256_file(cache.embedding_path),
            ),
        )
        atomic_write_json(root / "text-features.json", store.model_dump(mode="json"))
        return store
    finally:
        connection.close()
