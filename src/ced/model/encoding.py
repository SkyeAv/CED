from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

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
from ced.model.tensors import write_tensor
from ced.preflight import require_cpu_worker

PART_ROWS = 65_536


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
        embedding_dimension = self.model.get_sentence_embedding_dimension()
        if embedding_dimension is None:
            raise ValueError(f"configured model {settings.model} does not report an embedding dimension")
        self.dimension = int(embedding_dimension)
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
    """Append-only content-addressed dense-text store.

    Embeddings live in immutable Parquet parts and the sha -> row index in
    matching index parts. Appending N texts writes O(N) bytes; the previous
    single-array design rewrote every existing row on every run, which is
    O(total) per call and O(total^2) across incremental runs.
    """

    def __init__(self, root: Path, settings: TextEncodingSettings) -> None:
        self.root = root / encoder_key(settings)
        self.settings = settings
        self.manifest_path = self.root / "manifest.json"
        self.parts = self.root / "parts"
        self.index = self.root / "index"
        self.parts.mkdir(parents=True, exist_ok=True)
        self.index.mkdir(parents=True, exist_ok=True)

    def manifest(self) -> dict[str, Any]:
        """Return the cache manifest, or an empty one for a cache with no parts yet."""
        if not self.manifest_path.is_file():
            return {
                "provider": self.settings.provider,
                "model": self.settings.model,
                "dimension": self.settings.dimension,
                "rows": 0,
                "normalize": self.settings.normalize,
                "parts": [],
            }
        return json.loads(self.manifest_path.read_text())

    def create_index(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Expose the sha -> row map for the published parts as a temp table.

        Only parts listed in ``manifest.json`` are read. A crash between the index
        part write and the manifest update leaves an orphan file whose rows were
        never published; globbing the directory would treat those texts as cached
        while ``part_refs`` omits their vectors, so a dataset would be written with
        row ids that point at embeddings that do not exist. Ignoring orphans is also
        what lets a crashed run self-heal, because the next append reuses the name.
        Reading an explicit list also avoids DuckDB's failure on an empty glob.
        """
        names = [str(part["name"]) for part in self.manifest()["parts"]]
        if not names:
            connection.execute("CREATE OR REPLACE TEMP TABLE cache_index (text_sha256 VARCHAR, row BIGINT)")
            return
        listing = ", ".join(f"'{self.index / name}'" for name in names)
        connection.execute(
            f"""CREATE OR REPLACE TEMP TABLE cache_index AS
                SELECT text_sha256, min(row) AS row
                FROM read_parquet([{listing}]) GROUP BY text_sha256"""
        )

    def append_stream(self, batches: Iterator[tuple[Sequence[str], Any]]) -> int:
        """Append encoded batches as new parts and return the number of rows added.

        Only the current batch and the part being filled are resident, so peak RSS
        is bounded by PART_ROWS * dimension regardless of dataset size.
        """
        manifest = self.manifest()
        dimension = int(manifest["dimension"])
        row = int(manifest["rows"])
        parts = list(manifest["parts"])
        pending_keys: list[str] = []
        pending: list[np.ndarray] = []
        pending_rows = 0
        added = 0

        def flush() -> None:
            nonlocal row, pending_keys, pending, pending_rows
            if not pending_rows:
                return
            name = f"part-{len(parts):06d}.parquet"
            vectors = np.concatenate(pending) if len(pending) > 1 else pending[0]
            write_tensor(self.parts / name, vectors)
            # Atomic like the vector part: a concurrent reader must never observe a
            # half-written index, and a torn file stays an ignored orphan.
            temporary = self.index / f"{name}.tmp-{os.getpid()}"
            pq.write_table(
                pa.table(
                    {
                        "text_sha256": pa.array(pending_keys, pa.string()),
                        "row": pa.array(np.arange(row, row + pending_rows, dtype=np.int64)),
                    }
                ),
                temporary,
                compression="zstd",
            )
            temporary.replace(self.index / name)
            parts.append({"name": name, "rows": pending_rows})
            row += pending_rows
            pending_keys, pending, pending_rows = [], [], 0

        for texts, values in batches:
            array = np.ascontiguousarray(values, dtype=np.float32)
            if array.ndim != 2 or array.shape[0] != len(texts) or array.shape[1] != dimension:
                raise ValueError(f"encoded batch shape {array.shape} does not match cache dimension {dimension}")
            pending_keys.extend(text_key(text) for text in texts)
            pending.append(array)
            pending_rows += array.shape[0]
            added += array.shape[0]
            if pending_rows >= PART_ROWS:
                flush()
        flush()
        if added:
            atomic_write_json(self.manifest_path, {**manifest, "dimension": dimension, "rows": row, "parts": parts})
        return added

    def part_refs(self) -> list[TensorRef]:
        """One TensorRef per immutable part, in row order."""
        manifest = self.manifest()
        dimension = int(manifest["dimension"])
        references = []
        for part in manifest["parts"]:
            path = self.parts / str(part["name"])
            references.append(
                TensorRef(
                    path=str(path),
                    column="value",
                    dtype="float32",
                    shape=[int(part["rows"]), dimension],
                    sha256=sha256_file(path),
                )
            )
        return references


def _encode_batches(path: Path, encoder: TextEncoder, batch_size: int) -> Iterator[tuple[list[str], np.ndarray]]:
    """Stream the texts that still need encoding in bounded batches."""
    if not path.is_file():
        return
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["text"]):
        texts = batch.column("text").to_pylist()
        yield texts, encoder.encode(texts).detach().cpu().numpy()


def encode_dataset_text(runtime: RuntimeConfig, dataset: str) -> DenseFeatureStore:
    """Encode dataset text features on wenceslaus without re-running graph preparation."""
    require_cpu_worker(runtime)
    root = runtime.artifact_path("datasets", dataset)
    settings = runtime.text_encoding
    manifest = json.loads((root / "manifest.json").read_text())
    validate_checksums(root, manifest["checksums"])
    encoder = create_text_encoder(settings)
    cache = TextEmbeddingCache(runtime.artifact_path(str(settings.cache_subdirectory)), settings)
    partial = root / "text-records.partial.parquet"
    missing = root / "text-missing.partial.parquet"
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
            JOIN read_parquet('{root}/topology.parquet') s ON f.owner_id=s.statement_id
            WHERE f.kind='text'"""
        query = common.format(f"{node_query} UNION ALL BY NAME {statement_query}")
        connection.execute(f"COPY ({query}) TO '{partial}' (FORMAT PARQUET, COMPRESSION ZSTD)")

        cache.create_index(connection)
        connection.execute(
            f"""COPY (SELECT DISTINCT sha256(r.text) AS text_sha256, r.text
                      FROM read_parquet('{partial}') r
                      WHERE NOT EXISTS (SELECT 1 FROM cache_index c WHERE c.text_sha256 = sha256(r.text))
                      ORDER BY text_sha256)
                TO '{missing}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
        )
        cache.append_stream(_encode_batches(missing, encoder, settings.batch_size))
        # Refresh the temp table: the append wrote a new index part that the table
        # built for the anti-join above does not contain.
        cache.create_index(connection)

        index_path = root / "text-index.parquet"
        connection.execute(
            f"""COPY (SELECT r.owner_index, r.field_id, r.ordinal, r.role, r.language, r.text, c.row
                      FROM read_parquet('{partial}') r
                      JOIN cache_index c ON c.text_sha256 = sha256(r.text)
                      ORDER BY r.owner_index, r.field_id, r.ordinal)
                TO '{index_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
        )
        records = pq.ParquetFile(partial).metadata.num_rows
        resolved = pq.ParquetFile(index_path).metadata.num_rows
        if resolved != records:
            raise ValueError(f"{records - resolved} text records did not resolve to a cached embedding")
        store = DenseFeatureStore(
            provider=settings.provider,
            model=settings.model,
            encoder_key=encoder_key(settings),
            dimension=encoder.dimension,
            rows=resolved,
            index_path=str(index_path),
            parts=cache.part_refs(),
        )
        atomic_write_json(root / "text-features.json", store.model_dump(mode="json"))
        return store
    finally:
        # A mid-run failure must not strand intermediates in the published dataset.
        partial.unlink(missing_ok=True)
        missing.unlink(missing_ok=True)
        connection.close()
