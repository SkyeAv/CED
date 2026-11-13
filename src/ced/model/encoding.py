from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
from collections.abc import Iterator, Sequence
from itertools import islice
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
from ced.model.feature_store import convert_parquet_tensor, write_indptr
from ced.model.schemas import (
    DatasetManifest,
    DenseFeatureStore,
    DensePartRef,
    FileArtifact,
    NodeSemanticArtifacts,
    TensorRef,
)
from ced.model.tensors import write_tensor
from ced.preflight import require_cpu_worker

PART_ROWS = 65_536
INDEX_SCHEMA = pa.schema([("text_sha256", pa.string()), ("row", pa.int64())])


class TextEncoder(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def encoding_model(settings: TextEncodingSettings) -> str:
    return settings.ensemble.model_name if settings.provider == "modal_ensemble" else settings.model


def encoding_dimension(settings: TextEncodingSettings) -> int:
    return settings.ensemble.dimension if settings.provider == "modal_ensemble" else settings.dimension


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
        self.model = SentenceTransformer(
            settings.model, revision=settings.revision, device=device, local_files_only=settings.local_files_only
        )
        self.model.max_seq_length = settings.max_length
        embedding_dimension = self.model.get_sentence_embedding_dimension()
        if embedding_dimension is None:
            raise ValueError(f"{settings.model} does not expose an embedding dimension")
        self.dimension = embedding_dimension
        if self.dimension != settings.dimension:
            raise ValueError(
                f"configured text dimension {settings.dimension} does not match model dimension {self.dimension}"
            )

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        return self.model.encode(
            list(texts),
            batch_size=settings_batch_size(self.settings),
            normalize_embeddings=self.settings.normalize,
            convert_to_tensor=True,
            show_progress_bar=False,
        )


def settings_batch_size(settings: TextEncodingSettings) -> int:
    return settings.ensemble.batch_size if settings.provider == "modal_ensemble" else settings.batch_size


def create_text_encoder(settings: TextEncodingSettings) -> TextEncoder:
    if settings.provider == "random":
        return RandomTextEncoder(settings.dimension, settings.normalize)
    if settings.provider == "sentence_transformers":
        return SentenceTransformerTextEncoder(settings)
    if settings.provider == "modal_ensemble":
        raise ValueError("modal_ensemble runs only through Modal; use ced encode-text")
    raise ValueError("text encoding is disabled; choose random, sentence_transformers, or modal_ensemble")


def encoder_key(settings: TextEncodingSettings) -> str:
    if settings.provider == "modal_ensemble":
        payload: dict[str, Any] = {
            "provider": settings.provider,
            "ensemble": settings.ensemble.model_dump(mode="json", exclude={"gpu"}),
        }
    else:
        payload = settings.model_dump(mode="json", exclude={"cache_subdirectory", "ensemble"})
    payload["tensor_format"] = "safetensors"
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


class TextEmbeddingCache:
    """Persistent content-addressed dense-text store with bounded streaming updates."""

    def __init__(self, root: Path, settings: TextEncodingSettings) -> None:
        self.root = root / encoder_key(settings)
        self.settings = settings
        self.manifest_path = self.root / "manifest.json"
        self.parts_root = self.parts = self.root / "parts"
        self.index_root = self.index = self.root / "index"
        self.parts.mkdir(parents=True, exist_ok=True)
        self.index.mkdir(parents=True, exist_ok=True)

    def manifest(self) -> dict[str, Any]:
        if self.manifest_path.is_file():
            return json.loads(self.manifest_path.read_text())
        return {
            "provider": self.settings.provider,
            "model": encoding_model(self.settings),
            "dimension": encoding_dimension(self.settings),
            "rows": 0,
            "normalize": self.settings.normalize,
            "parts": [],
        }

    def create_index(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Expose only manifest-published rows; ignore orphan parts from interrupted writes."""
        names = [Path(str(part["name"])).with_suffix(".parquet").name for part in self.manifest()["parts"]]
        if not names:
            connection.execute("CREATE OR REPLACE TEMP TABLE cache_index (text_sha256 VARCHAR, row BIGINT)")
            return
        listing = ", ".join(f"'{self.index / name}'" for name in names)
        connection.execute(
            f"""CREATE OR REPLACE TEMP TABLE cache_index AS
                SELECT text_sha256,min(row) AS row FROM read_parquet([{listing}]) GROUP BY text_sha256"""
        )

    def _validate_new_keys(self, connection: duckdb.DuckDBPyConnection, index: pa.Table, name: str) -> None:
        """Register an incoming relation after proving its keys are new and unique."""
        connection.register("incoming_index", index)
        try:
            counts = connection.execute("SELECT count(*), count(DISTINCT text_sha256) FROM incoming_index").fetchone()
            assert counts is not None
            if counts[0] != counts[1]:
                raise ValueError(f"text keys are duplicated in {name}")
            cached = connection.execute(
                "SELECT count(*) FROM incoming_index i JOIN cache_index c USING (text_sha256)"
            ).fetchone()
            assert cached is not None
            if cached[0]:
                raise ValueError(f"text keys are already cached in {name}")
        except Exception:
            connection.unregister("incoming_index")
            raise

    def part_refs(self) -> list[TensorRef]:
        manifest = self.manifest()
        dimension = int(manifest["dimension"])
        references = []
        for part in manifest["parts"]:
            path = self.parts / str(part["name"])
            references.append(
                TensorRef(
                    path=str(path),
                    dtype="float32",
                    shape=[int(part["rows"]), dimension],
                    sha256=sha256_file(path),
                )
            )
        return references

    def append_stream(self, batches: Iterator[tuple[Sequence[str], Any]]) -> int:
        manifest = self.manifest()
        dimension = int(manifest["dimension"])
        row = int(manifest["rows"])
        parts = list(manifest["parts"])
        pending_keys: list[str] = []
        pending: list[np.ndarray] = []
        pending_rows = 0
        added = 0
        connection = duckdb.connect()
        self.create_index(connection)

        def flush() -> None:
            nonlocal row, pending_keys, pending, pending_rows
            if not pending_rows:
                return
            name = f"part-{len(parts):06d}.safetensors"
            index_name = Path(name).with_suffix(".parquet").name
            vectors = np.concatenate(pending) if len(pending) > 1 else pending[0]
            if not np.isfinite(vectors).all():
                raise ValueError("text encoder returned non-finite values")
            index = pa.table(
                {
                    "text_sha256": pa.array(pending_keys, pa.string()),
                    "row": pa.array(np.arange(row, row + pending_rows, dtype=np.int64)),
                }
            )
            self._validate_new_keys(connection, index, index_name)
            write_tensor(self.parts / name, vectors)
            temporary = self.index / f"{index_name}.tmp-{os.getpid()}"
            pq.write_table(index, temporary, compression="zstd")
            temporary.replace(self.index / index_name)
            parts.append({"name": name, "rows": pending_rows})
            row += pending_rows
            atomic_write_json(self.manifest_path, manifest | {"rows": row, "parts": parts})
            connection.execute("INSERT INTO cache_index SELECT text_sha256, row FROM incoming_index")
            connection.unregister("incoming_index")
            pending_keys, pending, pending_rows = [], [], 0

        try:
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
            return added
        finally:
            connection.close()

    def adopt_parts(
        self,
        encoded: Path,
        missing: Path,
        *,
        input_row_start: int = 0,
        rows: int | None = None,
    ) -> int:
        """Validate and publish each part of a selected missing-input slice."""
        previous = self.manifest()
        dimension = int(previous["dimension"])
        first_row = int(previous["rows"])
        parts = list(previous["parts"])
        vectors = sorted((encoded / "vectors").glob("part-*.safetensors"))
        expected_file = pq.ParquetFile(missing)
        available_rows = expected_file.metadata.num_rows
        expected_total = available_rows - input_row_start if rows is None else rows
        if input_row_start < 0 or expected_total < 0 or input_row_start + expected_total > available_rows:
            raise ValueError(
                f"invalid Modal input slice {input_row_start}:{input_row_start + expected_total} "
                f"for {available_rows} rows"
            )
        key_rows = (
            key
            for batch in expected_file.iter_batches(PART_ROWS, columns=["text_sha256"])
            for key in batch.column("text_sha256").to_pylist()
        )
        expected_keys = islice(key_rows, input_row_start, input_row_start + expected_total)
        consumed = 0
        self.parts_root.mkdir(exist_ok=True)
        self.index_root.mkdir(exist_ok=True)
        connection = duckdb.connect()
        try:
            self.create_index(connection)
            for vector_path in vectors:
                name = f"part-{len(parts):06d}.safetensors"
                if vector_path.name != name:
                    raise ValueError(f"unexpected Modal part {vector_path.name}; expected {name}")
                index_path = encoded / "index" / Path(name).with_suffix(".parquet").name
                index = pq.read_table(index_path, schema=INDEX_SCHEMA)
                part_rows = index.num_rows
                expected = pa.chunked_array([pa.array(list(islice(expected_keys, part_rows)), type=pa.string())])
                if part_rows == 0 or not index["text_sha256"].equals(expected):
                    raise ValueError(f"Modal text order mismatch in {name}")
                expected_rows = np.arange(first_row + consumed, first_row + consumed + part_rows, dtype=np.int64)
                if not np.array_equal(index["row"].to_numpy(zero_copy_only=False), expected_rows):
                    raise ValueError(f"Modal row mismatch in {name}")
                self._validate_new_keys(connection, index, name)
                from ced.model.tensors import read_tensor

                values = read_tensor(vector_path, column="value", dtype="float32", shape=[part_rows, dimension])
                if not np.isfinite(values).all():
                    raise ValueError(f"Modal returned non-finite values in {name}")
                destination = self.parts_root / name
                destination.unlink(missing_ok=True)
                vector_path.replace(destination)
                index_destination = self.index_root / index_path.name
                index_destination.unlink(missing_ok=True)
                index_path.replace(index_destination)
                parts.append({"name": name, "rows": part_rows})
                consumed += part_rows
                atomic_write_json(self.manifest_path, previous | {"rows": first_row + consumed, "parts": parts})
                connection.execute("INSERT INTO cache_index SELECT text_sha256, row FROM incoming_index")
                connection.unregister("incoming_index")
            if consumed != expected_total:
                raise ValueError(f"Modal returned {consumed} rows, expected {expected_total}")
        finally:
            connection.close()
        return consumed


def encode_batches(missing: Path, encoder: TextEncoder, batch_size: int) -> Iterator[tuple[list[str], np.ndarray]]:
    for batch in pq.ParquetFile(missing).iter_batches(batch_size, columns=["text"]):
        texts = batch.column("text").to_pylist()
        vectors = encoder.encode(texts).detach().cpu().numpy().astype(np.float32, copy=False)
        yield texts, np.ascontiguousarray(vectors)


def encode_missing_to_parts(
    missing: Path, output: Path, encoder: TextEncoder, *, start_row: int = 0, start_part: int = 0
) -> dict[str, Any]:
    """Encode the missing-text Parquet into cache-compatible vector and index parts."""
    (output / "vectors").mkdir(parents=True, exist_ok=True)
    (output / "index").mkdir(parents=True, exist_ok=True)
    rows = 0
    parts = 0
    files: list[str] = []
    for batch in pq.ParquetFile(missing).iter_batches(PART_ROWS, columns=["text_sha256", "text"]):
        texts = batch.column("text").to_pylist()
        keys = batch.column("text_sha256")
        vectors = np.ascontiguousarray(encoder.encode(texts).detach().cpu().numpy(), dtype=np.float32)
        if vectors.shape != (len(texts), encoder.dimension) or not np.isfinite(vectors).all():
            raise ValueError(f"invalid ensemble output shape or values: {vectors.shape}")
        name = f"part-{start_part + parts:06d}.safetensors"
        index_name = Path(name).with_suffix(".parquet").name
        write_tensor(output / "vectors" / name, vectors)
        pq.write_table(
            pa.table(
                {"text_sha256": keys, "row": range(start_row + rows, start_row + rows + len(texts))},
                schema=INDEX_SCHEMA,
            ),
            output / "index" / index_name,
            compression="zstd",
        )
        files.extend((f"vectors/{name}", f"index/{index_name}"))
        rows += len(texts)
        parts += 1
    return {"groups": 0, "rows": rows, "parts": parts, "dimension": encoder.dimension, "files": files}


def _with_prefix(reference: TensorRef, prefix: str) -> TensorRef:
    return reference.model_copy(update={"path": f"{prefix}/{reference.path}"})


def _write_node_semantics(
    connection: duckdb.DuckDBPyConnection, root: Path, artifact_prefix: str
) -> NodeSemanticArtifacts:
    connection.execute("""CREATE TEMP TABLE prefix_values AS
        SELECT value, row_number() OVER (ORDER BY value) AS value_id FROM (
          SELECT DISTINCT split_part(id, ':', 1) AS value FROM entities)""")
    connection.execute("""CREATE TEMP TABLE category_values AS
        SELECT value, row_number() OVER (ORDER BY value) AS value_id FROM (
          SELECT DISTINCT category AS value FROM entities CROSS JOIN UNNEST(categories) AS u(category))""")
    for table, name in (
        ("prefix_values", "prefix-vocabulary.parquet"),
        ("category_values", "category-vocabulary.parquet"),
    ):
        connection.execute(f"COPY {table} TO '{root / name}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    prefix_path = root / "node-prefixes.parquet"
    connection.execute(f"""COPY (SELECT CAST(coalesce(p.value_id, 0) AS BIGINT) AS value
        FROM entities e LEFT JOIN prefix_values p ON split_part(e.id, ':', 1)=p.value ORDER BY e.idx)
        TO '{prefix_path}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    node_count = connection.execute("SELECT count(*) FROM entities").fetchone()
    assert node_count is not None
    prefix = _with_prefix(
        convert_parquet_tensor(prefix_path, "value", "int64", root / "node-prefixes.safetensors"), artifact_prefix
    )
    category_path = root / "node-categories.parquet"
    connection.execute(f"""COPY (SELECT CAST(e.idx AS BIGINT) AS owner_index, CAST(c.value_id AS BIGINT) AS value_id
        FROM entities e CROSS JOIN UNNEST(e.categories) AS u(category)
        JOIN category_values c ON u.category=c.value ORDER BY e.idx,c.value_id)
        TO '{category_path}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    category_rows = connection.execute(f"SELECT count(*) FROM read_parquet('{category_path}')").fetchone()
    assert category_rows is not None
    category = _with_prefix(
        convert_parquet_tensor(category_path, "value_id", "int64", root / "node-categories.safetensors"),
        artifact_prefix,
    )
    indptr = _with_prefix(
        write_indptr(connection, category_path, node_count[0], root / "node-category-indptr.safetensors"),
        artifact_prefix,
    )
    prefix_count = connection.execute("SELECT count(*) FROM prefix_values").fetchone()
    category_count = connection.execute("SELECT count(*) FROM category_values").fetchone()
    assert prefix_count is not None
    assert category_count is not None
    return NodeSemanticArtifacts(
        prefix_id=prefix,
        category_id=category,
        category_indptr=indptr,
        prefix_vocabulary_size=prefix_count[0] + 1,
        category_vocabulary_size=category_count[0] + 1,
    )


def _cache_index_query(cache: TextEmbeddingCache) -> str:
    files = [
        str(cache.index / Path(str(part["name"])).with_suffix(".parquet").name) for part in cache.manifest()["parts"]
    ]
    if files:
        return f"SELECT text_sha256,row FROM read_parquet({json.dumps(files)})"
    return "SELECT CAST(NULL AS VARCHAR) AS text_sha256,CAST(NULL AS BIGINT) AS row WHERE false"


def encode_dataset_text(runtime: RuntimeConfig, dataset: str, *, assume_yes: bool = False) -> DenseFeatureStore:
    """Encode node names, qualifiers, and long attributes, then publish dense model inputs."""
    require_cpu_worker(runtime)
    settings = runtime.text_encoding
    if settings.provider == "none":
        raise ValueError("text encoding is disabled; choose random, sentence_transformers, or modal_ensemble")
    root = runtime.artifact_path("datasets", dataset)
    manifest = json.loads((root / "manifest.json").read_text())
    validate_checksums(root, manifest["checksums"])
    snapshot = runtime.artifact_path("snapshots", manifest["snapshot_id"])
    validate_checksums(snapshot, json.loads((snapshot / "checksums.json").read_text()))
    cache = TextEmbeddingCache(runtime.artifact_path(str(settings.cache_subdirectory)), settings)
    key = encoder_key(settings)
    artifact_name = f"text-{key}"
    artifact = root / artifact_name
    partial_artifact = root / f"{artifact_name}.partial"
    partial = root / "text-records.partial.parquet"
    missing = root / "text-missing.partial.parquet"
    connection = duckdb.connect()
    try:
        if partial_artifact.exists():
            shutil.rmtree(partial_artifact)
        (partial_artifact / "parts").mkdir(parents=True)
        connection.execute(f"SET threads={runtime.cpu_workers}")
        connection.execute("SET memory_limit=?", [runtime.memory_limit])
        connection.execute("SET temp_directory=?", [str(root / "spill")])
        connection.execute("SET preserve_insertion_order=false")
        connection.read_parquet(str(root / "entities.parquet")).create_view("entities")
        connection.read_parquet(str(root / "topology.parquet")).create_view("splits")
        node_count = manifest["entities"]
        node_query = f"""SELECT e.idx AS owner_index, f.field_id, f.ordinal, 'node' AS role, NULL AS language, f.text
            FROM read_parquet('{root}/node-features.parquet') f JOIN entities e ON f.owner_id=e.id WHERE f.kind='text'"""
        statement_query = f"""SELECT {node_count}+s.index AS owner_index, f.field_id, f.ordinal, 'statement' AS role,
            NULL AS language, f.text FROM read_parquet('{root}/statement-features.parquet') f
            JOIN splits s ON f.owner_id=s.statement_id WHERE f.kind='text'"""
        names_query = f"""SELECT e.idx AS owner_index, 0 AS field_id, CAST(u.ordinal-1 AS INTEGER) AS ordinal,
            'node_name' AS role, NULL AS language, trim(u.name) AS text
            FROM read_parquet('{snapshot}/nodes/*.parquet') n JOIN entities e ON n.id=e.id
            CROSS JOIN UNNEST(n.names) WITH ORDINALITY AS u(name,ordinal)
            WHERE u.name IS NOT NULL AND length(trim(u.name))>0"""
        classes_query = """SELECT e.idx AS owner_index, 0 AS field_id, CAST(u.ordinal-1 AS INTEGER) AS ordinal,
            'node_class' AS role, NULL AS language, trim(u.category) AS text
            FROM entities e CROSS JOIN UNNEST(e.categories) WITH ORDINALITY AS u(category,ordinal)
            WHERE u.category IS NOT NULL AND length(trim(u.category))>0"""
        qualifiers_query = f"""SELECT {node_count}+s.index AS owner_index, 0 AS field_id, 0 AS ordinal,
            'qualifier' AS role, NULL AS language,
            q.key || ': ' || coalesce(json_extract_string(q.value, '$'), 'null') AS text
            FROM read_parquet('{snapshot}/statements/*.parquet') statements JOIN splits s ON statements.id=s.statement_id
            CROSS JOIN json_each(statements.qualifiers_json) AS q WHERE q.key IS NOT NULL"""
        query = f"""SELECT owner_index,field_id,ordinal,role,language,text FROM (
            {node_query} UNION ALL BY NAME {statement_query} UNION ALL BY NAME {names_query}
            UNION ALL BY NAME {classes_query} UNION ALL BY NAME {qualifiers_query}
            ) ORDER BY owner_index,field_id,ordinal,role,text"""
        connection.execute(f"COPY ({query}) TO '{partial}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        from ced.model.semantic_encoding import encode_semantic_features

        encoder = None if settings.provider == "modal_ensemble" else create_text_encoder(settings)
        semantic = encode_semantic_features(
            runtime, root, snapshot, connection, cache, encoder, additional_records=partial, assume_yes=assume_yes
        )
        cache_index = _cache_index_query(cache)
        connection.execute(f"""COPY (SELECT p.owner_index,p.field_id,p.ordinal,p.role,p.language,p.text,c.row
            FROM read_parquet('{partial}') p JOIN ({cache_index}) c ON sha256(p.text)=c.text_sha256
            ORDER BY p.owner_index,p.field_id,p.ordinal,p.role,p.text)
            TO '{partial_artifact / "text-index.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        index_path = partial_artifact / "text-index.parquet"
        records = pq.ParquetFile(index_path).metadata.num_rows
        expected_records = pq.ParquetFile(partial).metadata.num_rows
        if records != expected_records:
            raise ValueError(f"{expected_records - records} text records did not resolve to a cached embedding")
        node = _write_node_semantics(connection, partial_artifact, artifact_name)
        cache_parts = cache.part_refs()
        starts: list[int] = []
        next_row = 0
        for reference in cache_parts:
            starts.append(next_row)
            next_row += reference.shape[0]
        row_values = pq.read_table(index_path, columns=["row"])["row"].to_numpy(zero_copy_only=False)
        used: np.ndarray = (
            np.unique(np.searchsorted(np.asarray(starts), row_values, side="right") - 1)
            if records
            else np.empty(0, dtype=np.int64)
        )
        if records and (not cache_parts or used.min() < 0 or used.max() >= len(cache_parts)):
            raise ValueError("text index references a missing cache part")
        parts: list[DensePartRef] = []
        for part_index_value in used.tolist():
            part_index = int(part_index_value)
            reference = cache_parts[part_index]
            source = Path(reference.path)
            destination = partial_artifact / "parts" / source.name
            try:
                os.link(source, destination)
            except OSError as exception:
                if exception.errno != errno.EXDEV:
                    raise
                shutil.copy2(source, destination)
            parts.append(
                DensePartRef.model_validate(
                    reference.model_dump()
                    | {"path": f"{artifact_name}/parts/{source.name}", "row_start": starts[part_index]}
                )
            )
        checksums = {
            f"{artifact_name}/{path.relative_to(partial_artifact)}": sha256_file(path)
            for path in sorted(partial_artifact.rglob("*"))
            if path.is_file()
        }
        store = DenseFeatureStore(
            provider=settings.provider,
            model=encoding_model(settings),
            encoder_key=key,
            dimension=encoding_dimension(settings),
            rows=records,
            index=FileArtifact(
                path=f"{artifact_name}/text-index.parquet", rows=records, sha256=sha256_file(index_path)
            ),
            node=node,
            parts=parts,
            checksums=checksums,
        )
        if artifact.exists():
            # Published cache-derived artifacts are immutable. Keep the old publication
            # intact on failures, including a mismatched rerun under the same key.
            for name, digest in checksums.items():
                published = root / name
                if not published.is_file() or sha256_file(published) != digest:
                    raise ValueError("published dense artifacts differ; use a new encoder configuration")
            shutil.rmtree(partial_artifact)
        else:
            partial_artifact.rename(artifact)
        atomic_write_json(root / "text-features.json", store.model_dump(mode="json"))
        manifest["semantic"] = semantic.model_dump(mode="json")
        from ced.model.semantic_encoding import _ARTIFACTS

        manifest["checksums"].update({name: sha256_file(root / name) for name in _ARTIFACTS})
        manifest["features"]["dense"] = store.model_dump(mode="json")
        manifest["checksums"].update(checksums)
        manifest["checksums"]["text-features.json"] = sha256_file(root / "text-features.json")
        DatasetManifest.model_validate(manifest)
        atomic_write_json(root / "manifest.json", manifest)
        return store
    finally:
        partial.unlink(missing_ok=True)
        missing.unlink(missing_ok=True)
        if partial_artifact.exists():
            shutil.rmtree(partial_artifact)
        connection.close()
