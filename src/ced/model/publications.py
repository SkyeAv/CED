"""Prepare precomputed publication embeddings as aligned statement features."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from duckdb.func import FunctionNullHandling
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from ced.acquisition.base import atomic_write_json, sha256_file
from ced.config.models import PublicationEmbeddingSettings, RuntimeConfig
from ced.kg.audit import validate_checksums
from ced.kg.records import canonical_json
from ced.model.schemas import DatasetManifest, PublicationFeatureStore, TensorRef
from ced.model.tensors import write_tensor_stream
from ced.preflight import require_cpu_worker, require_disk

_IDENTIFIER_PATTERNS = (
    (re.compile(r"^(?:PMID|PubMed)\s*:?\s*(\d+)$", re.IGNORECASE), "PMID"),
    (re.compile(r"^(?:PMCID\s*:\s*PMC|PMC)\s*:?\s*(\d+)$", re.IGNORECASE), "PMC"),
    (re.compile(r"^NCT\s*:?\s*(\d{8})$", re.IGNORECASE), "NCT"),
)


def normalize_publication_id(value: str) -> str | None:
    """Canonicalize supported identifiers without guessing bare or unsupported IDs."""
    text = value.strip()
    if not text:
        return None
    for pattern, prefix in _IDENTIFIER_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            return f"{prefix}{match.group(1)}"
    if "://" not in text:
        return None
    try:
        parsed = urlsplit(text)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        parts = [part for part in parsed.path.split("/") if part]
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    if host == "pubmed.ncbi.nlm.nih.gov" and len(parts) == 1:
        return normalize_publication_id(f"PMID:{parts[0]}")
    if host == "ncbi.nlm.nih.gov" and len(parts) == 2 and parts[0].lower() == "pubmed":
        return normalize_publication_id(f"PMID:{parts[1]}")
    if host in {"pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov"}:
        article = parts[-1] if len(parts) >= 2 and parts[-2].lower() == "articles" else ""
        return normalize_publication_id(article)
    if host == "clinicaltrials.gov" and parts:
        return normalize_publication_id(parts[-1])
    return None


def publication_feature_key(settings: PublicationEmbeddingSettings) -> str:
    payload = settings.model_dump(mode="json", exclude={"batch_size", "fast_transfer", "size"}) | {
        "aggregation": "mean-resolved-v1",
        "evidence": "log1p-distinct-v1",
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:32]


def resolve_publication_embeddings(runtime: RuntimeConfig, *, download: bool, assume_yes: bool = False) -> Path:
    """Resolve the pinned cache artifact, downloading only after explicit approval."""
    require_cpu_worker(runtime)
    settings = runtime.publication_embeddings
    if download:
        if not assume_yes:
            raise RuntimeError(
                f"publication embedding download requires {settings.size} bytes; rerun with --yes after review"
            )
        require_disk(runtime.artifact_path(), settings.size, runtime.disk_reserve_gb)
    if settings.fast_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    try:
        resolved = hf_hub_download(
            repo_id=settings.repo_id,
            repo_type="dataset",
            revision=settings.revision,
            filename=settings.filename,
            cache_dir=runtime.artifact_path("cache", "huggingface"),
            local_files_only=not download,
        )
    except (FileNotFoundError, LocalEntryNotFoundError) as error:
        raise FileNotFoundError(
            "publication embeddings are not cached; run ced encode-publications <dataset-id> --download --yes"
        ) from error
    path = Path(resolved)
    if path.stat().st_size != settings.size:
        raise ValueError(f"publication embedding size mismatch: {path}")
    if sha256_file(path) != settings.source_sha256:
        raise ValueError(f"publication embedding checksum mismatch: {path}")
    return path


def _validate_embedding_schema(source: Path, dimension: int) -> None:
    schema = pq.ParquetFile(source).schema_arrow
    if "id" not in schema.names or "embedding" not in schema.names:
        raise ValueError("publication embeddings require id and embedding columns")
    if not pa.types.is_string(schema.field("id").type):
        raise ValueError("publication embedding ids must be strings")
    embedding = schema.field("embedding").type
    if not pa.types.is_fixed_size_list(embedding) or not pa.types.is_float32(embedding.value_type):
        raise ValueError("publication embeddings must be fixed_size_list<float32>")
    if embedding.list_size != dimension:
        raise ValueError(f"publication embedding dimension {embedding.list_size} does not match configured {dimension}")


def _embedding_matrix(array: pa.Array, dimension: int) -> np.ndarray:
    if array.null_count or not isinstance(array, (pa.FixedSizeListArray, pa.ListArray)):
        raise ValueError("matched publication embeddings must be non-null fixed-size lists")
    if isinstance(array, pa.FixedSizeListArray):
        values_array = array.values.slice(array.offset * dimension, len(array) * dimension)
    else:
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        if len(offsets) and (offsets[0] < 0 or np.any(np.diff(offsets) != dimension)):
            raise ValueError("matched publication embedding rows have inconsistent dimensions")
        start = int(offsets[0]) if len(offsets) else 0
        values_array = array.values.slice(start, len(array) * dimension)
    if values_array.null_count:
        raise ValueError("matched publication embedding values must be non-null")
    values = values_array.to_numpy(zero_copy_only=False)
    if values.size != len(array) * dimension:
        raise ValueError("matched publication embedding rows have inconsistent dimensions")
    vectors = values.reshape(len(array), dimension)
    if not np.isfinite(vectors).all():
        raise ValueError("publication embeddings contain non-finite values")
    return vectors


def _write_pooled_tensors(
    matched: Path,
    evidence: Path,
    statement_count: int,
    dimension: int,
    batch_size: int,
    vectors_destination: Path,
    evidence_destination: Path,
) -> dict[str, int]:
    """Reduce matched rows with bounded RSS and deterministic, sorted accumulation."""
    if statement_count == 0:
        write_tensor_stream(vectors_destination, [], shape=[0, dimension], dtype="float32")
        write_tensor_stream(evidence_destination, [], shape=[0], dtype="float32")
        return {
            "statement_publications": 0,
            "matched_publications": 0,
            "statements_with_publications": 0,
            "statements_with_embeddings": 0,
        }
    scratch = vectors_destination.parent
    sums_path = scratch / "publication-sums.raw.tmp"
    matched_counts_path = scratch / "publication-matched-counts.raw.tmp"
    evidence_counts_path = scratch / "publication-evidence-counts.raw.tmp"
    sums = np.memmap(sums_path, mode="w+", dtype=np.float32, shape=(statement_count, dimension))
    matched_counts = np.memmap(matched_counts_path, mode="w+", dtype=np.int64, shape=(statement_count,))
    evidence_counts = np.memmap(evidence_counts_path, mode="w+", dtype=np.int64, shape=(statement_count,))
    try:
        sums[:] = 0
        matched_counts[:] = 0
        evidence_counts[:] = 0
        for batch in pq.ParquetFile(matched).iter_batches(batch_size=batch_size):
            owners = batch.column("statement_index").to_numpy(zero_copy_only=False)
            vectors = _embedding_matrix(batch.column("embedding"), dimension)
            np.add.at(sums, owners, vectors)
            np.add.at(matched_counts, owners, 1)
        for batch in pq.ParquetFile(evidence).iter_batches(batch_size=batch_size):
            owners = batch.column("statement_index").to_numpy(zero_copy_only=False)
            counts = batch.column("evidence_count").to_numpy(zero_copy_only=False)
            evidence_counts[owners] = counts

        def vector_batches(values: np.ndarray, matched: np.ndarray) -> Iterator[np.ndarray]:
            for start in range(0, statement_count, batch_size):
                stop = min(start + batch_size, statement_count)
                counts = matched[start:stop, None]
                chunk = values[start:stop]
                np.divide(chunk, counts, out=chunk, where=counts > 0)
                yield chunk

        write_tensor_stream(
            vectors_destination,
            vector_batches(sums, matched_counts),
            shape=[statement_count, dimension],
            dtype="float32",
        )
        write_tensor_stream(
            evidence_destination,
            (
                np.log1p(evidence_counts[start : start + batch_size]).astype(np.float32)
                for start in range(0, statement_count, batch_size)
            ),
            shape=[statement_count],
            dtype="float32",
        )
        sums.flush()
        matched_counts.flush()
        evidence_counts.flush()
        return {
            "statement_publications": int(evidence_counts.sum()),
            "matched_publications": int(matched_counts.sum()),
            "statements_with_publications": int(np.count_nonzero(evidence_counts)),
            "statements_with_embeddings": int(np.count_nonzero(matched_counts)),
        }
    finally:
        del sums, matched_counts, evidence_counts
        for path in (sums_path, matched_counts_path, evidence_counts_path):
            path.unlink(missing_ok=True)


def encode_dataset_publications(
    runtime: RuntimeConfig, dataset: str, *, download: bool = False, assume_yes: bool = False
) -> PublicationFeatureStore:
    """Attach mean publication vectors and log evidence strength to every statement."""
    require_cpu_worker(runtime)
    root = runtime.artifact_path("datasets", dataset)
    manifest = DatasetManifest.model_validate(json.loads((root / "manifest.json").read_text()))
    validate_checksums(root, manifest.checksums)
    settings = runtime.publication_embeddings
    source = resolve_publication_embeddings(runtime, download=download, assume_yes=assume_yes)
    _validate_embedding_schema(source, settings.dimension)
    snapshot = runtime.artifact_path("snapshots", manifest.snapshot_id)
    require_disk(
        runtime.artifact_path(),
        manifest.statements * (3 * settings.dimension + 1) * 4,
        runtime.disk_reserve_gb,
    )
    matched = root / "publication-matched.partial.parquet"
    evidence = root / "publication-evidence.partial.parquet"
    vectors_partial = root / "publication-vectors.partial.safetensors"
    evidence_partial = root / "publication-evidence.partial.safetensors"
    vectors_final = root / "publication-vectors.safetensors"
    evidence_final = root / "publication-evidence.safetensors"
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={runtime.cpu_workers}")
        connection.execute("SET memory_limit=?", [runtime.memory_limit])
        connection.execute("SET temp_directory=?", [str(root / "spill")])
        connection.read_parquet(str(snapshot / "statements/*.parquet")).create_view("snapshot_statements")
        connection.read_parquet(str(root / "topology.parquet")).create_view("topology")
        connection.read_parquet(str(source)).create_view("source_embeddings")
        connection.create_function(
            "normalize_publication_id", normalize_publication_id, [str], str, null_handling=FunctionNullHandling.SPECIAL
        )
        missing = connection.execute(
            "SELECT count(*) FROM topology t LEFT JOIN snapshot_statements s ON t.statement_id=s.id WHERE s.id IS NULL"
        ).fetchone()
        assert missing is not None
        if missing[0]:
            raise ValueError(f"{missing[0]} prepared statements are missing from snapshot {manifest.snapshot_id}")
        connection.execute(
            """CREATE TEMP TABLE publication_references AS
               SELECT t.index AS statement_index,
                      normalize_publication_id(trim(u.publication)) AS publication_id,
                      lower(trim(u.publication)) AS raw_id
               FROM snapshot_statements s
               JOIN topology t ON s.id=t.statement_id
               CROSS JOIN UNNEST(s.publications) AS u(publication)
               WHERE u.publication IS NOT NULL AND trim(u.publication) <> ''"""
        )
        connection.execute(
            """CREATE TEMP TABLE publication_evidence AS
               SELECT statement_index,
                      count(DISTINCT coalesce(publication_id, 'unsupported:' || raw_id)) AS evidence_count
               FROM publication_references GROUP BY statement_index"""
        )
        connection.execute(
            """CREATE TEMP TABLE embedded_references AS
               SELECT DISTINCT statement_index, publication_id FROM publication_references
               WHERE publication_id IS NOT NULL"""
        )
        connection.execute(
            f"""COPY (
                    SELECT r.statement_index, r.publication_id, e.embedding
                    FROM embedded_references r JOIN source_embeddings e ON r.publication_id=e.id
                    ORDER BY r.statement_index, r.publication_id
                ) TO '{matched}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
        )
        connection.execute(
            f"""COPY (SELECT statement_index, evidence_count FROM publication_evidence ORDER BY statement_index)
                TO '{evidence}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
        )
        duplicate = connection.execute(
            f"""SELECT count(*) FROM (
                    SELECT statement_index, publication_id FROM read_parquet('{matched}')
                    GROUP BY statement_index, publication_id HAVING count(*) > 1 LIMIT 1
                )"""
        ).fetchone()
        assert duplicate is not None
        if duplicate[0]:
            raise ValueError("publication embedding source ids are not unique")
        counts = _write_pooled_tensors(
            matched,
            evidence,
            manifest.statements,
            settings.dimension,
            settings.batch_size,
            vectors_partial,
            evidence_partial,
        )
        vectors_ref = TensorRef(
            path=vectors_final.name,
            dtype="float32",
            shape=[manifest.statements, settings.dimension],
            sha256=sha256_file(vectors_partial),
        )
        evidence_ref = TensorRef(
            path=evidence_final.name,
            dtype="float32",
            shape=[manifest.statements],
            sha256=sha256_file(evidence_partial),
        )
        store = PublicationFeatureStore(
            repo_id=settings.repo_id,
            revision=settings.revision,
            source_sha256=settings.source_sha256,
            dimension=settings.dimension,
            statements=manifest.statements,
            feature_key=publication_feature_key(settings),
            vectors=vectors_ref,
            evidence=evidence_ref,
            **counts,
        )
        vectors_partial.replace(vectors_final)
        evidence_partial.replace(evidence_final)
        manifest.publications = store
        manifest.checksums.update({vectors_ref.path: vectors_ref.sha256, evidence_ref.path: evidence_ref.sha256})
        atomic_write_json(root / "manifest.json", manifest.model_dump(mode="json"))
        return store
    finally:
        connection.close()
        for path in (matched, evidence, vectors_partial, evidence_partial):
            path.unlink(missing_ok=True)
