from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ced.acquisition.base import sha256_file
from ced.config.models import RuntimeConfig
from ced.kg.audit import validate_checksums
from ced.model.encoding import (
    TextEmbeddingCache,
    TextEncoder,
    encode_batches,
    encoder_key,
    encoding_dimension,
    encoding_model,
    settings_batch_size,
)
from ced.model.feature_store import convert_parquet_tensor, write_indptr
from ced.model.schemas import QualifierStatementCsr, SemanticFeatureStore, TensorRef
from ced.model.tensors import read_tensor, write_tensor_stream
from ced.preflight import require_disk

_ARTIFACTS = (
    "predicate-text.parquet",
    "qualifier-text.parquet",
    "predicate-embeddings.safetensors",
    "qualifier-embeddings.safetensors",
    "statement-qualifiers.parquet",
    "statement-qualifier-statements.safetensors",
    "statement-qualifier-ids.safetensors",
    "statement-qualifier-indptr.safetensors",
)


def _rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _write_cached_vectors(
    connection: duckdb.DuckDBPyConnection,
    cache: TextEmbeddingCache,
    index: Path,
    destination: Path,
    dimension: int,
) -> None:
    """Copy selected rows out of immutable cache parts without loading the whole cache."""
    schema = pa.schema(
        [
            ("role", pa.string()),
            ("owner_index", pa.int64()),
            ("vector", pa.list_(pa.float32(), dimension)),
        ]
    )
    writer = pq.ParquetWriter(destination, schema, compression="zstd")
    offset = 0
    try:
        for part in cache.manifest()["parts"]:
            rows = int(part["rows"])
            selected = connection.execute(
                """SELECT owner_index, role, row FROM read_parquet(?)
                   WHERE row>=? AND row<? ORDER BY row""",
                [str(index), offset, offset + rows],
            ).to_arrow_table()
            if selected.num_rows:
                vectors = read_tensor(
                    cache.parts / str(part["name"]),
                    column="value",
                    dtype="float32",
                    shape=[rows, dimension],
                )
                local = selected.column("row").to_numpy(zero_copy_only=False) - offset
                selected_vectors = np.ascontiguousarray(vectors[local])
                vector = pa.FixedSizeListArray.from_arrays(
                    pa.array(selected_vectors.reshape(-1), type=pa.float32()), dimension
                )
                writer.write_table(
                    pa.Table.from_arrays(
                        [selected.column("role"), selected.column("owner_index"), vector], schema=schema
                    )
                )
            offset += rows
    finally:
        writer.close()
    if offset == 0:
        raise ValueError("semantic encoding produced no cached vectors")
    if _rows(destination) != _rows(index):
        raise ValueError("semantic text records did not resolve to cached embeddings")


def _write_embeddings(
    connection: duckdb.DuckDBPyConnection,
    work: Path,
    role: str,
    output: str,
    owners: int,
    dimension: int,
) -> TensorRef:
    selected = work / "semantic-vectors.parquet"
    ordered = work / f"{role}-vectors.partial.parquet"
    connection.execute(
        f"""COPY (
            SELECT owner_index, vector FROM read_parquet('{selected}')
            WHERE role='{role}' ORDER BY owner_index)
        TO '{ordered}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
    )
    stats = connection.execute(
        f"""SELECT count(*), coalesce(min(owner_index), 0), coalesce(max(owner_index), -1),
            count(DISTINCT owner_index) FROM read_parquet('{ordered}')"""
    ).fetchone()
    assert stats is not None
    if stats != (owners, 0, owners - 1, owners) and not (owners == 0 and stats == (0, 0, -1, 0)):
        raise ValueError(f"{role} embeddings did not resolve to a complete tensor")
    parquet = pq.ParquetFile(ordered)

    def batches():
        for batch in parquet.iter_batches(columns=["vector"], batch_size=256):
            column = batch.column(0)
            if column.null_count:
                raise ValueError(f"{role} embeddings contain null vectors")
            values = column.values.to_numpy(zero_copy_only=False)
            if values.dtype != np.dtype("float32"):
                raise ValueError(f"{role} embeddings have dtype {values.dtype}, expected float32")
            yield np.ascontiguousarray(values)

    destination = work / output
    write_tensor_stream(destination, batches(), shape=[owners, dimension], dtype="float32")
    return TensorRef(
        path=output,
        dtype="float32",
        shape=[owners, dimension],
        sha256=sha256_file(destination),
    )


def encode_semantic_features(
    runtime: RuntimeConfig,
    root: Path,
    source: Path,
    connection: duckdb.DuckDBPyConnection,
    cache: TextEmbeddingCache,
    encoder: TextEncoder | None,
    *,
    additional_records: Path | None = None,
    assume_yes: bool = False,
) -> SemanticFeatureStore:
    """Encode every predicate and qualifier key-value pair into dataset-local tensors."""
    validate_checksums(source, json.loads((source / "checksums.json").read_text()))
    manifest = json.loads((root / "manifest.json").read_text())
    relation_count = int(manifest["relations"])
    statement_count = int(manifest["statements"])
    settings = runtime.text_encoding
    work = root / "semantic-work.partial"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    try:
        connection.read_parquet(str(root / "relations.parquet")).create_view("semantic_relations")
        connection.read_parquet(str(root / "topology.parquet")).create_view("semantic_topology")
        connection.read_parquet(str(source / "statements/*.parquet")).create_view("semantic_statements")
        predicate_files = sorted((source / "predicates").glob("*.parquet"))
        if predicate_files:
            connection.read_parquet(str(source / "predicates/*.parquet")).create_view("predicate_records")
            metadata = "SELECT id, name, definition FROM predicate_records"
        else:
            metadata = """SELECT CAST(NULL AS VARCHAR) AS id, CAST(NULL AS VARCHAR) AS name,
                CAST(NULL AS VARCHAR) AS definition WHERE FALSE"""
        connection.execute(
            f"""CREATE TEMP TABLE predicate_text AS
                WITH metadata AS (
                    SELECT id, string_agg(name, chr(10) ORDER BY name) AS names,
                        string_agg(definition, chr(10) ORDER BY definition) AS definitions
                    FROM (SELECT DISTINCT id, name, definition FROM ({metadata}))
                    GROUP BY id
                )
                SELECT CAST(r.idx AS BIGINT) AS owner_index,
                    'Predicate ID: ' || r.id || chr(10) || 'Name: ' || coalesce(m.names, r.id) ||
                    CASE WHEN m.definitions IS NULL THEN '' ELSE chr(10) || 'Definition: ' || m.definitions END AS text
                FROM semantic_relations r LEFT JOIN metadata m ON r.id=m.id"""
        )
        connection.execute(
            """CREATE TEMP VIEW qualifier_events AS
                SELECT CAST(t.index AS BIGINT) AS statement_id,
                    'Qualifier ID: ' || q.qualifier_key || chr(10) || 'Value: ' ||
                    CAST((CAST(s.qualifiers_json AS JSON) -> q.qualifier_key) AS VARCHAR) AS text
                FROM semantic_statements s
                JOIN semantic_topology t ON s.id=t.statement_id
                CROSS JOIN UNNEST(json_keys(CAST(s.qualifiers_json AS JSON))) AS q(qualifier_key)"""
        )
        connection.execute(
            """CREATE TEMP TABLE qualifier_vocabulary AS
                SELECT CAST(row_number() OVER (ORDER BY text)-1 AS BIGINT) AS owner_index, text
                FROM (SELECT DISTINCT text FROM qualifier_events)"""
        )
        connection.execute(
            """CREATE TEMP TABLE qualifier_links AS
                SELECT DISTINCT e.statement_id, CAST(v.owner_index AS BIGINT) AS qualifier_id
                FROM qualifier_events e JOIN qualifier_vocabulary v ON e.text=v.text
                ORDER BY e.statement_id, v.owner_index"""
        )
        records = work / "semantic-records.parquet"
        connection.execute(
            f"""COPY (
                SELECT owner_index, 'predicate' AS role, text FROM predicate_text
                UNION ALL
                SELECT owner_index, 'qualifier' AS role, text FROM qualifier_vocabulary)
            TO '{records}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
        )
        connection.execute(
            f"COPY (SELECT owner_index AS relation_id, text FROM predicate_text) "
            f"TO '{work / 'predicate-text.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        connection.execute(
            f"COPY (SELECT owner_index AS qualifier_id, text FROM qualifier_vocabulary) "
            f"TO '{work / 'qualifier-text.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        links_path = work / "statement-qualifiers.parquet"
        connection.execute(
            f"COPY (SELECT * FROM qualifier_links ORDER BY statement_id, qualifier_id) "
            f"TO '{links_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        qualifier_count = connection.execute("SELECT count(*) FROM qualifier_vocabulary").fetchone()
        assert qualifier_count is not None
        qualifiers = int(qualifier_count[0])

        cache.create_index(connection)
        missing = work / "semantic-missing.parquet"
        all_records = f"SELECT text FROM read_parquet('{records}')"
        if additional_records is not None:
            all_records += f" UNION ALL SELECT text FROM read_parquet('{additional_records}')"
        connection.execute(
            f"""COPY (SELECT DISTINCT sha256(r.text) AS text_sha256, r.text
                      FROM ({all_records}) r
                      WHERE NOT EXISTS (SELECT 1 FROM cache_index c WHERE c.text_sha256=sha256(r.text))
                      ORDER BY text_sha256)
            TO '{missing}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
        )
        dimension = encoding_dimension(settings)
        if _rows(missing):
            require_disk(runtime.artifact_path(), _rows(missing) * dimension * 8, runtime.disk_reserve_gb)
            if settings.provider == "modal_ensemble":
                from ced.modal_app import encode_text_remote

                encode_text_remote(runtime, missing, cache, assume_yes=assume_yes)
            else:
                if encoder is None:
                    raise ValueError("local text encoding requires an encoder")
                cache.append_stream(encode_batches(missing, encoder, settings_batch_size(settings)))
        cache.create_index(connection)
        index = work / "semantic-index.parquet"
        connection.execute(
            f"""COPY (SELECT r.owner_index, r.role, c.row
                      FROM read_parquet('{records}') r
                      JOIN cache_index c ON c.text_sha256=sha256(r.text)
                      ORDER BY r.role, r.owner_index)
            TO '{index}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
        )
        if _rows(index) != relation_count + qualifiers:
            raise ValueError("semantic text records did not resolve to cached embeddings")
        _write_cached_vectors(connection, cache, index, work / "semantic-vectors.parquet", dimension)
        predicate_ref = _write_embeddings(
            connection, work, "predicate", "predicate-embeddings.safetensors", relation_count, dimension
        )
        qualifier_ref = _write_embeddings(
            connection, work, "qualifier", "qualifier-embeddings.safetensors", qualifiers, dimension
        )
        statement_ref = convert_parquet_tensor(
            links_path, "statement_id", "int64", work / "statement-qualifier-statements.safetensors"
        )
        qualifier_id_ref = convert_parquet_tensor(
            links_path, "qualifier_id", "int64", work / "statement-qualifier-ids.safetensors"
        )
        indptr_ref = write_indptr(
            connection,
            links_path,
            statement_count,
            work / "statement-qualifier-indptr.safetensors",
            owner_column="statement_id",
        )
        links = QualifierStatementCsr(
            statement_id=statement_ref,
            qualifier_id=qualifier_id_ref,
            indptr=indptr_ref,
        )
        store = SemanticFeatureStore(
            provider=settings.provider,
            model=encoding_model(settings),
            revision=settings.revision,
            encoder_key=encoder_key(settings),
            dimension=dimension,
            relations=relation_count,
            qualifier_count=qualifiers,
            predicate_text_path="predicate-text.parquet",
            qualifier_text_path="qualifier-text.parquet",
            predicates=predicate_ref,
            qualifiers=qualifier_ref,
            statement_qualifiers=links,
        )
        for name in _ARTIFACTS:
            (work / name).replace(root / name)
        return store
    finally:
        shutil.rmtree(work, ignore_errors=True)
