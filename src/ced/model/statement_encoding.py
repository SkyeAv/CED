from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

from ced.acquisition.base import atomic_write_json, sha256_file
from ced.config.models import RuntimeConfig
from ced.kg.audit import validate_checksums
from ced.model.encoding import TextEmbeddingCache, create_text_encoder, encode_batches, encoder_key, text_key
from ced.model.feature_store import convert_parquet_tensor, write_indptr
from ced.model.schemas import DatasetManifest, StatementAttributeStore, TensorRef
from ced.model.tensors import read_tensor, write_tensor
from ced.preflight import require_cpu_worker

_ARTIFACTS = (
    "statement-attribute-category-vectors.safetensors",
    "statement-attribute-category-id.safetensors",
    "statement-attribute-category-indptr.safetensors",
    "statement-attribute-numeric-value.safetensors",
    "statement-attribute-numeric-field.safetensors",
    "statement-attribute-numeric-valid.safetensors",
    "statement-attribute-numeric-indptr.safetensors",
)


def _cache_rows(cache: TextEmbeddingCache) -> dict[str, int]:
    """Map every published text digest to its global cache row.

    Only parts listed in ``manifest.json`` are read, mirroring the cache index, so
    orphaned parts from a crashed append never contribute vectors.
    """
    rows: dict[str, int] = {}
    for part in cache.manifest()["parts"]:
        path = cache.index / Path(str(part["name"])).with_suffix(".parquet")
        table = pq.read_table(path, columns=["text_sha256", "row"])
        for digest, row in zip(table.column("text_sha256").to_pylist(), table.column("row").to_pylist(), strict=True):
            rows[digest] = int(row)
    return rows


def _vocabulary_vectors(cache: TextEmbeddingCache, pairs: list[tuple[int, str]], dimension: int) -> np.ndarray:
    """Gather vocabulary text embeddings in categorical-id order from the cache parts.

    Each referenced safetensors part is loaded at most once; parts hold at most
    ``PART_ROWS`` rows, so peak memory stays bounded by one part per encoder run.
    """
    rows = _cache_rows(cache)
    parts = cache.manifest()["parts"]
    starts = np.cumsum([0] + [int(part["rows"]) for part in parts])[:-1]
    loaded: dict[int, np.ndarray] = {}
    vectors = np.empty((len(pairs), dimension), dtype=np.float32)
    for categorical_id, text in pairs:
        global_row = rows[text_key(text)]
        part_index = int(np.searchsorted(starts, global_row, side="right") - 1)
        if part_index not in loaded:
            path = cache.parts / str(parts[part_index]["name"])
            loaded[part_index] = read_tensor(
                path, column="value", dtype="float32", shape=[int(parts[part_index]["rows"]), dimension]
            )
        vectors[categorical_id] = loaded[part_index][global_row - int(starts[part_index])]
    return vectors


def encode_statement_attributes(runtime: RuntimeConfig, dataset: str) -> StatementAttributeStore:
    """Encode statement categorical and numeric attributes on wenceslaus.

    Categorical values become a statement-level CSR over embedded ``path: value``
    texts from the configured text encoder; numeric attributes become a CSR of
    z-scored sparse entries with dense local field columns over the sorted fitted
    statement numeric fields. Re-running reuses the shared text embedding cache
    and republishes identical artifacts for identical inputs.
    """
    require_cpu_worker(runtime)
    root = runtime.artifact_path("datasets", dataset)
    manifest = json.loads((root / "manifest.json").read_text())
    validate_checksums(root, manifest["checksums"])
    statement_count = int(manifest["statements"])
    settings = runtime.text_encoding
    encoder = create_text_encoder(settings)
    cache = TextEmbeddingCache(runtime.artifact_path(str(settings.cache_subdirectory)), settings)
    work = root / "statement-attributes.partial"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={runtime.cpu_workers}")
        connection.execute("SET memory_limit=?", [runtime.memory_limit])
        connection.execute("SET temp_directory=?", [str(work / "spill")])
        connection.read_parquet(str(root / "statement-features.parquet")).create_view("attribute_features")
        connection.read_parquet(str(root / "topology.parquet")).create_view("attribute_topology")
        connection.execute(
            """CREATE TEMP VIEW statement_categories AS
                SELECT CAST(t.index AS BIGINT) AS owner_index,
                    f.path || ': ' ||
                      coalesce(json_extract_string(CAST(f.raw_json AS JSON), '$'), f.raw_json) AS text
                FROM attribute_features f JOIN attribute_topology t ON f.owner_id=t.statement_id
                WHERE f.kind='categorical'"""
        )
        connection.execute(
            """CREATE TEMP TABLE category_vocabulary AS
                SELECT CAST(row_number() OVER (ORDER BY text)-1 AS BIGINT) AS categorical_id, text
                FROM (SELECT DISTINCT text FROM statement_categories)"""
        )
        category_rows = work / "category-rows.parquet"
        connection.execute(
            f"""COPY (
                SELECT DISTINCT s.owner_index AS owner_index, v.categorical_id AS categorical_id
                FROM statement_categories s JOIN category_vocabulary v ON s.text=v.text
                ORDER BY owner_index, categorical_id)
            TO '{category_rows}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
        )
        # Held-out statements keep unfitted numeric fields at id 0, which carries no
        # fitted normalization, so those entries are excluded instead of guessed.
        numeric_rows = work / "numeric-rows.parquet"
        connection.execute(
            f"""COPY (
                SELECT CAST(t.index AS BIGINT) AS owner_index,
                    CAST(DENSE_RANK() OVER (ORDER BY f.field_id)-1 AS BIGINT) AS field_id,
                    CAST(f.numeric AS REAL) AS value,
                    CAST(f.valid AS BOOLEAN) AS valid
                FROM attribute_features f JOIN attribute_topology t ON f.owner_id=t.statement_id
                WHERE f.kind='numeric' AND f.field_id > 0
                ORDER BY owner_index, field_id)
            TO '{numeric_rows}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
        )
        counts = connection.execute(
            f"""SELECT
                  (SELECT count(*) FROM category_vocabulary),
                  (SELECT count(*) FROM read_parquet('{category_rows}')),
                  (SELECT count(*) FROM read_parquet('{numeric_rows}')),
                  (SELECT count(DISTINCT field_id) FROM read_parquet('{numeric_rows}'))"""
        ).fetchone()
        assert counts is not None
        categories, _, numeric_entries, numeric_fields = (int(value) for value in counts)

        if categories:
            records = work / "category-records.parquet"
            connection.execute(
                f"""COPY (SELECT categorical_id, text FROM category_vocabulary ORDER BY categorical_id)
                TO '{records}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
            )
            cache.create_index(connection)
            missing = work / "category-missing.parquet"
            connection.execute(
                f"""COPY (SELECT DISTINCT sha256(r.text) AS text_sha256, r.text
                          FROM read_parquet('{records}') r
                          WHERE NOT EXISTS (SELECT 1 FROM cache_index c WHERE c.text_sha256=sha256(r.text))
                          ORDER BY text_sha256)
                TO '{missing}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
            )
            cache.append_stream(encode_batches(missing, encoder, settings.batch_size))
            table = pq.read_table(records)
            pairs = [
                (int(categorical_id), text)
                for categorical_id, text in zip(
                    table.column("categorical_id").to_pylist(), table.column("text").to_pylist(), strict=True
                )
            ]
            embedded = work / "statement-attribute-category-vectors.safetensors"
            write_tensor(embedded, _vocabulary_vectors(cache, pairs, encoder.dimension))
            vectors = TensorRef(
                path=embedded.name,
                dtype="float32",
                shape=[categories, encoder.dimension],
                sha256=sha256_file(embedded),
            )
        else:
            embedded = work / "statement-attribute-category-vectors.safetensors"
            write_tensor(embedded, np.empty((0, encoder.dimension), dtype=np.float32))
            vectors = TensorRef(
                path=embedded.name, dtype="float32", shape=[0, encoder.dimension], sha256=sha256_file(embedded)
            )

        category_id = convert_parquet_tensor(
            category_rows, "categorical_id", "int64", work / "statement-attribute-category-id.safetensors"
        )
        numeric_value = convert_parquet_tensor(
            numeric_rows, "value", "float32", work / "statement-attribute-numeric-value.safetensors"
        )
        numeric_field = convert_parquet_tensor(
            numeric_rows, "field_id", "int64", work / "statement-attribute-numeric-field.safetensors"
        )
        numeric_valid = convert_parquet_tensor(
            numeric_rows, "valid", "bool", work / "statement-attribute-numeric-valid.safetensors"
        )
        if numeric_field.shape != [numeric_entries] or numeric_value.shape != [numeric_entries]:
            raise ValueError("statement numeric tensors are misaligned with their rows")
        store = StatementAttributeStore(
            provider=settings.provider,
            model=settings.model,
            encoder_key=encoder_key(settings),
            dimension=encoder.dimension,
            statement_count=statement_count,
            categorical_count=categories,
            numeric_fields=numeric_fields,
            categorical_vectors=vectors,
            categorical_id=category_id,
            categorical_indptr=write_indptr(
                connection, category_rows, statement_count, work / "statement-attribute-category-indptr.safetensors"
            ),
            numeric_value=numeric_value,
            numeric_field_id=numeric_field,
            numeric_valid=numeric_valid,
            numeric_indptr=write_indptr(
                connection, numeric_rows, statement_count, work / "statement-attribute-numeric-indptr.safetensors"
            ),
        )
        for name in _ARTIFACTS:
            (work / name).replace(root / name)
        manifest["statement_attributes"] = store.model_dump(mode="json")
        manifest["checksums"] = {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.name != "manifest.json"
            and "spill" not in path.parts
            and not any(part.endswith(".partial") for part in path.parts)
        }
        document = DatasetManifest.model_validate(manifest)
        atomic_write_json(root / "manifest.json", document.model_dump(mode="json"))
        return store
    finally:
        shutil.rmtree(work, ignore_errors=True)
        connection.close()
