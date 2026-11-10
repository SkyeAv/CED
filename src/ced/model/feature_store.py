from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import torch

from ced.model.schemas import CategoricalCsr, FeatureFieldSpec, FeatureStoreArtifact, NumericCoo, TensorRef


def _save(root: Path, name: str, values: np.ndarray) -> TensorRef:
    from ced.acquisition.base import sha256_file

    if isinstance(values, np.memmap):
        values.flush()
    else:
        np.save(root / f"{name}.npy", values, allow_pickle=False)
    return TensorRef(
        path=f"{name}.npy", dtype=str(values.dtype), shape=list(values.shape), sha256=sha256_file(root / f"{name}.npy")
    )


def _write_ordered(
    connection: duckdb.DuckDBPyConnection, query: str, root: Path, prefix: str, owners: int
) -> tuple[list[TensorRef], int]:
    count = connection.execute(f"SELECT count(*) FROM ({query})").fetchone()
    assert count is not None
    columns = {
        "owner": np.lib.format.open_memmap(root / f"{prefix}-owner.npy", mode="w+", dtype="int64", shape=(count[0],)),
        "field": np.lib.format.open_memmap(root / f"{prefix}-field.npy", mode="w+", dtype="int64", shape=(count[0],)),
    }
    if prefix == "categorical":
        extras = {
            "value": np.lib.format.open_memmap(
                root / f"{prefix}-value.npy", mode="w+", dtype="int64", shape=(count[0],)
            ),
            "ordinal": np.lib.format.open_memmap(
                root / f"{prefix}-ordinal.npy", mode="w+", dtype="int64", shape=(count[0],)
            ),
        }
    else:
        extras = {
            "value": np.lib.format.open_memmap(
                root / f"{prefix}-value.npy", mode="w+", dtype="float32", shape=(count[0],)
            ),
            "valid": np.lib.format.open_memmap(
                root / f"{prefix}-valid.npy", mode="w+", dtype="bool", shape=(count[0],)
            ),
        }
    offsets = np.zeros(owners + 1, dtype=np.int64)
    position = 0
    for batch in connection.execute(f"SELECT * FROM ({query})").fetch_record_batch(5000):
        rows = batch.to_pylist()
        end = position + len(rows)
        columns["owner"][position:end] = [row["owner_index"] for row in rows]
        columns["field"][position:end] = [row["field_id"] for row in rows]
        if prefix == "categorical":
            extras["value"][position:end] = [row["value_id"] for row in rows]
            extras["ordinal"][position:end] = [row["ordinal"] for row in rows]
        else:
            extras["value"][position:end] = [row["numeric"] if row["numeric"] is not None else 0 for row in rows]
            extras["valid"][position:end] = [bool(row["valid"]) for row in rows]
        for owner in [row["owner_index"] for row in rows]:
            offsets[owner + 1] += 1
        position = end
    if position != count[0]:
        raise ValueError(f"feature row count mismatch: {prefix}")
    offsets[:] = torch.from_numpy(offsets).cumsum(0).numpy()
    if prefix == "categorical":
        refs = [
            _save(root, f"{prefix}-owner", columns["owner"]),
            _save(root, f"{prefix}-field", columns["field"]),
            _save(root, f"{prefix}-value", extras["value"]),
            _save(root, f"{prefix}-ordinal", extras["ordinal"]),
            _save(root, f"{prefix}-indptr", offsets),
        ]
    else:
        refs = [
            _save(root, f"{prefix}-owner", columns["owner"]),
            _save(root, f"{prefix}-field", columns["field"]),
            _save(root, f"{prefix}-value", extras["value"]),
            _save(root, f"{prefix}-valid", extras["valid"]),
        ]
    return refs, count[0]


def prepare_features(
    connection: duckdb.DuckDBPyConnection, snapshot: Path, root: Path, node_count: int, statement_count: int
) -> tuple[list[FeatureFieldSpec], FeatureStoreArtifact]:
    """Fit categorical/numeric fields on training owners; held-out strings map to unknown=0."""
    connection.read_parquet(str(snapshot / "attributes/*.parquet")).create_view("attributes")
    connection.read_parquet(str(root / "topology.parquet")).create_view("splits")
    connection.execute("CREATE TABLE training_statements AS SELECT statement_id FROM splits WHERE split=0")
    connection.execute(
        "CREATE TABLE training_nodes AS SELECT subject AS id FROM statements JOIN training_statements t ON statements.id=t.statement_id UNION SELECT object FROM statements JOIN training_statements t ON statements.id=t.statement_id"
    )
    connection.execute("""CREATE VIEW feature_events AS SELECT a.*,
        CASE WHEN numeric IS NOT NULL THEN 'numeric'
          WHEN text IS NOT NULL AND length(text)>128 THEN 'text' ELSE 'categorical' END AS kind
        FROM attributes a WHERE value_type NOT IN ('object','list','null','nonfinite')""")
    connection.execute("""CREATE TABLE fitted_fields AS
        SELECT row_number() OVER (ORDER BY owner_kind,path,kind)+1 AS idx, owner_kind,path,kind,
            count(*) AS count, coalesce(avg(numeric),0) AS mean,
            greatest(coalesce(stddev_pop(numeric),1),1e-8) AS scale
        FROM feature_events WHERE
          (owner_kind='node' AND owner_id IN (SELECT id FROM training_nodes)) OR
          (owner_kind='statement' AND owner_id IN (SELECT statement_id FROM training_statements))
        GROUP BY owner_kind,path,kind""")
    connection.execute("""CREATE TABLE fitted_categories AS
        SELECT owner_kind,path,raw_json,row_number() OVER (PARTITION BY owner_kind,path ORDER BY raw_json)+1 AS value_id
        FROM (SELECT DISTINCT e.owner_kind,e.path,e.raw_json FROM feature_events e
          WHERE e.kind='categorical' AND (
            (owner_kind='node' AND owner_id IN (SELECT id FROM training_nodes)) OR
            (owner_kind='statement' AND owner_id IN (SELECT statement_id FROM training_statements))))""")
    fields = [
        FeatureFieldSpec(index=index, owner_kind=owner, path=path, kind=kind, count=count, mean=mean, scale=scale)
        for index, owner, path, kind, count, mean, scale in connection.execute(
            "SELECT * FROM fitted_fields ORDER BY idx"
        ).fetchall()
    ]
    for table in ("fitted_fields", "fitted_categories"):
        connection.execute(f"COPY {table} TO '{root / (table + '.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    # Shifted owner index: nodes use [0,node_count), statements use [node_count,node_count+statement_count).
    base = f"""SELECT CASE WHEN e.owner_kind='node' THEN e2.idx ELSE {node_count}+s.index END AS owner_index,
        coalesce(f.idx,0) AS field_id,e.ordinal,e.kind,coalesce(c.value_id,0) AS value_id,e.text,e.raw_json,
        CASE WHEN e.numeric IS NOT NULL THEN (e.numeric-coalesce(f.mean,0))/coalesce(f.scale,1) ELSE NULL END AS numeric,
        e.numeric IS NOT NULL AS valid
        FROM feature_events e LEFT JOIN entities e2 ON e.owner_kind='node' AND e.owner_id=e2.id
        LEFT JOIN splits s ON e.owner_kind='statement' AND e.owner_id=s.statement_id
        LEFT JOIN fitted_fields f ON e.owner_kind=f.owner_kind AND e.path=f.path AND e.kind=f.kind
        LEFT JOIN fitted_categories c ON e.owner_kind=c.owner_kind AND e.path=c.path AND e.raw_json=c.raw_json
        WHERE (e.owner_kind='node' AND e2.idx IS NOT NULL) OR (e.owner_kind='statement' AND s.index IS NOT NULL)"""
    categorical_refs, categorical_count = _write_ordered(
        connection,
        f"SELECT owner_index,field_id,ordinal,value_id FROM ({base}) WHERE kind='categorical' ORDER BY owner_index,field_id,ordinal",
        root,
        "categorical",
        node_count + statement_count,
    )
    numeric_refs, numeric_count = _write_ordered(
        connection,
        f"SELECT owner_index,field_id,0 AS ordinal,numeric,valid FROM ({base}) WHERE kind='numeric' ORDER BY owner_index,field_id",
        root,
        "numeric",
        node_count + statement_count,
    )
    artifact = FeatureStoreArtifact(
        categorical=CategoricalCsr(
            owner_id=categorical_refs[0],
            field_id=categorical_refs[1],
            value_id=categorical_refs[2],
            ordinal=categorical_refs[3],
            indptr=categorical_refs[4],
        ),
        numeric=NumericCoo(
            owner_id=numeric_refs[0], field_id=numeric_refs[1], value=numeric_refs[2], valid=numeric_refs[3]
        ),
    )
    # Raw events (including text) remain in Parquet for future providers; unknown channels stay explicit.
    for owner, vocabulary, identity in (("node", "entities", "id"), ("statement", "splits", "statement_id")):
        query = f"""SELECT v.{identity} AS owner_id, coalesce(f.idx,0) AS field_id,
          e.path,e.ordinal,e.kind,coalesce(c.value_id,0) AS value_id,e.text,e.raw_json,
          CASE WHEN e.numeric IS NOT NULL THEN (e.numeric-coalesce(f.mean,0))/coalesce(f.scale,1) ELSE NULL END AS numeric,
          e.numeric IS NOT NULL AS valid
          FROM feature_events e JOIN {vocabulary} v ON e.owner_id=v.{identity}
          LEFT JOIN fitted_fields f ON e.owner_kind=f.owner_kind AND e.path=f.path AND e.kind=f.kind
          LEFT JOIN fitted_categories c ON e.owner_kind=c.owner_kind AND e.path=c.path AND e.raw_json=c.raw_json
          WHERE e.owner_kind='{owner}' ORDER BY owner_id,field_id,e.ordinal"""
        connection.execute(
            f"COPY ({query}) TO '{root / (owner + '-features.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    _ = (categorical_count, numeric_count)
    return fields, artifact
