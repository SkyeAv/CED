from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

from ced.acquisition.base import sha256_file
from ced.model.schemas import CategoricalCsr, FeatureFieldSpec, FeatureStoreArtifact, NumericCoo, TensorRef


def write_indptr(connection: duckdb.DuckDBPyConnection, source: Path, owners: int, destination: Path) -> TensorRef:
    """Write CSR offsets for every owner in [0, owners] from an owner-ordered column file.

    ``indptr[o]`` is the number of rows whose owner index is below ``o``, so owners
    with no rows contribute explicit zeros and the CSR stays dense over all owners.
    The window runs over ``generate_series`` left-joined to per-owner counts, which
    DuckDB streams and spills under the configured memory limit instead of holding
    a Python histogram of every owner in RAM.
    """
    connection.execute(
        f"""
        COPY (
          SELECT CAST(coalesce(sum(n) OVER (ORDER BY owner ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0)
                     AS BIGINT) AS value
          FROM (SELECT g.owner AS owner, coalesce(p.n, 0) AS n
                FROM generate_series(0, {owners}) AS g(owner)
                LEFT JOIN (SELECT owner_index, count(*) AS n
                           FROM read_parquet('{source}') GROUP BY owner_index) p
                  ON p.owner_index = g.owner)
          ORDER BY owner)
        TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
    )
    values = pq.read_table(destination, columns=["value"]).column("value").to_numpy(zero_copy_only=False)
    rows = connection.execute(f"SELECT count(*) FROM read_parquet('{source}')").fetchone()
    assert rows is not None
    if len(values) != owners + 1 or values[0] != 0 or values[-1] != rows[0] or bool((np.diff(values) < 0).any()):
        raise ValueError(f"categorical indptr violates CSR invariants: {destination.name}")
    return TensorRef(
        path=destination.name,
        column="value",
        dtype="int64",
        shape=[owners + 1],
        sha256=sha256_file(destination),
    )


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
    owners = node_count + statement_count
    categorical_path = root / "categorical.parquet"
    connection.execute(
        f"""COPY (SELECT CAST(owner_index AS BIGINT) AS owner_index, CAST(field_id AS BIGINT) AS field_id,
                     CAST(value_id AS BIGINT) AS value_id, CAST(ordinal AS BIGINT) AS ordinal
              FROM ({base}) WHERE kind='categorical' ORDER BY owner_index, field_id, ordinal)
           TO '{categorical_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
    )
    categorical_rows = connection.execute(f"SELECT count(*) FROM read_parquet('{categorical_path}')").fetchone()
    assert categorical_rows is not None
    categorical_digest = sha256_file(categorical_path)

    def categorical_ref(column: str) -> TensorRef:
        return TensorRef(
            path="categorical.parquet",
            column=column,
            dtype="int64",
            shape=[categorical_rows[0]],
            sha256=categorical_digest,
        )

    numeric_path = root / "numeric.parquet"
    connection.execute(
        f"""COPY (SELECT CAST(owner_index AS BIGINT) AS owner_index, CAST(field_id AS BIGINT) AS field_id,
                     CAST(coalesce(numeric, 0) AS REAL) AS value, CAST(valid AS BOOLEAN) AS valid
              FROM ({base}) WHERE kind='numeric' ORDER BY owner_index, field_id)
           TO '{numeric_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
    )
    numeric_rows = connection.execute(f"SELECT count(*) FROM read_parquet('{numeric_path}')").fetchone()
    assert numeric_rows is not None
    numeric_digest = sha256_file(numeric_path)

    def numeric_ref(column: str, dtype: str) -> TensorRef:
        return TensorRef(
            path="numeric.parquet",
            column=column,
            dtype=dtype,
            shape=[numeric_rows[0]],
            sha256=numeric_digest,
        )

    artifact = FeatureStoreArtifact(
        categorical=CategoricalCsr(
            owner_id=categorical_ref("owner_index"),
            field_id=categorical_ref("field_id"),
            value_id=categorical_ref("value_id"),
            ordinal=categorical_ref("ordinal"),
            indptr=write_indptr(connection, categorical_path, owners, root / "categorical-indptr.parquet"),
        ),
        numeric=NumericCoo(
            owner_id=numeric_ref("owner_index", "int64"),
            field_id=numeric_ref("field_id", "int64"),
            value=numeric_ref("value", "float32"),
            valid=numeric_ref("valid", "bool"),
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
    return fields, artifact
