from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SUPPORTED_DTYPES = ("int64", "float32", "bool", "int8")
DEFAULT_COLUMN = "value"
_ARROW_TYPES: dict[str, pa.DataType] = {
    "int64": pa.int64(),
    "float32": pa.float32(),
    "bool": pa.bool_(),
    "int8": pa.int8(),
}


def arrow_type(dtype: str) -> pa.DataType:
    """Map a supported tensor dtype name onto its Arrow type."""
    try:
        return _ARROW_TYPES[dtype]
    except KeyError:
        raise ValueError(f"unsupported tensor dtype: {dtype}; expected one of {', '.join(SUPPORTED_DTYPES)}") from None


def write_tensor(path: Path, values: np.ndarray, *, column: str = DEFAULT_COLUMN, compression: str = "zstd") -> None:
    """Write every element of `values` as one flat C-order Parquet column.

    Flat storage keeps one serialization rule for tensors of any rank: the column
    holds `prod(shape)` elements and the reader reshapes. Nothing is memory mapped,
    so writes are sequential and buffered instead of scattered dirty pages.
    """
    array = np.ascontiguousarray(values)
    arrow = arrow_type(array.dtype.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    pq.write_table(pa.table({column: pa.array(array.reshape(-1), type=arrow)}), temporary, compression=compression)
    temporary.replace(path)


def read_tensor(path: Path, *, column: str, dtype: str, shape: Sequence[int]) -> np.ndarray:
    """Read one Parquet column back into an owned C-contiguous array of `shape`.

    Every declared property is checked against the file instead of trusted, because
    a silently narrower dtype or a truncated column would corrupt indices and
    features long after preparation finished.
    """
    if not path.is_file():
        raise FileNotFoundError(f"dataset tensor is missing: {path.name}")
    expected = arrow_type(dtype)
    schema = pq.ParquetFile(path).schema_arrow
    if column not in schema.names:
        raise ValueError(f"tensor column is missing: {path.name}:{column}")
    found = schema.field(column).type
    if found != expected:
        raise ValueError(f"tensor dtype mismatch: {path.name}:{column} is {found}, expected {expected}")
    chunked = pq.read_table(path, columns=[column]).column(column)
    if chunked.null_count:
        raise ValueError(f"tensor column contains nulls: {path.name}:{column}")
    values = chunked.to_numpy(zero_copy_only=False)
    expected_count = math.prod(shape)
    if values.dtype != np.dtype(dtype):
        raise ValueError(f"tensor dtype mismatch: {path.name}:{column} decoded as {values.dtype}")
    if values.size != expected_count:
        raise ValueError(
            f"tensor shape mismatch: {path.name}:{column} holds {values.size} values, expected {expected_count}"
        )
    values = values.reshape(tuple(shape))
    if not values.flags.writeable:
        values = values.copy()
    return np.ascontiguousarray(values)
