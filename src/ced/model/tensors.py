from __future__ import annotations

import math
import mmap
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from safetensors import TensorSpec, safe_open, serialize_file
from safetensors.numpy import save_file

SUPPORTED_DTYPES = ("int64", "float32", "bool", "int8")
DEFAULT_COLUMN = "value"


def tensor_dtype(dtype: str) -> np.dtype:
    """Return a supported tensor dtype or reject silent narrowing and widening."""
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported tensor dtype: {dtype}; expected one of {', '.join(SUPPORTED_DTYPES)}")
    return np.dtype(dtype)


def write_tensor(path: Path, values: np.ndarray, *, column: str = DEFAULT_COLUMN) -> None:
    """Atomically write one named, C-contiguous tensor to a safetensors file."""
    array = np.ascontiguousarray(values)
    tensor_dtype(array.dtype.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    save_file({column: array}, temporary)
    temporary.replace(path)


def write_tensor_stream(
    path: Path,
    batches: Iterable[np.ndarray],
    *,
    shape: Sequence[int],
    dtype: str,
    column: str = DEFAULT_COLUMN,
) -> None:
    """Stream batches through a mapped staging file into the official safetensors writer."""
    expected_dtype = tensor_dtype(dtype)
    little_dtype = expected_dtype.newbyteorder("<")
    expected_count = math.prod(shape)
    raw = path.with_name(f"{path.name}.raw.tmp-{os.getpid()}")
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with raw.open("wb") as stream:
            for values in batches:
                array = np.ascontiguousarray(values)
                if array.dtype.name != dtype:
                    raise ValueError(f"tensor dtype mismatch: {path.name} is {array.dtype}, expected {dtype}")
                array = array.astype(little_dtype, copy=False)
                array.tofile(stream)
                count += array.size
        if count != expected_count:
            raise ValueError(f"tensor shape mismatch: {path.name} holds {count} values, expected {expected_count}")
        if count == 0:
            write_tensor(path, np.empty(shape, dtype=little_dtype), column=column)
            return
        with raw.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
            array = np.frombuffer(buffer, dtype=little_dtype).reshape(shape)
            try:
                spec = TensorSpec(
                    dtype=array.dtype.name,
                    shape=array.shape,
                    data_ptr=array.ctypes.data,
                    data_len=array.nbytes,
                )
                serialize_file({column: spec}, temporary)
            finally:
                del array
        temporary.replace(path)
    finally:
        raw.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def read_tensor(path: Path, *, column: str, dtype: str, shape: Sequence[int]) -> np.ndarray:
    """Read one named tensor into an owned C-contiguous array of the declared dtype and shape."""
    if not path.is_file():
        raise FileNotFoundError(f"dataset tensor is missing: {path.name}")
    expected_dtype = tensor_dtype(dtype)
    with safe_open(path, framework="np") as handle:
        columns = set(handle.keys())
        if column not in columns:
            raise ValueError(f"tensor column is missing: {path.name}:{column}")
        values = handle.get_tensor(column)
    if values.dtype != expected_dtype:
        raise ValueError(f"tensor dtype mismatch: {path.name}:{column} is {values.dtype}, expected {expected_dtype}")
    expected_shape = tuple(shape)
    if values.shape != expected_shape:
        raise ValueError(
            f"tensor shape mismatch: {path.name}:{column} holds shape {values.shape}, expected {expected_shape}"
        )
    if not values.flags.writeable:
        values = values.copy()
    return np.ascontiguousarray(values)
