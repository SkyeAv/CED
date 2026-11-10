from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Protocol

from ced.acquisition.pipeline import SourceArtifacts
from ced.kg.records import CanonicalBatch


class CanonicalAdapter(Protocol):
    def batches(self, artifacts: SourceArtifacts) -> Iterator[CanonicalBatch]: ...


def batches[T, U](items: Iterable[T], size: int, factory: Callable[[list[T]], U]) -> Iterator[U]:
    if size < 1:
        raise ValueError("batch size must be positive")
    chunk: list[T] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield factory(chunk)
            chunk = []
    if chunk:
        yield factory(chunk)
