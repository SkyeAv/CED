from __future__ import annotations

import hashlib
from collections.abc import Mapping

from ced.config.models import SplitSettings


def statement_group(
    subject: str,
    predicate: str,
    object_id: str,
    *,
    symmetric: set[str] | None = None,
    inverses: Mapping[str, str] | None = None,
) -> str:
    if inverses and predicate in inverses:
        inverse = inverses[predicate]
        if inverse < predicate:
            subject, object_id, predicate = object_id, subject, inverse
    if symmetric and predicate in symmetric:
        subject, object_id = sorted((subject, object_id))
    return hashlib.sha256(f"{subject}\0{predicate}\0{object_id}".encode()).hexdigest()


def group_split(group: str, settings: SplitSettings, *, schema: bool = False) -> int:
    if schema:
        return 3
    value = int.from_bytes(hashlib.sha256(f"{settings.seed}\0{group}".encode()).digest()[:8], "big") / 2**64
    return 0 if value < settings.train else 1 if value < settings.train + settings.validation else 2
