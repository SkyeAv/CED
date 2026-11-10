from __future__ import annotations

import hashlib
from collections.abc import Mapping
from fractions import Fraction

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


def split_bounds(settings: SplitSettings) -> tuple[str, str]:
    """Hex-encoded 64-bit train and validation thresholds for the SQL split CASE.

    DuckDB compares the first 16 hex characters of ``sha256(seed\0group)`` as a
    UBIGINT, so the thresholds are bound as hex strings; the integers exceed
    BIGINT and cannot be passed as Python integers.
    """
    train_bound = int(Fraction(settings.train) * 2**64)
    valid_bound = int(Fraction(settings.train + settings.validation) * 2**64)
    return format(train_bound, "x"), format(valid_bound, "x")


def group_split(group: str, settings: SplitSettings, *, schema: bool = False) -> int:
    """Assign one split group to train/validation/test.

    The digest prefix is compared as an exact 64-bit integer against
    ``int(Fraction(ratio) * 2**64)``. ``prepare_dataset`` implements the same rule
    in SQL, and tests/test_prepare_topology.py pins the two engines together; a
    rewrite of one without the other would silently move statements between train
    and held-out splits. Measured against the previous float comparison
    (``digest[:8] / 2**64 < ratio``): 0 disagreements over 600k random digests for
    the ratio sets (0.8, 0.1), (0.7, 0.2) and (0.6, 0.2).
    """
    if schema:
        return 3
    value = int.from_bytes(hashlib.sha256(f"{settings.seed}\0{group}".encode()).digest()[:8], "big")
    train_bound, valid_bound = (int(bound, 16) for bound in split_bounds(settings))
    return 0 if value < train_bound else 1 if value < valid_bound else 2
