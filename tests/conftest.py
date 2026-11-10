from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from support import synthetic_dataset as build_synthetic_dataset


@pytest.fixture
def synthetic_dataset() -> Callable[..., int]:
    return build_synthetic_dataset


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    return tmp_path / "dataset"
