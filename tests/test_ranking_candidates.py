"""Ranking modes must share typed candidate semantics, including failure cases."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from ced.model.evaluation import Ranking, _sampled_selection, evaluate_dataset
from ced.model.negatives import TypedPools


def test_exact_and_sampled_agree_when_pool_fits(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    """Sampling every candidate is the same scientific task as exact ranking."""
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    exact = evaluate_dataset(root, device="cpu", ranking="exact", seed=7)
    sampled = evaluate_dataset(root, device="cpu", ranking="sampled", sampled_candidates=1000, seed=7)
    for key in ("mrr", "hits", "support"):
        assert exact[key] == sampled[key]


@pytest.mark.parametrize("ranking", ["exact", "sampled"])
def test_missing_ranking_pool_fails_explicitly(
    tmp_path: Path, synthetic_dataset: Callable[..., int], monkeypatch: pytest.MonkeyPatch, ranking: str
) -> None:
    """An absent domain must not silently become an all-entity or target-only ranking."""
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    monkeypatch.setattr("ced.model.evaluation.typed_pools", lambda *args: TypedPools({}, {}))
    with pytest.raises(ValueError, match="relation 0 has no subject ranking candidates"):
        evaluate_dataset(root, device="cpu", ranking=cast(Ranking, ranking))


@pytest.mark.parametrize("pool", [None, np.empty(0, dtype=np.int64)])
def test_empty_selection_pool_is_rejected(pool: np.ndarray | None) -> None:
    """A missing pool cannot manufacture rank one by retaining only the target."""
    with pytest.raises(ValueError, match="candidate pool must not be empty"):
        _sampled_selection(7, set(), pool, np.random.default_rng(1), 10)


def test_selection_retains_target_and_filters_known_positives() -> None:
    """Held-out target inclusion and positive filtering are independent of pool membership."""
    selected = _sampled_selection(7, {1, 7}, np.array([1, 2, 3]), np.random.default_rng(1), 10)
    assert selected.tolist() == [2, 3, 7]
