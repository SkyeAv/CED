"""Production-shaped edge-type persistence and checkpoint evaluation contracts.

Why these tests exist: shard training shares forward parameter IDs with reverse
messages. Reconstructing a second vocabulary during full-graph evaluation changes
encoder shapes and parameter meanings, so a checkpoint must round-trip through
strict loading instead of being tested only against a fresh model.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from ced.model.evaluation import evaluate_dataset
from ced.model.training import train_batches


def test_planned_count_must_match_persisted_count(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    """A stale shard-derived plan cannot override the global persisted vocabulary."""
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    with pytest.raises(ValueError, match="planned edge type count 2 differs from persisted count 1"):
        train_batches(root, epochs=1, edge_type_count=2, device="cpu")


def test_training_checkpoint_loads_during_evaluation(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    checkpoint = tmp_path / "run" / "model.pt"

    metrics = train_batches(root, epochs=1, seed=5, checkpoint_path=checkpoint, device="cpu")
    report = evaluate_dataset(root, split="test", checkpoint=checkpoint, seed=5, device="cpu")

    assert metrics["batches"] == 1
    assert report["support"] == 2


@pytest.mark.parametrize("mutation", ["missing_count", "wrong_count", "missing_tensor"])
def test_edge_type_metadata_errors_are_explicit(
    tmp_path: Path, synthetic_dataset: Callable[..., int], mutation: str
) -> None:
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    manifest_path = root / "manifest.json"
    document = json.loads(manifest_path.read_text())
    if mutation == "missing_count":
        document.pop("edge_type_count", None)
    elif mutation == "wrong_count":
        document["edge_type_count"] = 99
    else:
        del document["tensors"]["edge_type_id"]
    manifest_path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="edge type"):
        train_batches(root, epochs=1, seed=5, device="cpu")
