"""Equivalence tests: vectorized/batched evaluation vs the frozen reference.

Why this exists: evaluation reports are scientific results. Faster message and
positive-index construction, plus relation-batched scoring, must not change
integer ranks, sampled RNG consumption, aggregate metrics, or group metrics.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from perf_support import synthetic_graph  # noqa: E402
from reference_evaluation import _full_messages as full_messages_reference  # noqa: E402
from reference_evaluation import evaluate_dataset_reference  # noqa: E402
from support import write_table  # noqa: E402

from ced.model.evaluation import _build_positives, _full_messages, _score_relation_batch, evaluate_dataset  # noqa: E402
from ced.model.prepare import save_tensor  # noqa: E402
from ced.model.shards import _edge_type_ids  # noqa: E402
from ced.model.training import build_model  # noqa: E402


def _assert_full_messages_match(seed: int, *, nodes: int = 40, edges: int = 240, relations: int = 3) -> None:
    graph = synthetic_graph(nodes=nodes, edges=edges, relations=relations, type_sets=3, seed=seed)
    edge_types = _edge_type_ids(graph["edge_index"], graph["relation_id"], graph["split"], graph["type_set_id"])
    arguments = (
        torch.from_numpy(graph["edge_index"]),
        torch.from_numpy(graph["relation_id"]),
        torch.from_numpy(graph["split"].astype(np.int64)),
        torch.from_numpy(edge_types),
    )
    expected = full_messages_reference(*arguments)
    produced = _full_messages(*arguments)
    assert produced[-1] == expected[-1]
    for want, got in zip(expected[:-1], produced[:-1], strict=True):
        assert torch.equal(want, got)


def test_positive_index_matches_reference_loop() -> None:
    # Duplicate edges and reciprocal edges must collapse to the same sets; an
    # overflow-free lexsort also handles sparse, non-contiguous node/relation ids.
    edge_index = torch.tensor([[10, 10, 30, 20, 10], [20, 20, 10, 10, 30]])
    relations = torch.tensor([4, 4, 9, 4, 4])
    expected: dict[tuple[int, int], set[int]] = {}
    for head, relation, tail in zip(edge_index[0].tolist(), relations.tolist(), edge_index[1].tolist(), strict=True):
        expected.setdefault((head, relation), set()).add(tail)
        expected.setdefault((tail, relation), set()).add(head)
    assert _build_positives(edge_index, relations) == expected
    assert _build_positives(torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.long)) == {}


def test_full_messages_preserve_first_occurrence_ids() -> None:
    # Repeated and reverse-equivalent type triples exercise first-occurrence
    # numbering, which determines statement-feature rows and model numerics.
    _assert_full_messages_match(3)


def test_full_messages_match_on_random_graphs() -> None:
    for seed in (5, 7, 11):
        _assert_full_messages_match(seed)


def test_full_messages_rejects_missing_training_topology() -> None:
    edge_index = torch.tensor([[0], [1]])
    relations = torch.tensor([0])
    edge_types = torch.tensor([0])
    for implementation in (full_messages_reference, _full_messages):
        with pytest.raises(ValueError, match="at least one training statement"):
            implementation(edge_index, relations, torch.tensor([2]), edge_types)


@pytest.mark.parametrize("ranking", ["exact", "sampled"])
def test_evaluation_report_identical_to_reference(
    tmp_path: Path, synthetic_dataset: Callable[..., int], ranking: Literal["exact", "sampled"]
) -> None:
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    expected = evaluate_dataset_reference(
        root, split="test", seed=3, device="cpu", ranking=ranking, sampled_candidates=2
    )
    produced = evaluate_dataset(root, split="test", seed=3, device="cpu", ranking=ranking, sampled_candidates=2)
    assert produced == expected


def _multi_target_dataset(root: Path) -> None:
    root.mkdir(parents=True)
    statements = 10
    topology = root / "topology.parquet"
    pq.write_table(
        pa.table(
            {
                "index": pa.array(list(range(statements)), pa.int64()),
                "statement_id": pa.array(list(range(statements)), pa.int64()),
                "group_id": pa.array([0] * statements, pa.int64()),
                "split": pa.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2], pa.int8()),
                "head": pa.array([0, 1, 2, 0, 3, 4, 1, 5, 2, 4], pa.int64()),
                "tail": pa.array([1, 2, 3, 2, 4, 5, 0, 0, 4, 5], pa.int64()),
                "relation_id": pa.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0], pa.int64()),
            }
        ),
        topology,
        compression="zstd",
    )
    entities = root / "entities.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array([f"CED:Family_{index}" for index in range(6)]),
                "type_set_id": pa.array([0] * 6, pa.int64()),
            }
        ),
        entities,
        compression="zstd",
    )
    write_table(root / "relations.parquet", "id", ["p0", "p1"])
    write_table(root / "types.parquet", "categories", [["biolink:Gene"]])
    edge_types = _edge_type_ids(
        np.array([[0, 1, 2, 0, 3, 4, 1, 5, 2, 4], [1, 2, 3, 2, 4, 5, 0, 0, 4, 5]], dtype=np.int64),
        np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64),
        np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int8),
        np.zeros(6, dtype=np.int64),
    )
    references = {
        "head": save_tensor(root, "head", np.array([0, 1, 2, 0, 3, 4, 1, 5, 2, 4], dtype=np.int64)),
        "tail": save_tensor(root, "tail", np.array([1, 2, 3, 2, 4, 5, 0, 0, 4, 5], dtype=np.int64)),
        "relation_id": save_tensor(root, "relation_id", np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64)),
        "split": save_tensor(root, "split", np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int8)),
        "type_set_id": save_tensor(root, "type_set_id", np.zeros(6, dtype=np.int64)),
        "edge_type_id": save_tensor(root, "edge_type_id", edge_types),
    }
    document = {
        "relations": 2,
        "entities": 6,
        "statements": statements,
        "edge_type_count": 1 + int(edge_types.max()),
        "tensors": {name: reference.model_dump(mode="json") for name, reference in references.items()},
    }
    (root / "manifest.json").write_text(json.dumps(document))
    pq.write_table(
        pa.table({"idx": pa.array([], pa.int64()), "kind": pa.array([], pa.string())}),
        root / "fitted_fields.parquet",
    )
    pq.write_table(
        pa.table({"field_id": pa.array([], pa.int64()), "value_id": pa.array([], pa.int64())}),
        root / "fitted_categories.parquet",
    )


@pytest.mark.parametrize("ranking", ["exact", "sampled"])
def test_multi_target_report_identical_after_relation_grouping(
    tmp_path: Path, ranking: Literal["exact", "sampled"]
) -> None:
    # Four test targets across two relations catch rank-order and sampled-RNG
    # drift that the one-target contract fixture cannot expose.
    root = tmp_path / "dataset"
    _multi_target_dataset(root)
    expected = evaluate_dataset_reference(
        root, split="test", seed=17, device="cpu", ranking=ranking, sampled_candidates=1
    )
    produced = evaluate_dataset(root, split="test", seed=17, device="cpu", ranking=ranking, sampled_candidates=1)
    assert produced == expected


def test_relation_batched_scores_are_bitwise_identical(tmp_path: Path) -> None:
    # Batched GEMMs must not perturb scores: exact filtered ranks compare each
    # score against the target score, so even tiny drift can change a tie.
    root = tmp_path / "dataset"
    _multi_target_dataset(root)
    document = json.loads((root / "manifest.json").read_text())
    model = build_model(document, root, edge_type_count=4, seed=23)
    state = torch.randn(6, 32, generator=torch.Generator().manual_seed(29))
    heads, tails, relation = [0, 2, 4], [1, 3, 5], 1
    tail_scores, head_scores = _score_relation_batch(model, state, heads, tails, relation)
    relation_ids = torch.full((6,), relation, dtype=torch.long)
    for row, (head, tail) in enumerate(zip(heads, tails, strict=True)):
        expected_tail = model.decoder.score(state[head].expand_as(state), relation_ids, state)
        expected_head = model.decoder.score(state, relation_ids, state[tail].expand_as(state))
        assert torch.equal(tail_scores[row], expected_tail)
        assert torch.equal(head_scores[row], expected_head)
