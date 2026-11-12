"""Equivalence tests: segmented encoder forward vs the frozen per-type-mask reference.

Why this exists: the encoder restructure must not perturb training numerics at
all - same parameters, same state_dict, bitwise-identical forward on CPU. Any
drift would silently change every downstream run without a version bump.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from perf_support import synthetic_graph  # noqa: E402
from reference_encoder import RelationalAttentionEncoderReference  # noqa: E402

from ced.model.encoder import RelationalAttentionEncoder  # noqa: E402


def _pair(seed: int, dimension: int, statement_dimension: int, edge_types: int, layers: int):
    torch.manual_seed(seed)
    reference = RelationalAttentionEncoderReference(
        dimension, statement_dimension, edge_types=edge_types, layers=layers
    )
    torch.manual_seed(seed)
    produced = RelationalAttentionEncoder(dimension, statement_dimension, edge_types=edge_types, layers=layers)
    return reference, produced


def _inputs(seed: int, nodes: int, edges: int, edge_types: int, dimension: int, statement_dimension: int):
    generator = torch.Generator().manual_seed(seed)
    graph = synthetic_graph(nodes=nodes, edges=edges, relations=edge_types, type_sets=2, seed=seed)
    edge_type_id = torch.from_numpy(graph["edge_type_id"] % edge_types)
    return (
        torch.randn(nodes, dimension, generator=generator),
        torch.from_numpy(graph["edge_index"]),
        edge_type_id,
        torch.randn(edges, statement_dimension, generator=generator),
    )


def test_encoder_state_dict_and_init_are_unchanged() -> None:
    reference, produced = _pair(3, 8, 4, edge_types=3, layers=2)
    reference_state = reference.state_dict()
    produced_state = produced.state_dict()
    assert list(reference_state) == list(produced_state)
    for name, values in reference_state.items():
        assert torch.equal(values, produced_state[name]), name


def test_encoder_forward_is_bitwise_identical() -> None:
    reference, produced = _pair(5, 16, 8, edge_types=4, layers=2)
    nodes, edge_index, edge_type_id, statements = _inputs(11, 60, 400, 4, 16, 8)
    with torch.no_grad():
        expected = reference(nodes, edge_index, edge_type_id, statements)
        actual = produced(nodes, edge_index, edge_type_id, statements)
    assert torch.equal(expected, actual)


def test_encoder_forward_identical_with_sparse_type_ids() -> None:
    # Only a subset of parameter groups is exercised; raw type ids index the
    # ModuleLists, so gaps must not shift anything.
    reference, produced = _pair(7, 8, 4, edge_types=6, layers=2)
    nodes, edge_index, edge_type_id, statements = _inputs(13, 40, 200, 6, 8, 4)
    edge_type_id = torch.where(edge_type_id % 3 == 0, torch.zeros_like(edge_type_id), torch.full_like(edge_type_id, 5))
    with torch.no_grad():
        expected = reference(nodes, edge_index, edge_type_id, statements)
        actual = produced(nodes, edge_index, edge_type_id, statements)
    assert torch.equal(expected, actual)


def test_encoder_forward_identical_without_edges() -> None:
    reference, produced = _pair(17, 8, 4, edge_types=2, layers=2)
    nodes = torch.randn(5, 8, generator=torch.Generator().manual_seed(19))
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_type_id = torch.zeros(0, dtype=torch.long)
    statements = torch.zeros((0, 4))
    with torch.no_grad():
        expected = reference(nodes, edge_index, edge_type_id, statements)
        actual = produced(nodes, edge_index, edge_type_id, statements)
    assert torch.equal(expected, actual)


def test_encoder_gradients_match() -> None:
    # Training depends on gradients, not just forward values: the segment
    # rewrite must produce identical parameter grads on CPU.
    reference, produced = _pair(23, 8, 4, edge_types=3, layers=2)
    nodes, edge_index, edge_type_id, statements = _inputs(29, 30, 120, 3, 8, 4)
    expected = reference(nodes, edge_index, edge_type_id, statements).square().mean()
    actual = produced(nodes, edge_index, edge_type_id, statements).square().mean()
    expected.backward()
    actual.backward()
    for (name, want), (_, got) in zip(reference.named_parameters(), produced.named_parameters(), strict=True):
        # Unexercised parameter groups (e.g. outputs beyond layers) stay None in both.
        if want.grad is None or got.grad is None:
            assert want.grad is None, name
            assert got.grad is None, name
            continue
        assert torch.equal(want.grad, got.grad), name


def test_encoder_validation_errors_preserved() -> None:
    _, produced = _pair(31, 8, 4, edge_types=2, layers=1)
    nodes = torch.randn(4, 8)
    with pytest.raises(ValueError, match="int64"):
        produced(nodes, torch.zeros((2, 1), dtype=torch.int32), torch.zeros(1, dtype=torch.long), torch.zeros((1, 4)))
    with pytest.raises(ValueError, match="unaligned"):
        produced(
            nodes,
            torch.zeros((2, 2), dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros((1, 4)),
        )


def test_encoder_equivalence_under_training_batch_shapes() -> None:
    graph = synthetic_graph(nodes=200, edges=1500, relations=5, type_sets=3, seed=37)
    edge_types = int(graph["edge_type_id"].max()) + 1
    reference, produced = _pair(41, 32, 16, edge_types=edge_types, layers=2)
    nodes = torch.randn(200, 32, generator=torch.Generator().manual_seed(43))
    statements = torch.randn(1500, 16, generator=torch.Generator().manual_seed(47))
    with torch.no_grad():
        expected = reference(
            nodes, torch.from_numpy(graph["edge_index"]), torch.from_numpy(graph["edge_type_id"]), statements
        )
        actual = produced(
            nodes, torch.from_numpy(graph["edge_index"]), torch.from_numpy(graph["edge_type_id"]), statements
        )
    assert torch.equal(expected, actual)
