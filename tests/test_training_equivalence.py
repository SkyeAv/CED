"""Equivalence tests: sync-free training internals vs frozen references.

Why this exists: removing .cpu() round-trips and per-field host syncs must not
perturb loss values or gradients - training trajectories and checkpointed
metrics depend on them bitwise on CPU. Each test compares values AND gradients
against the pre-optimization reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from reference_training import _auxiliary_tensors_reference, group_loss_reference

from ced.model.feature_index import FeatureIndex
from ced.model.losses import AuxiliaryAttributeDecoder
from ced.model.prepare import save_tensor
from ced.model.training import _auxiliary_tensors


def _decoder_pair(seed: int) -> tuple[AuxiliaryAttributeDecoder, AuxiliaryAttributeDecoder]:
    torch.manual_seed(seed)
    produced = AuxiliaryAttributeDecoder(8, {1: 3, 2: 2}, [5])
    torch.manual_seed(seed)
    reference = AuxiliaryAttributeDecoder(8, {1: 3, 2: 2}, [5])
    return reference, produced


def _compare_group_loss(
    seed: int,
    states: torch.Tensor,
    fields: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor | None,
) -> None:
    reference, produced = _decoder_pair(seed)
    index = produced.categorical_index if valid is None else produced.numeric_index
    heads = produced.categorical_heads if valid is None else produced.numeric_heads
    reference_index = reference.categorical_index if valid is None else reference.numeric_index
    reference_heads = reference.categorical_heads if valid is None else reference.numeric_heads
    expected = group_loss_reference(states, fields, targets, reference_index, reference_heads, valid)
    actual = produced._group_loss(states, fields, targets, index, heads, valid)
    assert torch.equal(expected, actual)


def test_group_loss_categorical_identical() -> None:
    generator = torch.Generator().manual_seed(3)
    states = torch.randn(12, 8, generator=generator)
    # Field 7 has no head; fields 1/2 exercise both head slots.
    fields = torch.tensor([1, 2, 1, 7, 2, 1, 2, 7, 1, 2, 1, 2])
    targets = torch.tensor([0, 1, 2, 0, 0, 1, 1, 0, 0, 0, 2, 1])
    _compare_group_loss(5, states, fields, targets, None)


def test_group_loss_numeric_masked_identical() -> None:
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(10, 8, generator=generator)
    # Field 5 rows are all invalid -> the historical per-field .any() skipped it;
    # field 9 has no head; field 5 row 7 is valid.
    fields = torch.tensor([5, 5, 9, 5, 9, 5, 9, 5, 9, 5])
    targets = torch.randn(10, generator=generator)
    valid = torch.tensor([False, False, False, False, False, False, False, True, False, True])
    _compare_group_loss(11, states, fields, targets, valid)


def test_group_loss_all_invalid_identical() -> None:
    states = torch.randn(4, 8, generator=torch.Generator().manual_seed(13))
    fields = torch.tensor([5, 5, 5, 5])
    targets = torch.randn(4, generator=torch.Generator().manual_seed(17))
    valid = torch.zeros(4, dtype=torch.bool)
    _compare_group_loss(19, states, fields, targets, valid)


def test_group_loss_empty_fields_identical() -> None:
    states = torch.randn(4, 8, generator=torch.Generator().manual_seed(23))
    empty_long = torch.empty(0, dtype=torch.long)
    _compare_group_loss(29, states, empty_long, torch.empty(0, dtype=torch.long), None)
    _compare_group_loss(29, states, empty_long, torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.bool))


def test_group_loss_gradients_identical() -> None:
    fields = torch.tensor([1, 2, 1, 2, 1])
    targets = torch.tensor([0, 1, 2, 0, 1])
    collected = []
    for use_reference in (True, False):
        reference, produced = _decoder_pair(37)
        model = reference if use_reference else produced
        states = torch.randn(5, 8, generator=torch.Generator().manual_seed(41)).requires_grad_(True)
        if use_reference:
            loss = group_loss_reference(states, fields, targets, model.categorical_index, model.categorical_heads)
        else:
            loss = model._group_loss(states, fields, targets, model.categorical_index, model.categorical_heads)
        loss.backward()
        grads: dict[str, torch.Tensor | None] = {"states": states.grad}
        grads.update({name: parameter.grad for name, parameter in model.named_parameters()})
        collected.append(grads)
    want, got = collected
    assert set(want) == set(got)
    for name in want:
        # Numeric heads stay unexercised by the categorical loss: None in both.
        if want[name] is None or got[name] is None:
            assert want[name] is None, name
            assert got[name] is None, name
            continue
        assert torch.equal(want[name], got[name]), name


def _feature_index(root: Path) -> tuple[FeatureIndex, dict]:
    root.mkdir(parents=True, exist_ok=True)
    categorical = {
        "owner_id": save_tensor(root, "cat-owner", np.array([0, 0, 1, 3], dtype=np.int64)),
        "field_id": save_tensor(root, "cat-field", np.array([1, 2, 1, 2], dtype=np.int64)),
        "value_id": save_tensor(root, "cat-value", np.array([0, 0, 1, 0], dtype=np.int64)),
    }
    numeric = {
        "owner_id": save_tensor(root, "num-owner", np.array([0, 3], dtype=np.int64)),
        "field_id": save_tensor(root, "num-field", np.array([5, 5], dtype=np.int64)),
        "value": save_tensor(root, "num-value", np.array([0.5, -0.25], dtype=np.float32)),
        "valid": save_tensor(root, "num-valid", np.array([True, True])),
    }
    document = {
        "entities": 3,
        "features": {
            "categorical": {name: ref.model_dump(mode="json") for name, ref in categorical.items()},
            "numeric": {name: ref.model_dump(mode="json") for name, ref in numeric.items()},
        },
    }
    return FeatureIndex(root, document), document


def test_auxiliary_tensors_identical(tmp_path: Path) -> None:
    features, _ = _feature_index(tmp_path / "features")
    features.to(torch.device("cpu"))
    state = torch.randn(3, 8, generator=torch.Generator().manual_seed(43))
    global_nodes = torch.tensor([0, 1, 2], dtype=torch.long)
    statements = torch.tensor([0], dtype=torch.long)
    expected = _auxiliary_tensors_reference(features, state, global_nodes, statements)
    produced = _auxiliary_tensors(features, state, global_nodes, statements)
    for want, got in zip(expected, produced, strict=True):
        assert torch.equal(want, got)


def test_auxiliary_tensors_empty_statements_identical(tmp_path: Path) -> None:
    features, _ = _feature_index(tmp_path / "features")
    state = torch.randn(3, 8, generator=torch.Generator().manual_seed(47))
    global_nodes = torch.tensor([0, 1, 2], dtype=torch.long)
    statements = torch.empty(0, dtype=torch.long)
    expected = _auxiliary_tensors_reference(features, state, global_nodes, statements)
    produced = _auxiliary_tensors(features, state, global_nodes, statements)
    for want, got in zip(expected, produced, strict=True):
        assert torch.equal(want, got)
