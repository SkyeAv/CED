"""Frozen pre-optimization references for training internals (do not edit).

Verbatim copies of _owner_states/_auxiliary_tensors (training.py) and the
_group_loss loop (losses.py) as of commit d0e610f. The production versions
removed per-step .cpu() round-trips and per-field host syncs;
tests/test_training_equivalence.py proves bitwise-identical results on CPU.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ced.model.feature_index import FeatureIndex


def _owner_states_reference(
    state: torch.Tensor, mapping_ids: torch.Tensor, owner_rows: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    positions = torch.searchsorted(mapping_ids, owner_rows.to(mapping_ids.device))
    del rows
    return state[positions]


def _auxiliary_tensors_reference(
    features: FeatureIndex | None, state: torch.Tensor, global_nodes: torch.Tensor, statements: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    empty_long = torch.empty(0, dtype=torch.long, device=state.device)
    empty_float = torch.empty(0, dtype=torch.float32, device=state.device)
    empty_bool = torch.empty(0, dtype=torch.bool, device=state.device)
    if features is None or statements.numel() == 0:
        return state[:0], empty_long, empty_long, empty_long, empty_float, empty_bool
    statement_owners = features.statement_owner(statements.cpu())
    node_rows, node_fields, node_values = features.categorical_rows(global_nodes.cpu())
    statement_rows, statement_fields, statement_values = features.categorical_rows(statement_owners)
    categorical_fields = torch.cat((node_fields, statement_fields)).to(state.device)
    categorical_targets = torch.cat((node_values, statement_values)).to(state.device)
    categorical_states = torch.cat(
        (
            _owner_states_reference(state, global_nodes, features.cat_owner[node_rows], node_rows),
            _owner_states_reference(
                state, statements, features.cat_owner[statement_rows] - features.node_count, statement_rows
            ),
        )
    )
    numeric_rows, numeric_fields, numeric_values, numeric_valid = features.numeric_rows(global_nodes.cpu())
    statement_numeric_rows, statement_numeric_fields, statement_numeric_values, statement_numeric_valid = (
        features.numeric_rows(statement_owners)
    )
    numeric_fields = torch.cat((numeric_fields, statement_numeric_fields)).to(state.device)
    numeric_values = torch.cat((numeric_values, statement_numeric_values)).to(state.device)
    numeric_valid = torch.cat((numeric_valid, statement_numeric_valid)).to(state.device)
    numeric_states = torch.cat(
        (
            _owner_states_reference(state, global_nodes, features.num_owner[numeric_rows], numeric_rows),
            _owner_states_reference(
                state,
                statements,
                features.num_owner[statement_numeric_rows] - features.node_count,
                statement_numeric_rows,
            ),
        )
    )
    return (
        torch.cat((categorical_states, numeric_states)),
        categorical_fields,
        categorical_targets,
        numeric_fields,
        numeric_values,
        numeric_valid,
    )


def group_loss_reference(
    states: torch.Tensor,
    fields: torch.Tensor,
    targets: torch.Tensor,
    index: dict[int, int],
    heads: nn.ModuleList,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = states.sum() * 0
    used = torch.zeros_like(targets, dtype=torch.bool) if valid is None else valid.clone()
    for field in torch.unique(fields).tolist():
        head_index = index.get(int(field))
        if head_index is None:
            continue
        selected = fields == field
        if valid is not None:
            selected &= valid
        if not selected.any():
            continue
        used |= selected
        prediction = heads[head_index](states[selected])
        if valid is None:
            loss = loss + F.cross_entropy(prediction, targets[selected])
        else:
            loss = loss + F.mse_loss(prediction.squeeze(-1), targets[selected])
    if not used.any():
        return states.sum() * 0
    return loss / max(int(used.sum()), 1)
