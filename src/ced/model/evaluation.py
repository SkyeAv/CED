from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import torch  # pyright: ignore[reportMissingImports]

from ced.model.negatives import TypedPools, typed_pools
from ced.model.schemas import TensorRef
from ced.model.training import Device, RelationalAutoencoder, build_model, resolve_device

Ranking = Literal["exact", "sampled"]
_TARGET_BATCH_SIZE = 32


def _first_occurrence_ids(rows: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Label each row with the id that first-occurrence order would assign.

    Reproduces ``keys.setdefault(tuple(row), len(keys))`` over ``rows`` with no
    Python loop and no per-row host sync: ``unique`` sorts the distinct rows, a
    segment-min recovers each unique row's first position, and an ``argsort`` of
    those positions renumbers the uniques by appearance. The returned ids are
    therefore bitwise identical to the historical insertion order.
    """
    unique_rows, inverse = torch.unique(rows, return_inverse=True, dim=0)
    count = int(unique_rows.shape[0])
    if count == 0:
        return torch.empty(rows.shape[0], dtype=torch.long, device=rows.device), 0
    inverse = inverse.reshape(-1)
    positions = torch.arange(rows.shape[0], device=rows.device)
    first_position = torch.full((count,), rows.shape[0], dtype=torch.long, device=rows.device)
    first_position.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
    appearance = torch.argsort(first_position)
    renumbered = torch.empty(count, dtype=torch.long, device=rows.device)
    renumbered[appearance] = torch.arange(count, device=rows.device)
    return renumbered[inverse], count


def _group_pairs(pairs: np.ndarray, values: np.ndarray) -> dict[tuple[int, int], set[int]]:
    """Group values by two-column keys with vectorized sorting, not row loops."""
    if not values.size:
        return {}
    order = np.lexsort((values, pairs[:, 1], pairs[:, 0]))
    sorted_pairs = pairs[order]
    sorted_values = values[order]
    starts = np.concatenate(
        (np.array([0], dtype=np.int64), np.flatnonzero(np.any(sorted_pairs[1:] != sorted_pairs[:-1], axis=1)) + 1)
    )
    ends = np.append(starts[1:], values.size)
    return {
        (int(sorted_pairs[start, 0]), int(sorted_pairs[start, 1])): set(sorted_values[start:end].tolist())
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True)
    }


def _build_positives(edge_index: torch.Tensor, relations: torch.Tensor) -> dict[tuple[int, int], set[int]]:
    """Build the bidirectional (node, relation) positive index in bulk."""
    heads = edge_index[0].detach().cpu().numpy()
    tails = edge_index[1].detach().cpu().numpy()
    relation_values = relations.detach().cpu().numpy()
    pairs = np.concatenate((np.column_stack((heads, relation_values)), np.column_stack((tails, relation_values))))
    return _group_pairs(pairs, np.concatenate((tails, heads)))


def _full_messages(
    edge_index: torch.Tensor, relations: torch.Tensor, splits: torch.Tensor, types: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    train = torch.nonzero(splits == 0).flatten()
    if not train.numel():
        raise ValueError("evaluation requires at least one training statement for topology")
    max_relation = int(relations.max())
    heads = edge_index[0, train]
    tails = edge_index[1, train]
    train_relations = relations[train]
    type_head = types[heads]
    type_tail = types[tails]
    # Interleave forward then reverse triples per train position so one labeling
    # pass reproduces the historical order: the forward key is inserted before
    # the reverse key for the same statement.
    forward_triples = torch.stack((type_head, train_relations, type_tail), dim=1)
    reverse_triples = torch.stack((type_tail, train_relations + max_relation + 1, type_head), dim=1)
    sequence = torch.stack((forward_triples, reverse_triples), dim=1).reshape(-1, 3)
    labels, key_count = _first_occurrence_ids(sequence)
    forward = labels[0::2]
    reverse = labels[1::2]
    return (
        torch.cat((edge_index[:, train], edge_index[[1, 0]][:, train]), dim=1),
        torch.cat((train_relations, train_relations + max_relation + 1)),
        torch.cat((forward, reverse)),
        key_count,
    )


def filtered_rank(scores: torch.Tensor, target: int, filtered: set[int]) -> int:
    mask = torch.ones(scores.shape, dtype=torch.bool, device=scores.device)
    excluded = [index for index in filtered if index != target]
    if excluded:
        mask[torch.as_tensor(excluded, dtype=torch.long, device=scores.device)] = False
    mask[target] = True
    return int((scores[mask] > scores[target]).sum().item()) + 1


def _sampled_selection(
    target: int,
    filtered: set[int],
    candidates: np.ndarray | None,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    """Select ranked candidates without changing the historical RNG stream."""
    pool = np.array([], dtype=np.int64) if candidates is None else candidates
    if pool.size > count:
        pool = rng.choice(pool, size=count, replace=False)
    chosen = np.union1d(pool, np.array([target], dtype=np.int64))
    if filtered:
        excluded = np.array(sorted(index for index in filtered if index != target), dtype=np.int64)
        chosen = chosen[~np.isin(chosen, excluded)]
    if target not in chosen:
        chosen = np.append(chosen, target)
    return chosen


def _selection_rank(scores: torch.Tensor, target: int, chosen: np.ndarray) -> int:
    selected = torch.as_tensor(chosen, dtype=torch.long, device=scores.device)
    return int((scores[selected] > scores[target]).sum().item()) + 1


def sampled_rank(
    scores: torch.Tensor,
    target: int,
    filtered: set[int],
    candidates: np.ndarray | None,
    rng: np.random.Generator,
    count: int,
) -> int:
    chosen = _sampled_selection(target, filtered, candidates, rng, count)
    return _selection_rank(scores, target, chosen)


def _score_relation_batch(
    model: RelationalAutoencoder,
    state: torch.Tensor,
    heads: list[int],
    tails: list[int],
    relation: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score same-relation targets together with bounded batched decoder calls."""
    head_ids = torch.tensor(heads, dtype=torch.long, device=state.device)
    tail_ids = torch.tensor(tails, dtype=torch.long, device=state.device)
    relation_ids = torch.full((len(heads),), relation, dtype=torch.long, device=state.device)
    node_real, node_imaginary = model.decoder.complex_states(state)
    head_real, head_imaginary = model.decoder.complex_states(state[head_ids])
    tail_real, tail_imaginary = model.decoder.complex_states(state[tail_ids])
    relation_real, relation_imaginary = model.decoder.relations(relation_ids).chunk(2, dim=-1)
    # Project each distinct target and the graph once, instead of projecting an
    # expanded (targets x entities) matrix. Broadcasting only repeats the same
    # elementwise ComplEx products and reduction order as decoder.score().
    tail_scores = (
        head_real.unsqueeze(1) * relation_real.unsqueeze(1)
        - head_imaginary.unsqueeze(1) * relation_imaginary.unsqueeze(1)
    ).mul(node_real).sum(-1) + (
        head_real.unsqueeze(1) * relation_imaginary.unsqueeze(1)
        + head_imaginary.unsqueeze(1) * relation_real.unsqueeze(1)
    ).mul(node_imaginary).sum(-1)
    head_scores = (node_real * relation_real.unsqueeze(1) - node_imaginary * relation_imaginary.unsqueeze(1)).mul(
        tail_real.unsqueeze(1)
    ).sum(-1) + (node_real * relation_imaginary.unsqueeze(1) + node_imaginary * relation_real.unsqueeze(1)).mul(
        tail_imaginary.unsqueeze(1)
    ).sum(-1)
    return tail_scores, head_scores


def evaluate_dataset(
    root: Path,
    *,
    split: str = "test",
    checkpoint: Path | None = None,
    seed: int = 20261110,
    family_prefix: str = "CED:Family_",
    hits: tuple[int, ...] = (1, 3, 10),
    device: Device = "auto",
    ranking: Ranking = "exact",
    sampled_candidates: int = 1000,
) -> dict[str, object]:
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    if ranking not in {"exact", "sampled"}:
        raise ValueError("ranking must be exact or sampled")
    if sampled_candidates < 1:
        raise ValueError("sampled candidate count must be positive")
    resolved = resolve_device(device)
    document = json.loads((root / "manifest.json").read_text())
    references = {name: TensorRef.model_validate(item) for name, item in document["tensors"].items()}
    arrays = {name: reference.load(root) for name, reference in references.items()}
    pools: TypedPools | None = (
        typed_pools(
            np.stack((arrays["head"], arrays["tail"])), arrays["relation_id"], arrays["split"], arrays["type_set_id"]
        )
        if ranking == "sampled"
        else None
    )

    def tensor(name: str) -> torch.Tensor:
        return torch.from_numpy(arrays[name]).to(resolved)

    edge_index = torch.from_numpy(np.stack((arrays["head"], arrays["tail"])))
    relations, splits, types = (
        tensor("relation_id"),
        tensor("split"),
        tensor("type_set_id"),
    )
    messages, message_relations, message_types, edge_type_count = _full_messages(edge_index, relations, splits, types)
    messages, message_relations, message_types = (
        messages.to(resolved),
        message_relations.to(resolved),
        message_types.to(resolved),
    )
    model = build_model(document, root, edge_type_count, seed).to(resolved)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=resolved, weights_only=True)["state"])
    model.eval()
    entity_ids = torch.arange(int(document["entities"]), device=resolved)
    with torch.no_grad():
        state = model.encoder(
            model.features.node_features(entity_ids),
            messages,
            message_types,
            model.features.statement_features(
                message_relations, torch.arange(message_relations.numel(), device=resolved)
            ),
        )
    positives = _build_positives(edge_index, relations)
    split_id = 1 if split == "validation" else 2
    targets = torch.nonzero(splits == split_id).flatten()
    if not targets.numel():
        raise ValueError(f"no target statements in {split} split")
    entities = pq.read_table(root / "entities.parquet", columns=["id"])["id"].to_pylist()
    predicates = pq.read_table(root / "relations.parquet", columns=["id"])["id"].to_pylist()
    ranks: list[int] = []
    groups: dict[str, list[int]] = {}
    random = np.random.default_rng(seed)
    # Bulk host fetches preserve target order; sampled candidates are selected
    # in that same order before relation grouping, so the RNG stream is unchanged.
    head_list = edge_index[0, targets].tolist()
    relation_list = relations[targets].tolist()
    tail_list = edge_index[1, targets].tolist()
    target_rows = list(zip(head_list, relation_list, tail_list, strict=True))
    selections: list[tuple[np.ndarray, np.ndarray]] | None = None
    if ranking == "sampled":
        assert pools is not None
        selections = [
            (
                _sampled_selection(
                    tail,
                    positives.get((head, relation), set()),
                    pools.objects.get(relation),
                    random,
                    sampled_candidates,
                ),
                _sampled_selection(
                    head,
                    positives.get((tail, relation), set()),
                    pools.subjects.get(relation),
                    random,
                    sampled_candidates,
                ),
            )
            for head, relation, tail in target_rows
        ]
    relation_positions: dict[int, list[int]] = {}
    for position, relation in enumerate(relation_list):
        relation_positions.setdefault(relation, []).append(position)
    rank_pairs: list[tuple[int, int] | None] = [None] * len(target_rows)
    with torch.no_grad():
        for relation, positions in relation_positions.items():
            for start in range(0, len(positions), _TARGET_BATCH_SIZE):
                batch_positions = positions[start : start + _TARGET_BATCH_SIZE]
                tail_scores, head_scores = _score_relation_batch(
                    model,
                    state,
                    [head_list[position] for position in batch_positions],
                    [tail_list[position] for position in batch_positions],
                    relation,
                )
                for row, position in enumerate(batch_positions):
                    head, _, tail = target_rows[position]
                    if selections is None:
                        tail_rank = filtered_rank(tail_scores[row], tail, positives.get((head, relation), set()))
                        head_rank = filtered_rank(head_scores[row], head, positives.get((tail, relation), set()))
                    else:
                        tail_selection, head_selection = selections[position]
                        tail_rank = _selection_rank(tail_scores[row], tail, tail_selection)
                        head_rank = _selection_rank(head_scores[row], head, head_selection)
                    rank_pairs[position] = (tail_rank, head_rank)
    # Restore statement order before aggregating so floating-point reduction
    # order, group insertion order, and the final report match the reference.
    for (head, relation, tail), rank_pair in zip(target_rows, rank_pairs, strict=True):
        assert rank_pair is not None
        for rank, endpoint in zip(rank_pair, (tail, head), strict=True):
            ranks.append(rank)
            groups.setdefault(f"predicate:{predicates[relation]}", []).append(rank)
            identifier = entities[endpoint]
            if identifier.startswith(family_prefix):
                groups.setdefault("ced_family", []).append(rank)
    tensor_ranks = torch.tensor(ranks, dtype=torch.float64)
    report: dict[str, object] = {
        "split": split,
        "ranking": ranking,
        "device": str(resolved),
        "support": len(ranks),
        "mrr": float((1 / tensor_ranks).mean()),
        "hits": {f"hits@{k}": float((tensor_ranks <= k).float().mean()) for k in hits},
    }
    if ranking == "sampled":
        report["sampled_candidates"] = sampled_candidates
    for name, values in sorted(groups.items()):
        values_tensor = torch.tensor(values, dtype=torch.float64)
        report[name] = {
            "support": len(values),
            "mrr": float((1 / values_tensor).mean()),
            "hits": {f"hits@{k}": float((values_tensor <= k).float().mean()) for k in hits},
        }
    return report
