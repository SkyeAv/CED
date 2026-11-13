"""Independent evaluation reference for the persisted edge-type contract.

The loop-based message construction and ungrouped scoring remain independent of
the optimized implementation. Both sides consume the same persisted edge-type
IDs; ranking and aggregation must remain numerically identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import torch  # pyright: ignore[reportMissingImports]

from ced.model.negatives import TypedPools, typed_pools
from ced.model.schemas import TensorRef
from ced.model.training import Device, build_model, persisted_edge_type_count, resolve_device

Ranking = Literal["exact", "sampled"]


def _full_messages(
    edge_index: torch.Tensor, relations: torch.Tensor, splits: torch.Tensor, edge_types: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    train = torch.nonzero(splits == 0).flatten()
    if not train.numel():
        raise ValueError("evaluation requires at least one training statement for topology")
    forward = edge_types[train].clone()
    reverse = edge_types[train].clone()
    return (
        torch.cat((edge_index[:, train], edge_index[[1, 0]][:, train]), dim=1),
        torch.cat((relations[train], relations[train] + int(relations.max()) + 1)),
        torch.cat((forward, reverse)),
        1 + int(edge_types[train].max()),
    )


def filtered_rank(scores: torch.Tensor, target: int, filtered: set[int]) -> int:
    mask = torch.ones(scores.shape, dtype=torch.bool, device=scores.device)
    excluded = [index for index in filtered if index != target]
    if excluded:
        mask[torch.as_tensor(excluded, dtype=torch.long, device=scores.device)] = False
    mask[target] = True
    return int((scores[mask] > scores[target]).sum().item()) + 1


def sampled_rank(
    scores: torch.Tensor,
    target: int,
    filtered: set[int],
    candidates: np.ndarray | None,
    rng: np.random.Generator,
    count: int,
) -> int:
    pool = np.array([], dtype=np.int64) if candidates is None else candidates
    if pool.size > count:
        pool = rng.choice(pool, size=count, replace=False)
    chosen = np.union1d(pool, np.array([target], dtype=np.int64))
    if filtered:
        excluded = np.array(sorted(index for index in filtered if index != target), dtype=np.int64)
        chosen = chosen[~np.isin(chosen, excluded)]
    if target not in chosen:
        chosen = np.append(chosen, target)
    selected = torch.as_tensor(chosen, dtype=torch.long, device=scores.device)
    return int((scores[selected] > scores[target]).sum().item()) + 1


def evaluate_dataset_reference(
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

    edge_type_count = persisted_edge_type_count(document, root, arrays=arrays)
    edge_index = torch.from_numpy(np.stack((arrays["head"], arrays["tail"]))).to(resolved)
    relations, splits, edge_types = tensor("relation_id"), tensor("split"), tensor("edge_type_id")
    messages, message_relations, message_types, message_count = _full_messages(
        edge_index, relations, splits, edge_types
    )
    if message_count != edge_type_count:
        raise ValueError("edge type count differs between full evaluation messages and the manifest")
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
    positives: dict[tuple[int, int], set[int]] = {}
    for head, relation, tail in zip(edge_index[0].tolist(), relations.tolist(), edge_index[1].tolist(), strict=True):
        positives.setdefault((head, relation), set()).add(tail)
        positives.setdefault((tail, relation), set()).add(head)
    split_id = 1 if split == "validation" else 2
    targets = torch.nonzero(splits == split_id).flatten()
    if not targets.numel():
        raise ValueError(f"no target statements in {split} split")
    entities = pq.read_table(root / "entities.parquet", columns=["id"])["id"].to_pylist()
    predicates = pq.read_table(root / "relations.parquet", columns=["id"])["id"].to_pylist()
    ranks: list[int] = []
    groups: dict[str, list[int]] = {}
    random = np.random.default_rng(seed)
    with torch.no_grad():
        for target in targets.tolist():
            head, relation, tail = int(edge_index[0, target]), int(relations[target]), int(edge_index[1, target])
            relation_tensor = torch.full((state.shape[0],), relation, dtype=torch.long, device=resolved)
            tail_scores = model.decoder.score(state[head].expand_as(state), relation_tensor, state)
            head_scores = model.decoder.score(state, relation_tensor, state[tail].expand_as(state))
            if ranking == "sampled":
                assert pools is not None
                tail_rank = sampled_rank(
                    tail_scores,
                    tail,
                    positives.get((head, relation), set()),
                    pools.objects.get(relation),
                    random,
                    sampled_candidates,
                )
                head_rank = sampled_rank(
                    head_scores,
                    head,
                    positives.get((tail, relation), set()),
                    pools.subjects.get(relation),
                    random,
                    sampled_candidates,
                )
            else:
                tail_rank = filtered_rank(tail_scores, tail, positives.get((head, relation), set()))
                head_rank = filtered_rank(head_scores, head, positives.get((tail, relation), set()))
            for rank, predicate_id, endpoint in zip(
                (tail_rank, head_rank), (relation, relation), (tail, head), strict=True
            ):
                ranks.append(rank)
                groups.setdefault(f"predicate:{predicates[predicate_id]}", []).append(rank)
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
