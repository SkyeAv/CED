from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from torch import nn

from ced.acquisition.base import atomic_write_json, sha256_file
from ced.kg.records import canonical_json
from ced.model.decoder import ComplexDecoder
from ced.model.encoder import RelationalAttentionEncoder
from ced.model.feature_index import FeatureIndex, load_feature_metadata
from ced.model.losses import AuxiliaryAttributeDecoder, auxiliary_loss, link_loss
from ced.model.schemas import GraphBatch, SemanticFeatureStore, TensorRef, TrainingExample
from ced.model.semantic import SemanticFeatureProvider

Device = Literal["auto", "cpu", "cuda"]


class FeatureProvider(Protocol):
    def node_features(self, global_node_id: torch.Tensor) -> torch.Tensor: ...

    def statement_features(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor: ...

    def reconstruction_loss(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor | None: ...


class RandomProjectionProvider(nn.Module):
    """Deterministic temporary provider; semantic providers remain swappable."""

    node_vectors: torch.Tensor
    statement_vectors: torch.Tensor

    def __init__(self, dimension: int, statement_dimension: int, relations: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("node_vectors", torch.randn(65_536, dimension, generator=generator))
        self.register_buffer(
            "statement_vectors", torch.randn(max(1, 2 * relations), statement_dimension, generator=generator)
        )

    def node_features(self, global_node_id: torch.Tensor) -> torch.Tensor:
        return self.node_vectors[global_node_id % self.node_vectors.shape[0]]

    def statement_features(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor:
        del statement_id
        return self.statement_vectors[relation_id.clamp(0, self.statement_vectors.shape[0] - 1)]

    def reconstruction_loss(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> None:
        del relation_id, statement_id


class RelationalAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: RelationalAttentionEncoder,
        decoder: ComplexDecoder,
        attributes: AuxiliaryAttributeDecoder,
        features: FeatureProvider,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attributes = attributes
        self.features = features


def _load(reference: TensorRef | dict[str, Any], root: Path, device: torch.device) -> torch.Tensor:
    ref: TensorRef = reference if isinstance(reference, TensorRef) else TensorRef.model_validate(reference)
    return torch.from_numpy(ref.load(root).copy()).to(device)


def resolve_device(device: Device) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_model(document: dict[str, Any], root: Path, edge_type_count: int, seed: int) -> RelationalAutoencoder:
    relation_count = document["relations"]
    counts, numeric = load_feature_metadata(root)
    torch.manual_seed(seed)
    encoder = RelationalAttentionEncoder(32, 16, edge_type_count)
    decoder = ComplexDecoder(32, relation_count)
    attributes = AuxiliaryAttributeDecoder(32, counts, numeric)
    features: FeatureProvider
    if document.get("semantic"):
        store = SemanticFeatureStore.model_validate(document["semantic"])
        features = SemanticFeatureProvider(32, 16, relation_count, seed, store, root)
    else:
        features = RandomProjectionProvider(32, 16, relation_count, seed)
    return RelationalAutoencoder(encoder=encoder, decoder=decoder, attributes=attributes, features=features)


def config_hash(
    document: dict[str, Any],
    *,
    learning_rate: float,
    seed: int,
    auxiliary_weight: float,
    semantic_weight: float = 0.1,
) -> str:
    payload = {
        "dataset_id": document.get("dataset_id"),
        "snapshot_id": document.get("snapshot_id"),
        "learning_rate": learning_rate,
        "seed": seed,
        "auxiliary_weight": auxiliary_weight,
        "model_version": 2,
    }
    if document.get("semantic"):
        payload.update(
            model_version=3,
            semantic_weight=semantic_weight,
            semantic=document["semantic"],
        )
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _subsample(
    edge_index: torch.Tensor,
    relation_id: torch.Tensor,
    edge_type_id: torch.Tensor,
    statement_id: torch.Tensor,
    reduction: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if reduction <= 1:
        return edge_index, relation_id, edge_type_id, statement_id
    keep = torch.arange(edge_index.shape[1], device=edge_index.device) % reduction == 0
    return edge_index[:, keep], relation_id[keep], edge_type_id[keep], statement_id[keep]


def _owner_states(
    state: torch.Tensor, mapping_ids: torch.Tensor, owner_rows: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    positions = torch.searchsorted(mapping_ids, owner_rows.to(mapping_ids.device))
    del rows
    return state[positions]


def _auxiliary_tensors(
    features: FeatureIndex | None, state: torch.Tensor, global_nodes: torch.Tensor, statements: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    empty_long = torch.empty(0, dtype=torch.long, device=state.device)
    empty_float = torch.empty(0, dtype=torch.float32, device=state.device)
    empty_bool = torch.empty(0, dtype=torch.bool, device=state.device)
    if features is None or statements.numel() == 0:
        return state[:0], empty_long, empty_long, empty_long, empty_float, empty_bool
    statement_owners = features.statement_owner(statements)
    node_rows, node_fields, node_values = features.categorical_rows(global_nodes)
    statement_rows, statement_fields, statement_values = features.categorical_rows(statement_owners)
    categorical_fields = torch.cat((node_fields, statement_fields)).to(state.device)
    categorical_targets = torch.cat((node_values, statement_values)).to(state.device)
    categorical_states = torch.cat(
        (
            _owner_states(state, global_nodes, features.cat_owner[node_rows], node_rows),
            _owner_states(state, statements, features.cat_owner[statement_rows] - features.node_count, statement_rows),
        )
    )
    numeric_rows, numeric_fields, numeric_values, numeric_valid = features.numeric_rows(global_nodes)
    statement_numeric_rows, statement_numeric_fields, statement_numeric_values, statement_numeric_valid = (
        features.numeric_rows(statement_owners)
    )
    numeric_fields = torch.cat((numeric_fields, statement_numeric_fields)).to(state.device)
    numeric_values = torch.cat((numeric_values, statement_numeric_values)).to(state.device)
    numeric_valid = torch.cat((numeric_valid, statement_numeric_valid)).to(state.device)
    numeric_states = torch.cat(
        (
            _owner_states(state, global_nodes, features.num_owner[numeric_rows], numeric_rows),
            _owner_states(
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


def train_step(
    model: RelationalAutoencoder,
    optimizer: torch.optim.Optimizer,
    root: Path,
    document: dict[str, Any],
    graph: GraphBatch,
    features: FeatureIndex | None,
    *,
    device: torch.device,
    auxiliary_weight: float,
    semantic_weight: float = 0.1,
    memory_reduction: int = 1,
    use_bf16: bool = False,
) -> float:
    edge_index = _load(graph.edge_index, root, device)
    relation_id = _load(graph.relation_id, root, device)
    edge_type_id = _load(graph.edge_type_id, root, device)
    statement_id = _load(graph.statement_id, root, device)
    positive = _load(graph.positive_statement_id, root, device)
    global_nodes = _load(graph.global_node_id, root, device)
    edge_index, relation_id, edge_type_id, statement_id = _subsample(
        edge_index, relation_id, edge_type_id, statement_id, memory_reduction
    )
    unique_statements = torch.unique(statement_id)
    selected = torch.isin(positive, unique_statements)
    if not bool(selected.any()):
        return 0.0
    positive = positive[selected]
    positions = torch.searchsorted(unique_statements, positive)
    heads = edge_index[0, positions]
    tails = edge_index[1, positions]
    relations = relation_id[positions]
    negative_subject = _load(graph.negative.subject, root, device)[selected]
    negative_object = _load(graph.negative.object, root, device)[selected]
    negative_valid = _load(graph.negative.valid, root, device)[selected]
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else torch.autocast(device.type, enabled=False)
    )
    with context:
        state = model.encoder(
            model.features.node_features(global_nodes),
            edge_index,
            edge_type_id,
            model.features.statement_features(relation_id, statement_id),
        )
        subjects = torch.searchsorted(global_nodes, negative_subject[negative_valid])
        objects = torch.searchsorted(global_nodes, negative_object[negative_valid])
        negative_relations = (
            relations[:, None].expand(-1, negative_subject.shape[1]).reshape(-1)[negative_valid.reshape(-1)]
        )
        positive_scores = model.decoder.score(state[heads], relations, state[tails])
        negative_scores = model.decoder.score(state[subjects], negative_relations, state[objects])
        scores = torch.cat((positive_scores, negative_scores))
        targets = torch.cat((torch.ones_like(positive_scores), torch.zeros_like(negative_scores)))
        loss = link_loss(scores, targets)
        reconstruction = model.features.reconstruction_loss(relations, positive)
        if reconstruction is not None:
            loss = loss + semantic_weight * reconstruction
        if features is not None:
            states, categorical_fields, categorical_targets, numeric_fields, numeric_values, numeric_valid = (
                _auxiliary_tensors(features, state, global_nodes, positive)
            )
            loss = loss + auxiliary_loss(
                model.attributes,
                states,
                categorical_fields,
                categorical_targets,
                numeric_fields,
                numeric_values,
                numeric_valid,
                weight=auxiliary_weight,
            )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    optimizer.step()
    return float(loss.detach())


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    (path.parent / "checkpoint.sha256").write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    atomic_write_json(
        path.parent / "checkpoint.json",
        {key: payload[key] for key in ("epoch", "cursor", "config_hash", "metrics", "device") if key in payload},
    )


def train_batches(
    root: Path,
    *,
    epochs: int = 1,
    learning_rate: float = 1e-3,
    seed: int = 20261110,
    checkpoint_path: Path | None = None,
    device: Device = "auto",
    selected_indices: Sequence[int] | None = None,
    start_cursor: int = 0,
    max_batches: int | None = None,
    auxiliary_weight: float = 0.1,
    semantic_weight: float = 0.1,
    edge_type_count: int | None = None,
    max_memory_reductions: int = 4,
) -> dict[str, Any]:
    if semantic_weight < 0:
        raise ValueError("semantic weight must be nonnegative")
    document = json.loads((root / "manifest.json").read_text())
    examples = [TrainingExample.model_validate(item) for item in document["batches"]]
    resolved_device = resolve_device(device)
    if edge_type_count is None:
        edge_type_count = 1 + max(
            int(_load(example.graph.edge_type_id, root, torch.device("cpu")).max()) for example in examples
        )
    model = build_model(document, root, edge_type_count, seed).to(resolved_device)
    # fused AdamW is a CUDA-only kernel choice; the CPU path stays bitwise unchanged.
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=True)
        if resolved_device.type == "cuda"
        else torch.optim.AdamW(model.parameters(), lr=learning_rate)
    )
    identity = config_hash(
        document,
        learning_rate=learning_rate,
        seed=seed,
        auxiliary_weight=auxiliary_weight,
        semantic_weight=semantic_weight,
    )
    metrics: dict[str, float] = {"loss": 0.0, "batches": 0.0}
    total = 0.0
    trained = 0
    start_epoch = 0
    start_cursor = 0
    if checkpoint_path is not None and checkpoint_path.is_file():
        restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if restored.get("config_hash") != identity:
            raise ValueError("checkpoint configuration does not match requested run")
        model.load_state_dict(restored["state"])
        optimizer.load_state_dict(restored["optimizer"])
        metrics = dict(restored["metrics"])
        total = float(metrics.get("loss", 0.0)) * float(metrics.get("batches", 0.0))
        trained = int(metrics.get("batches", 0.0))
        start_epoch = int(restored["epoch"])
        start_cursor = int(restored["cursor"])
        random.setstate(restored["rng"]["python"])
        np.random.set_state(restored["rng"]["numpy"])
        torch.set_rng_state(restored["rng"]["torch"])
    features = (
        FeatureIndex(root, document).to(resolved_device) if document.get("features", {}).get("categorical") else None
    )
    use_bf16 = resolved_device.type == "cuda" and torch.cuda.get_device_capability(resolved_device)[0] >= 8
    all_indices = list(range(len(examples)))
    consumed = 0
    current_epoch = start_epoch
    current_cursor = start_cursor
    stop = False

    def run_indices(indices: Sequence[int]) -> None:
        nonlocal consumed, current_cursor, stop, total, trained
        for index in indices:
            if max_batches is not None and consumed >= max_batches:
                stop = True
                return
            consumed += 1
            current_cursor += 1
            example = examples[index]
            if example.split != "train":
                continue
            reduction = 1
            while True:
                try:
                    loss = train_step(
                        model,
                        optimizer,
                        root,
                        document,
                        example.graph,
                        features,
                        device=resolved_device,
                        auxiliary_weight=auxiliary_weight,
                        semantic_weight=semantic_weight,
                        memory_reduction=reduction,
                        use_bf16=use_bf16,
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    if reduction >= 2**max_memory_reductions or resolved_device.type != "cuda":
                        raise
                    torch.cuda.empty_cache()
                    reduction *= 2
            total += loss
            trained += 1

    if selected_indices is not None:
        run_indices(list(selected_indices)[start_cursor:])
    else:
        for epoch in range(start_epoch, epochs):
            current_epoch = epoch
            run_indices(all_indices[start_cursor:] if epoch == start_epoch else all_indices)
            if stop:
                break
            current_cursor = 0
            current_epoch = epoch + 1
    metrics = {
        "loss": total / trained if trained else 0.0,
        "batches": float(trained),
        "epoch": float(current_epoch),
        "cursor": float(current_cursor),
        "consumed": float(consumed),
    }
    if checkpoint_path is not None:
        _save_checkpoint(
            checkpoint_path,
            {
                "state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": current_epoch,
                "cursor": current_cursor,
                "metrics": metrics,
                "config_hash": identity,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
                "device": str(resolved_device),
            },
        )
    if not consumed and not trained:
        raise ValueError("dataset contains no training batches")
    return metrics


def checkpoint(model: nn.Module, path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"state": model.state_dict(), "metrics": metrics}, temporary)
    temporary.replace(path)
    (path.parent / "checkpoint.sha256").write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
