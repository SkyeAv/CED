"""Decoder supervision must not depend on OOM-thinned message positions.

Why this test exists: bidirectional messages store all forward statements before
all reverse statements. After thinning a context-only statement, a positional
search over the remaining statement IDs can select a reverse edge with the wrong
head, tail, and relation. Node-ID features make that corruption directly visible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ced.model.decoder import ComplexDecoder
from ced.model.dense_features import RandomProjectionProvider
from ced.model.encoder import RelationalAttentionEncoder
from ced.model.losses import AuxiliaryAttributeDecoder
from ced.model.prepare import save_tensor
from ced.model.schemas import GraphBatch, NegativeBatch
from ced.model.training import RelationalAutoencoder, train_step


class _NodeIdEncoder(RelationalAttentionEncoder):
    def __init__(self) -> None:
        super().__init__(2, 1, edge_types=1, layers=1)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_id: torch.Tensor,
        statement_features: torch.Tensor,
    ) -> torch.Tensor:
        del edge_index, edge_type_id, statement_features
        return nodes


class _NodeIdFeatures(RandomProjectionProvider):
    def __init__(self) -> None:
        super().__init__(2, 1, relations=1, seed=3)

    def node_features(self, global_node_id: torch.Tensor) -> torch.Tensor:
        return global_node_id[:, None].to(torch.float32).expand(-1, 2)

    def statement_features(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor:
        del statement_id
        return torch.zeros((relation_id.numel(), 1), dtype=torch.float32, device=relation_id.device)


class _RecordingDecoder(ComplexDecoder):
    def __init__(self) -> None:
        super().__init__(2, relations=1)
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def score(self, subjects: torch.Tensor, relations: torch.Tensor, objects: torch.Tensor) -> torch.Tensor:
        self.calls.append((subjects.detach().clone(), relations.detach().clone(), objects.detach().clone()))
        return super().score(subjects, relations, objects)


def _write_graph(root: Path) -> GraphBatch:
    directory = root / "oom"
    directory.mkdir(parents=True)

    def save(name: str, values: np.ndarray):
        reference = save_tensor(directory, name, values)
        reference.path = f"oom/{reference.path}"
        return reference

    return GraphBatch(
        global_node_id=save("global_node_id", np.arange(6, dtype=np.int64)),
        seed_id=save("seed_id", np.arange(6, dtype=np.int64)),
        type_set_id=save("type_set_id", np.zeros(6, dtype=np.int64)),
        edge_index=save(
            "edge_index",
            np.array([[0, 2, 4, 1, 3, 5], [1, 3, 5, 0, 2, 4]], dtype=np.int64),
        ),
        relation_id=save("relation_id", np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)),
        edge_type_id=save("edge_type_id", np.zeros(6, dtype=np.int64)),
        statement_id=save("statement_id", np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)),
        positive_statement_id=save("positive_statement_id", np.array([0, 2], dtype=np.int64)),
        negative=NegativeBatch(
            subject=save("negative_subject", np.array([[2, 4], [0, 2]], dtype=np.int64)),
            object=save("negative_object", np.array([[1, 1], [5, 5]], dtype=np.int64)),
            valid=save("negative_valid", np.ones((2, 2), dtype=np.bool_)),
        ),
    )


def _decoder_calls(root: Path, graph: GraphBatch, reduction: int) -> list[tuple[torch.Tensor, ...]]:
    decoder = _RecordingDecoder()
    model = RelationalAutoencoder(
        encoder=_NodeIdEncoder(),
        decoder=decoder,
        attributes=AuxiliaryAttributeDecoder(2, {}, []),
        features=_NodeIdFeatures(),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_step(
        model,
        optimizer,
        root,
        {},
        graph,
        None,
        device=torch.device("cpu"),
        auxiliary_weight=0.0,
        memory_reduction=reduction,
    )
    assert np.isfinite(loss)
    return decoder.calls


def test_oom_thinning_preserves_all_decoder_triples(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path)

    baseline = _decoder_calls(tmp_path, graph, reduction=1)
    reduced = _decoder_calls(tmp_path, graph, reduction=2)

    assert len(baseline) == len(reduced) == 2
    for original_calls, reduced_calls in zip(baseline, reduced, strict=True):
        for original, thinned in zip(original_calls, reduced_calls, strict=True):
            assert torch.equal(original, thinned)
    positive_subjects, positive_relations, positive_objects = baseline[0]
    assert positive_subjects[:, 0].tolist() == [0, 4]
    assert positive_relations.tolist() == [0, 0]
    assert positive_objects[:, 0].tolist() == [1, 5]
    negative_subjects, negative_relations, negative_objects = baseline[1]
    assert negative_subjects[:, 0].tolist() == [2, 4, 0, 2]
    assert negative_relations.tolist() == [0, 0, 0, 0]
    assert negative_objects[:, 0].tolist() == [1, 1, 5, 5]
