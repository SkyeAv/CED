from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from ced.acquisition.base import atomic_write_json
from ced.config.models import RuntimeConfig
from ced.modal_plan import dataset_transfer_plan
from ced.model.evaluation import evaluate_dataset
from ced.model.prepare import save_tensor
from ced.model.schemas import (
    DatasetManifest,
    QualifierStatementCsr,
    SemanticFeatureStore,
    TrainingExample,
)
from ced.model.semantic import SemanticFeatureProvider
from ced.model.training import build_model, train_batches, train_step


def add_semantic_features(root: Path) -> SemanticFeatureStore:
    predicates = save_tensor(root, "predicate-embeddings", np.array([[1, 0, 0, 0]], dtype=np.float32))
    qualifiers = save_tensor(root, "qualifier-embeddings", np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32))
    links = save_tensor(root, "statement-qualifier-statements", np.array([0, 0, 2], dtype=np.int64))
    qualifier_id = save_tensor(root, "statement-qualifier-ids", np.array([0, 1, 1], dtype=np.int64))
    indptr = save_tensor(root, "statement-qualifier-indptr", np.array([0, 2, 2, 3], dtype=np.int64))
    store = SemanticFeatureStore(
        provider="random",
        model="fixture",
        encoder_key="fixture",
        dimension=4,
        relations=1,
        qualifier_count=2,
        predicate_text_path="predicate-text.parquet",
        qualifier_text_path="qualifier-text.parquet",
        predicates=predicates,
        qualifiers=qualifiers,
        statement_qualifiers=QualifierStatementCsr(
            statement_id=links,
            qualifier_id=qualifier_id,
            indptr=indptr,
        ),
    )
    document = json.loads((root / "manifest.json").read_text())
    document["semantic"] = store.model_dump(mode="json")
    for reference in (predicates, qualifiers, links, qualifier_id, indptr):
        document["checksums"][reference.path] = reference.sha256
    validated = DatasetManifest.model_validate(document)
    atomic_write_json(root / "manifest.json", validated.model_dump(mode="json"))
    return store


def test_semantic_provider_autoencodes_predicates_and_rare_qualifiers(
    tmp_path: Path, synthetic_dataset: Callable[..., int]
) -> None:
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    add_semantic_features(root)
    document = json.loads((root / "manifest.json").read_text())
    model = build_model(document, root, edge_type_count=2, seed=5)
    provider = model.features
    assert isinstance(provider, SemanticFeatureProvider)

    features = provider.statement_features(
        torch.tensor([0, 1, 0], dtype=torch.long), torch.tensor([0, 0, 1], dtype=torch.long)
    )
    assert features.shape == (3, 16)
    assert torch.equal(features[0], features[1])
    expected = provider.predicate_head.reconstruction_loss(provider.predicate_vectors[:1])
    expected = expected + provider.qualifier_head.reconstruction_loss(provider.qualifier_vectors[1:2])
    actual = provider.reconstruction_loss(torch.tensor([0]), torch.tensor([2]))
    assert torch.allclose(actual, expected)
    empty = provider.reconstruction_loss(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    assert empty.item() == 0
    actual.backward()
    assert provider.predicate_head.encoder.weight.grad is not None
    assert provider.qualifier_head.encoder.weight.grad is not None
    assert not any(name.endswith("predicate_vectors") for name in model.state_dict())
    assert not any(name.endswith("qualifier_indptr") for name in model.state_dict())


def test_semantic_training_step_has_finite_gradients(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    add_semantic_features(root)
    document = json.loads((root / "manifest.json").read_text())
    model = build_model(document, root, edge_type_count=2, seed=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    example = TrainingExample.model_validate(document["batches"][0])
    loss = train_step(
        model,
        optimizer,
        root,
        document,
        example.graph,
        None,
        device=torch.device("cpu"),
        auxiliary_weight=0.1,
    )
    assert np.isfinite(loss)
    provider = model.features
    assert isinstance(provider, SemanticFeatureProvider)
    for head in (provider.predicate_head, provider.qualifier_head):
        gradient = head.encoder.weight.grad
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert bool((gradient != 0).any())


def test_semantic_evaluation_and_transfer_use_statement_qualifiers(
    tmp_path: Path, synthetic_dataset: Callable[..., int]
) -> None:
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    add_semantic_features(root)
    report = evaluate_dataset(root, split="test", seed=3, device="cpu", ranking="exact")
    assert report["support"] == 2
    runtime = RuntimeConfig(remote_project=tmp_path, artifact_root=tmp_path / ".ced", disk_reserve_gb=0)
    plan = dataset_transfer_plan(root, runtime, "train")
    assert {
        "predicate-embeddings.safetensors",
        "qualifier-embeddings.safetensors",
        "statement-qualifier-statements.safetensors",
        "statement-qualifier-ids.safetensors",
        "statement-qualifier-indptr.safetensors",
    } <= set(plan.shared)


def test_semantic_checkpoint_resume(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    add_semantic_features(root)
    checkpoint = tmp_path / "run" / "model.pt"
    first = train_batches(root, epochs=1, seed=5, checkpoint_path=checkpoint, device="cpu")
    second = train_batches(root, epochs=2, seed=5, checkpoint_path=checkpoint, device="cpu")
    assert first["batches"] == 1
    assert second["batches"] == 2
    restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert restored["epoch"] == 2
    assert any(name.startswith("features.predicate_head") for name in restored["state"])
