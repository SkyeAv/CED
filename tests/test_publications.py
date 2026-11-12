"""Hermetic contracts for precomputed publication features and their model wiring."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_prepare_topology import STATEMENTS, prepare

from ced.acquisition.base import sha256_file
from ced.config.models import RuntimeConfig
from ced.kg.audit import validate_checksums
from ced.model.publications import encode_dataset_publications, normalize_publication_id
from ced.model.schemas import DatasetManifest, PublicationFeatureStore
from ced.model.training import PublicationFeatureProvider, RandomProjectionProvider, build_model


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PMID:123", "PMID123"),
        (" pubmed 123 ", "PMID123"),
        ("https://pubmed.ncbi.nlm.nih.gov/123/", "PMID123"),
        ("PMCID:PMC2564495", "PMC2564495"),
        ("pmc:2564495", "PMC2564495"),
        ("https://pmc.ncbi.nlm.nih.gov/articles/PMC2564495/", "PMC2564495"),
        ("nct:12345678", "NCT12345678"),
        ("https://clinicaltrials.gov/study/NCT12345678", "NCT12345678"),
        ("DailyMed:abc123", None),
        ("DOI:10.1000/example", None),
        ("12345", None),
        ("PMID:not-a-number", None),
        ("", None),
    ],
)
def test_normalize_publication_id(raw: str, expected: str | None) -> None:
    assert normalize_publication_id(raw) == expected


def _fixture_embeddings(
    runtime: RuntimeConfig, path: Path, ids: list[str], vectors: list[list[float]], dimension: int
) -> None:
    pq.write_table(
        pa.table(
            {
                "id": pa.array(ids, pa.string()),
                "embedding": pa.array(vectors, pa.list_(pa.float32(), dimension)),
            }
        ),
        path,
        compression="zstd",
    )
    runtime.publication_embeddings = runtime.publication_embeddings.model_copy(
        update={
            "filename": path.name,
            "dimension": dimension,
            "source_sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    )


def _mock_embedding_download(monkeypatch: pytest.MonkeyPatch, source: Path) -> None:
    monkeypatch.setattr("ced.model.publications.require_cpu_worker", lambda runtime: None)
    monkeypatch.setattr("ced.model.publications.hf_hub_download", lambda **kwargs: str(source))


def test_publication_encoding_pools_embeddings_and_preserves_unmatched_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Available PubMed/PMC/NCT vectors are mean-pooled per statement.

    Missing and unsupported citations contribute no vector, but their distinct
    citations still count toward evidence strength. Repeated aliases of one PMID
    must not inflate either the mean or the scalar.
    """
    monkeypatch.setitem(
        STATEMENTS,
        "publications",
        [
            ["PMID:1", "PMC2", "DailyMed:example", "PMID1"],
            ["PMID:404"],
            ["NCT12345678"],
            [],
            ["https://pmc.ncbi.nlm.nih.gov/articles/PMC2/"],
            [],
        ],
    )
    _snapshot, dataset, runtime = prepare(tmp_path, monkeypatch)
    source = tmp_path / "fixture-embeddings.parquet"
    _fixture_embeddings(
        runtime,
        source,
        ["PMID1", "PMC2", "NCT12345678", "PMID999"],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 9, 9]],
        3,
    )
    _mock_embedding_download(monkeypatch, source)

    store = encode_dataset_publications(runtime, dataset.name)
    assert isinstance(store, PublicationFeatureStore)
    assert store.dimension == 3
    assert store.statement_publications == 6
    assert store.matched_publications == 4
    assert store.statements_with_publications == 4
    assert store.statements_with_embeddings == 3
    vectors = store.vectors.load(dataset)
    evidence = store.evidence.load(dataset)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(
        vectors,
        np.array(
            [
                [0.5, 0.5, 0],
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(evidence, np.log1p([3, 1, 1, 0, 1, 0]).astype(np.float32))
    document = json.loads((dataset / "manifest.json").read_text())
    assert document["publications"]["feature_key"] == store.feature_key
    validate_checksums(dataset, document["checksums"])

    first = (sha256_file(dataset / store.vectors.path), sha256_file(dataset / store.evidence.path))
    again = encode_dataset_publications(runtime, dataset.name)
    second = (sha256_file(dataset / again.vectors.path), sha256_file(dataset / again.evidence.path))
    assert first == second
    assert not list(dataset.glob("*.partial*"))
    assert not list(dataset.glob("*.raw.tmp-*"))


def test_publication_encoding_rejects_dimension_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(STATEMENTS, "publications", [["PMID:1"], *([[]] * 5)])
    _snapshot, dataset, runtime = prepare(tmp_path, monkeypatch)
    source = tmp_path / "wrong-width.parquet"
    _fixture_embeddings(runtime, source, ["PMID1"], [[1, 2]], 2)
    runtime.publication_embeddings = runtime.publication_embeddings.model_copy(update={"dimension": 3})
    _mock_embedding_download(monkeypatch, source)
    with pytest.raises(ValueError, match=r"dimension 2.*configured 3"):
        encode_dataset_publications(runtime, dataset.name)
    assert json.loads((dataset / "manifest.json").read_text())["publications"] is None


def test_publication_encoding_rejects_nonfinite_vectors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(STATEMENTS, "publications", [["PMID:1"], *([[]] * 5)])
    _snapshot, dataset, runtime = prepare(tmp_path, monkeypatch)
    source = tmp_path / "nonfinite.parquet"
    _fixture_embeddings(runtime, source, ["PMID1"], [[1, math.nan, 0]], 3)
    _mock_embedding_download(monkeypatch, source)
    with pytest.raises(ValueError, match="non-finite"):
        encode_dataset_publications(runtime, dataset.name)
    assert not list(dataset.glob("*.partial*"))
    assert not list(dataset.glob("*.raw.tmp-*"))


def test_publication_download_requires_explicit_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ced.model.publications import resolve_publication_embeddings

    runtime = RuntimeConfig(remote_project=tmp_path, artifact_root=tmp_path / ".ced", disk_reserve_gb=0)
    monkeypatch.setattr("ced.model.publications.require_cpu_worker", lambda runtime: None)
    called = False

    def unexpected(**kwargs: object) -> str:
        nonlocal called
        called = True
        return "unused"

    monkeypatch.setattr("ced.model.publications.hf_hub_download", unexpected)
    with pytest.raises(RuntimeError, match=r"download requires.*--yes"):
        resolve_publication_embeddings(runtime, download=True, assume_yes=False)
    assert not called


def _attach_publication_features(root: Path) -> PublicationFeatureStore:
    document = json.loads((root / "manifest.json").read_text())
    from ced.model.prepare import save_tensor

    store = PublicationFeatureStore(
        repo_id="fixture/embeddings",
        revision="a" * 40,
        source_sha256="b" * 64,
        dimension=3,
        statements=document["statements"],
        statement_publications=2,
        matched_publications=1,
        statements_with_publications=2,
        statements_with_embeddings=1,
        feature_key="fixture-key",
        vectors=save_tensor(
            root,
            "publication-vectors",
            np.array([[1, 2, 3], [0, 0, 0], [4, 5, 6]], dtype=np.float32),
        ),
        evidence=save_tensor(root, "publication-evidence", np.array([1, 0, 0.5], dtype=np.float32)),
    )
    manifest = DatasetManifest.model_validate(document)
    manifest.publications = store
    for reference in (store.vectors, store.evidence):
        manifest.checksums[reference.path] = reference.sha256
    (root / "manifest.json").write_text(manifest.model_dump_json())
    return store


def test_publication_provider_projects_features_and_excludes_them_from_checkpoints() -> None:
    base = RandomProjectionProvider(8, 4, relations=1, seed=11)
    vectors = torch.tensor([[1, 2, 3], [0, 0, 0], [4, 5, 6]], dtype=torch.float32)
    evidence = torch.tensor([1, 0, 0.5], dtype=torch.float32)
    provider = PublicationFeatureProvider(8, 4, 1, 11, vectors, evidence)
    relations = torch.tensor([0, 1, 0], dtype=torch.long)
    statements = torch.tensor([0, 1, 2], dtype=torch.long)
    expected_base = base.statement_features(relations, statements)
    projected = provider.publication_projection(torch.cat((vectors, evidence[:, None]), dim=1))
    actual = provider.statement_features(relations, statements)
    torch.testing.assert_close(actual, expected_base + projected)
    assert torch.equal(actual[1], expected_base[1])
    actual.sum().backward()
    assert provider.publication_projection.weight.grad is not None
    assert torch.isfinite(provider.publication_projection.weight.grad).all()
    assert "publication_vectors" not in provider.state_dict()
    assert "publication_evidence" not in provider.state_dict()


def test_publication_features_reach_training_and_modal_transfer(tmp_path: Path, synthetic_dataset) -> None:
    from ced.modal_plan import dataset_transfer_plan
    from ced.model.prepare import save_tensor
    from ced.model.schemas import TrainingExample
    from ced.model.training import checkpoint, train_step

    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    store = _attach_publication_features(root)
    document = json.loads((root / "manifest.json").read_text())
    model = build_model(document, root, edge_type_count=2, seed=7)
    assert isinstance(model.features, PublicationFeatureProvider)
    graph = TrainingExample.model_validate(document["batches"][0]).graph
    graph.global_node_id = save_tensor(root, "publication-test-global-nodes", np.array([0, 1], dtype=np.int64))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_step(model, optimizer, root, document, graph, None, device=torch.device("cpu"), auxiliary_weight=0.1)
    assert math.isfinite(loss)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint(model, checkpoint_path, {"loss": loss})
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state"]
    assert "features.publication_projection.weight" in state
    assert "features.publication_vectors" not in state

    runtime = RuntimeConfig(remote_project=tmp_path, artifact_root=tmp_path / ".ced", disk_reserve_gb=0)
    for mode in ("train", "evaluate"):
        plan = dataset_transfer_plan(root, runtime, mode)
        assert store.vectors.path in plan.shared
        assert store.evidence.path in plan.shared


def test_evaluation_reuses_source_statement_ids_for_reverse_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_dataset
) -> None:
    from ced.model.evaluation import evaluate_dataset

    class RecordingProvider(RandomProjectionProvider):
        seen: torch.Tensor | None = None

        def statement_features(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor:
            self.seen = statement_id.detach().cpu().clone()
            return super().statement_features(relation_id, statement_id)

    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    document = json.loads((root / "manifest.json").read_text())
    model = build_model(document, root, edge_type_count=2, seed=7)
    provider = RecordingProvider(32, 16, relations=1, seed=7)
    model.features = provider
    monkeypatch.setattr("ced.model.evaluation.build_model", lambda *args, **kwargs: model)
    evaluate_dataset(root, split="test", seed=3, device="cpu")
    assert provider.seen is not None
    assert provider.seen.tolist() == [0, 0]
