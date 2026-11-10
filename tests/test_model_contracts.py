from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from ced.config.models import SplitSettings
from ced.kg.records import attribute_events
from ced.model.features import fit_features
from ced.model.negatives import sample_negatives, typed_pools
from ced.model.prepare import save_tensor
from ced.model.split import group_split, statement_group


def test_grouped_splits_and_declared_equivalences() -> None:
    settings = SplitSettings()
    group = statement_group("X:1", "p", "X:2")
    assert group_split(group, settings) == group_split(group, settings)
    assert group_split(group, settings, schema=True) == 3
    assert statement_group("X:1", "p", "X:2", symmetric={"p"}) == statement_group("X:2", "p", "X:1", symmetric={"p"})
    inverses = {"forward": "reverse", "reverse": "forward"}
    assert statement_group("X:1", "forward", "X:2", inverses=inverses) == statement_group(
        "X:2", "reverse", "X:1", inverses=inverses
    )


def test_feature_fit_excludes_heldout_owners() -> None:
    events = attribute_events("node", "train", "s", {"numeric": 2, "category": "known"})
    events += attribute_events("node", "heldout", "s", {"numeric": 1000, "category": "secret", "test_only": "leak"})
    fitted = fit_features(events, {"train"}, set())
    assert next(field for field in fitted.fields if field.path == "numeric").mean == 2
    assert not any(field.path == "test_only" for field in fitted.fields)
    assert '"secret"' not in fitted.categories["node\0category"]


def test_negatives_use_global_ids_and_filter_all_positives() -> None:
    edges = np.array([[10, 11, 10], [12, 13, 13]], dtype=np.int64)
    relations = np.array([0, 0, 0], dtype=np.int64)
    splits = np.array([0, 0, 2], dtype=np.int8)
    types = np.array([*([9] * 10), 1, 1, 2, 2, 2], dtype=np.int64)
    pools = typed_pools(edges, relations, splits, types)
    positives = {(10, 0, 12), (11, 0, 13), (10, 0, 13)}
    head, tail, valid = sample_negatives([(10, 0, 12)], pools, positives, count=8, seed=1)
    assert valid.any()
    for h, t, v in zip(head[0], tail[0], valid[0], strict=True):
        if v:
            assert (int(h), 0, int(t)) not in positives
            assert h >= 10
            assert t >= 12


def test_tensor_ref_checks_actual_dtype_shape_and_hash(tmp_path: Path) -> None:
    ref = save_tensor(tmp_path, "ids", np.array([1, 2], dtype=np.int64))
    assert ref.load(tmp_path).tolist() == [1, 2]
    bad = ref.model_copy(update={"shape": [3]})
    with pytest.raises(ValueError, match="shape/dtype"):
        bad.load(tmp_path)
    np.save(tmp_path / ref.path, np.array([3, 4], dtype=np.int64))
    with pytest.raises(ValueError, match="checksum mismatch"):
        ref.load(tmp_path)


def test_neighbor_sampler_excludes_heldout_and_schema() -> None:
    from ced.model.batches import sample_neighborhood

    edges = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int64)
    relations = np.array([0, 0, 0], dtype=np.int64)
    splits = np.array([0, 2, 3], dtype=np.int8)
    result = sample_neighborhood(edges, relations, splits, np.array([0]), fanout=[10], seed=1, node_limit=4)
    assert result.global_node_id.tolist() == [0, 1]
    assert result.statement_id.tolist() == [0, 0]
    assert result.relation_id.tolist() == [0, 1]
    assert result.edge_index.tolist() == [[0, 1], [1, 0]]


def test_disk_graph_matches_in_memory_negatives_and_sampling(tmp_path: Path) -> None:
    from ced.model.disk_graph import DiskGraph
    from ced.model.negatives import sample_negatives, typed_pools

    edges = np.array([[10, 10, 11, 12], [12, 13, 13, 13]], dtype=np.int64)
    relations = np.array([0, 1, 0, 1], dtype=np.int64)
    splits = np.array([0, 0, 0, 2], dtype=np.int8)
    types = np.array([*([9] * 10), 1, 1, 2, 2], dtype=np.int64)
    positives = {(10, 0, 12), (11, 0, 13), (10, 1, 13)}
    pools = typed_pools(edges, relations, splits, types)
    triples = [(10, 0, 12), (10, 1, 13)]
    disk = DiskGraph(
        tmp_path / "graph.db", {"edge_index": edges, "relation_id": relations, "split": splits, "type_set_id": types}
    )
    try:
        disk_heads, disk_tails, disk_valid = disk.negatives(triples, count=16, seed=7)
        mem_heads, mem_tails, mem_valid = sample_negatives(triples, pools, positives, count=16, seed=7)
        assert np.array_equal(disk_heads, mem_heads)
        assert np.array_equal(disk_tails, mem_tails)
        assert np.array_equal(disk_valid, mem_valid)
        for h, t, v in zip(disk_heads[0], disk_tails[0], disk_valid[0], strict=True):
            if v:
                assert (int(h), 0, int(t)) not in positives
        neighborhood = disk.neighborhood(edges, relations, np.array([10]), fanout=[10], seed=1, node_limit=14)
        assert set(map(int, neighborhood.global_node_id)) == {10, 12, 13}
        assert set(map(int, neighborhood.statement_id)) == {0, 1}
    finally:
        disk.close()


def test_torch_encoder_decoder_and_losses() -> None:
    torch = pytest.importorskip("torch")
    from ced.model.decoder import ComplexDecoder
    from ced.model.encoder import RelationalAttentionEncoder
    from ced.model.losses import AuxiliaryAttributeDecoder, link_loss

    encoder = RelationalAttentionEncoder(8, 4, edge_types=2, layers=1)
    state = encoder(torch.randn(3, 8), torch.tensor([[0, 1], [1, 2]]), torch.tensor([0, 1]), torch.randn(2, 4))
    decoder = ComplexDecoder(8, 2)
    scores = decoder.score(state[[0, 1]], torch.tensor([0, 1]), state[[1, 2]])
    assert scores.shape == (2,)
    assert torch.isfinite(link_loss(scores, torch.tensor([1.0, 0.0]))).item()
    auxiliary = AuxiliaryAttributeDecoder(8, {2: 4}, [5])
    category, number = auxiliary(
        state,
        torch.tensor([2, 2, 2]),
        torch.tensor([0, 1, 2]),
        torch.tensor([5, 5, 5]),
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([True, False, True]),
    )
    assert torch.isfinite(category + number).item()


def test_exact_filtered_rank_removes_known_positives() -> None:
    torch = pytest.importorskip("torch")
    from ced.model.evaluation import filtered_rank

    scores = torch.tensor([3.0, 2.0, 1.0])
    assert filtered_rank(scores, 0, {1}) == 1
    assert filtered_rank(scores, 2, {0}) == 2


def test_training_checkpoint_resumes_epoch(tmp_path: Path) -> None:
    import json

    from ced.model.prepare import save_tensor
    from ced.model.schemas import GraphBatch, NegativeBatch, TrainingExample
    from ced.model.training import train_batches

    root = tmp_path
    topology = {
        "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "relation_id": np.array([0, 0], dtype=np.int64),
        "split": np.array([0, 1], dtype=np.int8),
        "type_set_id": np.array([0, 0], dtype=np.int64),
    }
    tensors = {name: save_tensor(root, name, values) for name, values in topology.items()}
    directory = root / "batch"
    directory.mkdir()

    def save(name: str, values: np.ndarray):
        reference = save_tensor(directory, name, values)
        reference.path = f"batch/{reference.path}"
        return reference

    graph = GraphBatch(
        global_node_id=save("global_node_id", np.array([0, 1], dtype=np.int64)),
        seed_id=save("seed_id", np.array([0], dtype=np.int64)),
        type_set_id=save("type_set_id", np.array([0, 0], dtype=np.int64)),
        edge_index=save("edge_index", np.array([[0, 1], [1, 0]], dtype=np.int64)),
        relation_id=save("relation_id", np.array([0, 1], dtype=np.int64)),
        edge_type_id=save("edge_type_id", np.array([0, 1], dtype=np.int64)),
        statement_id=save("statement_id", np.array([0, 0], dtype=np.int64)),
        positive_statement_id=save("positive_statement_id", np.array([0], dtype=np.int64)),
        negative=NegativeBatch(
            subject=save("negative_subject", np.zeros((1, 2), dtype=np.int64)),
            object=save("negative_object", np.ones((1, 2), dtype=np.int64)),
            valid=save("negative_valid", np.array([[True, False]])),
        ),
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "relations": 1,
                "entities": 2,
                "statements": 2,
                "tensors": {k: v.model_dump(mode="json") for k, v in tensors.items()},
                "batches": [TrainingExample(graph=graph, split="train").model_dump(mode="json")],
            }
        )
    )
    checkpoint = tmp_path / "run" / "model.pt"
    torch = pytest.importorskip("torch")
    first = train_batches(root, epochs=1, seed=1, checkpoint_path=checkpoint)
    assert first["batches"] == 1
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"] == 1
    second = train_batches(root, epochs=2, seed=1, checkpoint_path=checkpoint)
    assert second["batches"] == 2
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"] == 2


def test_random_text_encoder_is_deterministic_and_normalized() -> None:
    torch = pytest.importorskip("torch")
    from ced.model.encoding import RandomTextEncoder

    encoder = RandomTextEncoder(8, normalize=True, seed=7)
    first = encoder.encode(["alpha", "beta"])
    second = encoder.encode(["alpha", "beta"])
    assert torch.equal(first, second)
    assert torch.allclose(first.norm(dim=-1), torch.ones(2))


def test_text_embedding_cache_reuses_and_appends(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from ced.config.models import TextEncodingSettings
    from ced.model.encoding import RandomTextEncoder, TextEmbeddingCache, text_key

    settings = TextEncodingSettings(provider="random", dimension=4, batch_size=2)
    encoder = RandomTextEncoder(4, seed=11)
    cache = TextEmbeddingCache(tmp_path, settings)
    rows, entries = cache.embed(["one", "two"], encoder)
    assert len(set(entries)) == 2
    assert rows.tolist() == [entries[text_key("one")], entries[text_key("two")]]
    rows2, entries2 = cache.embed(["one", "three"], encoder)
    assert rows2[0] == rows[0]
    assert entries2[text_key("one")] == entries[text_key("one")]
    assert len(entries2) == 3


def test_sentence_transformer_encoder_uses_local_model(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    import sys
    import types

    from ced.config.models import TextEncodingSettings
    from ced.model.encoding import SentenceTransformerTextEncoder

    calls: dict[str, tuple] = {}

    class FakeModel:
        max_seq_length = 0

        def __init__(self, model: str, device: str, local_files_only: bool) -> None:
            calls["init"] = (model, device, local_files_only)

        def get_sentence_embedding_dimension(self) -> int:
            return 4

        def encode(self, texts: list[str], **kwargs: object) -> object:
            calls["encode"] = (texts, kwargs)
            return torch.ones((len(texts), 4))

    module = types.SimpleNamespace(SentenceTransformer=FakeModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    settings = TextEncodingSettings(
        provider="sentence_transformers", model="fixture/model", dimension=4, device="cpu", local_files_only=True
    )
    encoder = SentenceTransformerTextEncoder(settings)
    assert calls["init"] == ("fixture/model", "cpu", True)
    assert encoder.model.max_seq_length == settings.max_length
    vectors = encoder.encode(["alpha"])
    assert vectors.shape == (1, 4)
    encode_call = calls["encode"]
    assert isinstance(encode_call, tuple)
    assert encode_call[1]["normalize_embeddings"] is True


def test_feature_ranges(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from ced.model.feature_index import _ranges

    owners = torch.tensor([0, 0, 2, 2, 2, 5], dtype=torch.long)
    rows = _ranges(owners, torch.tensor([0, 2, 5], dtype=torch.long))
    assert rows.tolist() == [0, 1, 2, 3, 4, 5]
    assert _ranges(owners, torch.tensor([1], dtype=torch.long)).numel() == 0


def test_evaluation_supports_exact_and_sampled_ranking(tmp_path: Path, synthetic_dataset: Callable[..., int]) -> None:
    pytest.importorskip("torch")
    from ced.model.evaluation import evaluate_dataset

    root = tmp_path / "dataset"
    synthetic_dataset(root, batches=1)
    exact = evaluate_dataset(root, split="test", seed=3, device="cpu", ranking="exact")
    sampled = evaluate_dataset(root, split="test", seed=3, device="cpu", ranking="sampled", sampled_candidates=2)
    assert exact["ranking"] == "exact"
    assert sampled["ranking"] == "sampled"
    assert sampled["sampled_candidates"] == 2
    assert exact["support"] == sampled["support"] == 2
    assert "ced_family" in exact
    assert (
        evaluate_dataset(root, split="test", seed=3, device="cpu", ranking="sampled", sampled_candidates=2) == sampled
    )
