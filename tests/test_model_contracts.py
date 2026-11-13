from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from ced.config.models import SplitSettings
from ced.kg.records import attribute_events
from ced.model.features import fit_features
from ced.model.negatives import negative_context, sample_negatives, typed_pools
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
    context = negative_context(edges[0], relations, edges[1])
    head, tail, valid = sample_negatives([(10, 0, 12)], pools, context, count=8, seed=1)
    assert valid.any()
    for h, t, v in zip(head[0], tail[0], valid[0], strict=True):
        if v:
            assert (int(h), 0, int(t)) not in positives
            assert h >= 10
            assert t >= 12


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("int64", np.array([0, -1, 2], dtype=np.int64)),
        ("float32", np.array([0.5, -1.25, 2.0], dtype=np.float32)),
        ("bool", np.array([True, False, True])),
        ("int8", np.array([0, 3, 2], dtype=np.int8)),
    ],
)
def test_safetensors_tensor_ref_round_trips_every_supported_dtype(
    tmp_path: Path, dtype: str, values: np.ndarray
) -> None:
    """Every model dtype must survive the safetensors encoding exactly."""
    from ced.model.tensors import DEFAULT_COLUMN

    ref = save_tensor(tmp_path, "values", values)
    assert ref.path == "values.safetensors"
    assert ref.column == DEFAULT_COLUMN
    assert ref.dtype == dtype
    assert ref.shape == [3]
    assert (tmp_path / "values.safetensors").is_file()
    loaded = ref.load(tmp_path)
    assert loaded.dtype == np.dtype(dtype)
    assert np.array_equal(loaded, values)
    assert loaded.flags.writeable
    assert loaded.flags.c_contiguous


def test_safetensors_tensor_ref_restores_multidimensional_shape(tmp_path: Path) -> None:
    """Batch matrices must retain their rank and reject same-sized shape changes."""
    values = np.arange(6, dtype=np.int64).reshape(2, 3)
    ref = save_tensor(tmp_path, "edge_index", values)
    assert ref.shape == [2, 3]
    loaded = ref.load(tmp_path)
    assert loaded.shape == (2, 3)
    assert np.array_equal(loaded, values)
    assert loaded.flags.c_contiguous
    with pytest.raises(ValueError, match="shape mismatch"):
        ref.model_copy(update={"shape": [3, 2]}).load(tmp_path)


def test_safetensors_tensor_ref_round_trips_empty_tensor(tmp_path: Path) -> None:
    values = np.array([], dtype=np.int64)
    ref = save_tensor(tmp_path, "empty", values)
    assert ref.shape == [0]
    assert np.array_equal(ref.load(tmp_path), values)


def test_safetensors_streaming_writer_round_trips_batches(tmp_path: Path) -> None:
    from ced.acquisition.base import sha256_file
    from ced.model.schemas import TensorRef
    from ced.model.tensors import write_tensor_stream

    path = tmp_path / "streamed.safetensors"
    write_tensor_stream(
        path,
        [np.array([0, 1], dtype=np.int64), np.array([2], dtype=np.int64)],
        shape=[3],
        dtype="int64",
    )
    ref = TensorRef(path=path.name, dtype="int64", shape=[3], sha256=sha256_file(path))
    assert ref.load(tmp_path).tolist() == [0, 1, 2]
    empty_path = tmp_path / "empty-stream.safetensors"
    write_tensor_stream(empty_path, [], shape=[0], dtype="float32")
    empty = TensorRef(path=empty_path.name, dtype="float32", shape=[0], sha256=sha256_file(empty_path))
    assert empty.load(tmp_path).shape == (0,)
    with pytest.raises(ValueError, match="shape mismatch"):
        write_tensor_stream(path, [np.array([3], dtype=np.int64)], shape=[2], dtype="int64")
    with pytest.raises(ValueError, match="dtype mismatch"):
        write_tensor_stream(path, [np.array([3], dtype=np.float32)], shape=[1], dtype="int64")
    assert not list(tmp_path.glob("*.raw.tmp-*"))


def test_safetensors_streaming_writer_preserves_serialization_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ced.model import tensors

    def fail_serialization(*args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(tensors, "serialize_file", fail_serialization)
    with pytest.raises(OSError, match="disk full"):
        tensors.write_tensor_stream(
            tmp_path / "failed.safetensors",
            [np.array([1], dtype=np.int64)],
            shape=[1],
            dtype="int64",
        )
    assert not list(tmp_path.glob("*.tmp-*"))


def test_safetensors_tensor_refs_share_one_file_by_column(tmp_path: Path) -> None:
    """Several refs may address named tensors in one safetensors file."""
    from safetensors.numpy import save_file

    from ced.acquisition.base import sha256_file
    from ced.model.schemas import TensorRef

    path = tmp_path / "topology.safetensors"
    save_file(
        {
            "head": np.array([0, 1], dtype=np.int64),
            "tail": np.array([1, 0], dtype=np.int64),
        },
        path,
    )
    digest = sha256_file(path)

    def ref(column: str) -> TensorRef:
        return TensorRef(path=path.name, column=column, dtype="int64", shape=[2], sha256=digest)

    assert ref("head").load(tmp_path).tolist() == [0, 1]
    assert ref("tail").load(tmp_path).tolist() == [1, 0]
    with pytest.raises(ValueError, match="column is missing"):
        ref("absent").load(tmp_path)
    with pytest.raises(ValueError, match="dtype mismatch"):
        ref("head").model_copy(update={"dtype": "float32"}).load(tmp_path)
    with pytest.raises(ValueError, match="shape mismatch"):
        ref("head").model_copy(update={"shape": [3]}).load(tmp_path)


def test_safetensors_tensor_ref_rejects_hash_mismatch(tmp_path: Path) -> None:
    """A mutated tensor payload must fail before training or evaluation reads it."""
    ref = save_tensor(tmp_path, "ids", np.array([1, 2], dtype=np.int64))
    assert ref.load(tmp_path).tolist() == [1, 2]
    save_tensor(tmp_path, "ids", np.array([3, 4], dtype=np.int64))
    with pytest.raises(ValueError, match="checksum mismatch"):
        ref.load(tmp_path)


def test_write_tensor_rejects_unsupported_dtype(tmp_path: Path) -> None:
    """An unsupported dtype must raise rather than be quietly downcast.

    Why: float64 is the natural dtype of a NumPy computation. Silently storing it as
    float32 would change feature values; the writer must refuse instead.
    """
    from ced.model.tensors import write_tensor

    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        write_tensor(tmp_path / "bad.safetensors", np.array([1.0, 2.0], dtype=np.float64))


def test_read_tensor_rejects_missing_files_columns_and_dtype_drift(tmp_path: Path) -> None:
    """read_tensor is the last line of defense before indices reach training; a missing
    file, a renamed column and a dtype that drifted from the manifest must all raise."""
    from ced.model.tensors import read_tensor, tensor_dtype, write_tensor

    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        tensor_dtype("float64")
    with pytest.raises(FileNotFoundError, match="tensor is missing"):
        read_tensor(tmp_path / "absent.parquet", column="value", dtype="int64", shape=[1])

    path = tmp_path / "tensors.safetensors"
    write_tensor(path, np.array([1, 2, 3], dtype=np.int64), column="head")
    loaded = read_tensor(path, column="head", dtype="int64", shape=[3])
    assert loaded.tolist() == [1, 2, 3]

    write_tensor(path, np.array([1, 2], dtype=np.int64), column="tail")
    with pytest.raises(ValueError, match="column is missing"):
        read_tensor(path, column="head", dtype="int64", shape=[2])
    with pytest.raises(ValueError, match="dtype mismatch"):
        read_tensor(path, column="tail", dtype="float32", shape=[2])


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
    from ced.model.negatives import negative_context, sample_negatives, typed_pools

    edges = np.array([[10, 10, 11, 12], [12, 13, 13, 13]], dtype=np.int64)
    relations = np.array([0, 1, 0, 1], dtype=np.int64)
    splits = np.array([0, 0, 0, 2], dtype=np.int8)
    types = np.array([*([9] * 10), 1, 1, 2, 2], dtype=np.int64)
    positives = {(10, 0, 12), (11, 0, 13), (10, 1, 13)}
    pools = typed_pools(edges, relations, splits, types)
    context = negative_context(edges[0], relations, edges[1])
    triples = [(10, 0, 12), (10, 1, 13)]
    disk = DiskGraph(
        tmp_path / "graph.db",
        {
            "head": edges[0],
            "tail": edges[1],
            "relation_id": relations,
            "split": splits,
            "type_set_id": types,
        },
    )
    try:
        disk_heads, disk_tails, disk_valid = disk.negatives(triples, count=16, seed=7)
        mem_heads, mem_tails, mem_valid = sample_negatives(triples, pools, context, count=16, seed=7)
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
    # _auxiliary_tensors returns categorical states first, then numeric states;
    # each group loss sees only its own slice.
    mixed = torch.cat((state, state))
    category, number = auxiliary(
        mixed,
        torch.tensor([2, 2, 2]),
        torch.tensor([0, 1, 2]),
        torch.tensor([5, 5, 5]),
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([True, False, True]),
    )
    assert torch.isfinite(category + number).item()
    with pytest.raises(ValueError, match="misaligned"):
        auxiliary(
            state,
            torch.tensor([2, 2, 2]),
            torch.tensor([0, 1, 2]),
            torch.tensor([5, 5, 5]),
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([True, False, True]),
        )


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
        "head": np.array([0, 1], dtype=np.int64),
        "tail": np.array([1, 0], dtype=np.int64),
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


def test_text_embedding_cache_appends_parts_without_rewriting(tmp_path: Path) -> None:
    """Appending writes only new parts and never reassigns an existing row id.

    Why: the previous single-array cache rewrote every stored vector on every run,
    so incremental encoding cost O(total) per call and O(total^2) overall. A
    renumbered row would also silently point datasets that already reference
    text-index.parquet at the wrong embedding.
    """
    pytest.importorskip("torch")
    import duckdb

    from ced.acquisition.base import sha256_file
    from ced.config.models import TextEncodingSettings
    from ced.model.encoding import RandomTextEncoder, TextEmbeddingCache, text_key

    settings = TextEncodingSettings(provider="random", dimension=4, batch_size=2)
    encoder = RandomTextEncoder(4, seed=11)
    cache = TextEmbeddingCache(tmp_path, settings)

    def batches(texts: list[str]):
        return iter([(texts, encoder.encode(texts).detach().cpu().numpy())])

    assert cache.append_stream(batches(["one", "two"])) == 2
    first_parts = {path.name: sha256_file(path) for path in sorted(cache.parts.glob("*.safetensors"))}
    assert cache.append_stream(batches(["three"])) == 1

    second_parts = {path.name: sha256_file(path) for path in sorted(cache.parts.glob("*.safetensors"))}
    assert first_parts == {name: second_parts[name] for name in first_parts}
    assert len(second_parts) == len(first_parts) + 1
    assert cache.manifest()["rows"] == 3

    connection = duckdb.connect()
    try:
        cache.create_index(connection)
        rows = dict(connection.execute("SELECT text_sha256, row FROM cache_index").fetchall())
    finally:
        connection.close()
    assert rows == {text_key("one"): 0, text_key("two"): 1, text_key("three"): 2}

    references = cache.part_refs()
    assert [reference.shape for reference in references] == [[2, 4], [1, 4]]
    loaded = references[1].load(tmp_path)
    assert loaded.shape == (1, 4)
    assert np.isfinite(loaded).all()


def test_text_embedding_cache_rejects_wrong_dimension(tmp_path: Path) -> None:
    """A batch whose width disagrees with the cache dimension must raise.

    Why: writing a mis-shaped batch would corrupt every later row offset in the
    part, and the damage would only surface as garbage embeddings much later.
    """
    pytest.importorskip("torch")
    from ced.config.models import TextEncodingSettings
    from ced.model.encoding import TextEmbeddingCache

    settings = TextEncodingSettings(provider="random", dimension=4, batch_size=2)
    cache = TextEmbeddingCache(tmp_path, settings)
    with pytest.raises(ValueError, match="does not match cache dimension"):
        cache.append_stream(iter([(["one"], np.zeros((1, 8), dtype=np.float32))]))


def test_sentence_transformer_encoder_uses_local_model(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    import sys
    import types

    from ced.config.models import TextEncodingSettings
    from ced.model.encoding import SentenceTransformerTextEncoder

    calls: dict[str, tuple] = {}

    class FakeModel:
        max_seq_length = 0

        def __init__(self, model: str, revision: str | None, device: str, local_files_only: bool) -> None:
            calls["init"] = (model, revision, device, local_files_only)

        def get_sentence_embedding_dimension(self) -> int:
            return 4

        def encode(self, texts: list[str], **kwargs: object) -> object:
            calls["encode"] = (texts, kwargs)
            return torch.ones((len(texts), 4))

    module = types.SimpleNamespace(SentenceTransformer=FakeModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    settings = TextEncodingSettings(
        provider="sentence_transformers",
        model="fixture/model",
        revision="pinned-revision",
        dimension=4,
        device="cpu",
        local_files_only=True,
    )
    encoder = SentenceTransformerTextEncoder(settings)
    assert calls["init"] == ("fixture/model", "pinned-revision", "cpu", True)
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


def test_split_bounds_stay_inside_ubigint_for_a_zero_test_split() -> None:
    """A zero test split must not push the validation bound past UBIGINT range.

    Why: `train + validation == 1.0` gives an unclamped bound of 2**64, which needs 17
    hex digits. The SQL split CASE casts the hex string to UBIGINT, so preparation
    would abort with a conversion error while the Python reference kept working. The
    two engines would disagree in a way no user could diagnose from the message.
    """
    import duckdb

    from ced.model.split import split_bounds

    settings = SplitSettings(train=0.9, validation=0.1, test=0.0)
    train_bound, valid_bound = split_bounds(settings)
    assert len(train_bound) <= 16
    assert len(valid_bound) <= 16

    connection = duckdb.connect()
    try:
        cast = connection.execute("SELECT ('0x' || ?)::UBIGINT, ('0x' || ?)::UBIGINT", [train_bound, valid_bound])
        bounds = cast.fetchone()
    finally:
        connection.close()
    assert bounds is not None
    assert bounds[1] == 2**64 - 1

    groups = [statement_group(f"X:{i}", "p", f"Y:{i}") for i in range(500)]
    assert {group_split(group, settings) for group in groups} <= {0, 1}
    for train, validation, test in ((0.8, 0.1, 0.1), (0.7, 0.2, 0.1), (0.6, 0.2, 0.2)):
        bounds = split_bounds(SplitSettings(train=train, validation=validation, test=test))
        assert all(int(bound, 16) <= 2**64 - 1 for bound in bounds)


def test_text_embedding_cache_ignores_unpublished_index_parts(tmp_path: Path) -> None:
    """An index part missing from the manifest must not make its texts look cached.

    Why: a crash between the index part write and the manifest update leaves exactly
    such an orphan. If the sha -> row table included it, the anti-join would skip those
    texts, no rows would be appended, and the emitted text index would carry row ids
    past the end of the published vectors. Every consumer would then read the wrong
    embedding, or none, without any error being raised.
    """
    pytest.importorskip("torch")
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ced.config.models import TextEncodingSettings
    from ced.model.encoding import RandomTextEncoder, TextEmbeddingCache, text_key

    settings = TextEncodingSettings(provider="random", dimension=4, batch_size=2)
    encoder = RandomTextEncoder(4, seed=11)
    cache = TextEmbeddingCache(tmp_path, settings)

    def batches(texts: list[str]):
        return iter([(texts, encoder.encode(texts).detach().cpu().numpy())])

    assert cache.append_stream(batches(["one"])) == 1

    # Simulate the crash: a complete index part the manifest never learned about.
    pq.write_table(
        pa.table({"text_sha256": pa.array([text_key("ghost")], pa.string()), "row": pa.array([1], pa.int64())}),
        cache.index / "part-000001.parquet",
    )

    def index_rows() -> dict[str, int]:
        connection = duckdb.connect()
        try:
            cache.create_index(connection)
            return dict(connection.execute("SELECT text_sha256, row FROM cache_index").fetchall())
        finally:
            connection.close()

    assert index_rows() == {text_key("one"): 0}

    # The next append reuses the orphan name, so the crash self-heals.
    assert cache.append_stream(batches(["two"])) == 1
    assert [part["name"] for part in cache.manifest()["parts"]] == [
        "part-000000.safetensors",
        "part-000001.safetensors",
    ]
    assert index_rows() == {text_key("one"): 0, text_key("two"): 1}
    assert not list(cache.index.glob("*.tmp-*"))
