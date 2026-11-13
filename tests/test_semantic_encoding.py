from __future__ import annotations

import sys
import types
from collections.abc import Sequence
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn

from ced.config.models import EnsembleTextEncodingSettings, TextEncodingSettings, TextModelSettings
from ced.model.encoding import (
    RandomTextEncoder,
    TextEmbeddingCache,
    encode_missing_to_parts,
    iter_encode_missing_parts,
    text_key,
)
from ced.model.tensors import write_tensor
from ced.model.text_models import ConcatenatedTextEncoder, SapBertTextEncoder, SentenceTransformerComponent

REVISION = "0" * 40


class Component(nn.Module):
    def __init__(self, dimension: int, value: float) -> None:
        super().__init__()
        self.dimension = dimension
        self.value = value

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        values = torch.arange(1, self.dimension + 1, dtype=torch.float32) * self.value
        return values.expand(len(texts), self.dimension)


def test_concatenated_encoder_normalizes_each_block_in_order() -> None:
    encoder = ConcatenatedTextEncoder([Component(2, 3), Component(3, 4), Component(1, 5)])
    vectors = encoder.encode(["a", "b"])
    assert vectors.shape == (2, 6)
    first = torch.nn.functional.normalize(torch.tensor([3.0, 6.0]), dim=0)
    assert torch.allclose(vectors[:, :2], first.expand(2, 2))
    assert torch.allclose(vectors[:, 2:5].norm(dim=-1), torch.ones(2))
    assert torch.equal(vectors[:, 5], torch.ones(2))


def test_ensemble_defaults_pin_models_and_cheap_gpu() -> None:
    settings = TextEncodingSettings(provider="modal_ensemble")
    assert settings.ensemble.dimension == 2560
    assert settings.ensemble.gpu == "L4"
    assert settings.ensemble.sapbert.dimension == 768
    assert settings.ensemble.gemma.dimension == 768
    assert settings.ensemble.qwen.dimension == 1024
    for spec in (settings.ensemble.sapbert, settings.ensemble.gemma, settings.ensemble.qwen):
        assert len(spec.revision) == 40


def test_sapbert_uses_cls_and_length_grouped_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class Tokens(dict[str, torch.Tensor]):
        def to(self, device: torch.device) -> Tokens:
            return self

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Tokenizer:
            return cls()

        def __call__(self, texts: list[str], **kwargs: object) -> Tokens:
            calls.append(texts)
            return Tokens(input_ids=torch.tensor([[len(text), 0] for text in texts], dtype=torch.float32))

    class Model:
        config = types.SimpleNamespace(hidden_size=2)

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Model:
            return cls()

        def to(self, device: torch.device) -> Model:
            return self

        def eval(self) -> Model:
            return self

        def __call__(self, input_ids: torch.Tensor) -> types.SimpleNamespace:
            first = torch.stack((input_ids[:, 0], input_ids[:, 0] + 1), dim=-1).unsqueeze(1)
            second = torch.zeros_like(first)
            return types.SimpleNamespace(last_hidden_state=torch.cat((first, second), dim=1))

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModel=Model, AutoTokenizer=Tokenizer))
    settings = EnsembleTextEncodingSettings(batch_size=2, max_length=8, local_files_only=True)
    encoder = SapBertTextEncoder(
        TextModelSettings(model="fixture/sap", revision=REVISION, dimension=2), settings, "cpu"
    )
    vectors = encoder.encode(["a", "bbbb", "cc"])
    assert calls == [["a", "cc"], ["bbbb"]]
    for index, length in enumerate((1, 4, 2)):
        expected = torch.nn.functional.normalize(torch.tensor([float(length), float(length + 1)]), dim=0)
        assert torch.allclose(vectors[index], expected)


def test_sentence_transformer_components_use_text_only_document_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class Model:
        max_seq_length = 0

        def __init__(self, model: str, **kwargs: object) -> None:
            calls[model] = kwargs

        def eval(self) -> Model:
            return self

        def get_sentence_embedding_dimension(self) -> int:
            return 2

        def encode(self, texts: list[str], **kwargs: object) -> torch.Tensor:
            calls["encode"] = kwargs
            return torch.ones((len(texts), 2))

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Model))
    settings = EnsembleTextEncodingSettings(local_files_only=True)
    spec = TextModelSettings(model="fixture/gemma", revision=REVISION, dimension=2)
    encoder = SentenceTransformerComponent(spec, settings, "cpu", gemma=True)
    init = calls[spec.model]
    assert isinstance(init, dict)
    assert init["config_kwargs"] == {"vision_config": None, "audio_config": None}
    encoder.encode(["name"])
    assert isinstance(calls["encode"], dict)
    assert calls["encode"]["prompt_name"] == "document"

    qwen = TextModelSettings(model="fixture/qwen", revision=REVISION, dimension=2)
    encoder = SentenceTransformerComponent(qwen, settings, "cpu", qwen=True)
    init = calls[qwen.model]
    assert isinstance(init, dict)
    assert init["tokenizer_kwargs"] == {"padding_side": "left"}
    assert encoder.prompt_name is None


def test_encode_missing_streams_each_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Part generation is lazy and carries deterministic transfer metadata.

    Why: the consumer must finish adopting one part before the encoder computes
    the next. Descriptors also need stable offsets and checksums so interrupted
    streams cannot publish partial or misordered files.
    """
    from ced.acquisition.base import sha256_file

    monkeypatch.setattr("ced.model.encoding.PART_ROWS", 2)
    missing = tmp_path / "missing.parquet"
    texts = ["one", "two", "three"]
    pq.write_table(pa.table({"text_sha256": [text_key(text) for text in texts], "text": texts}), missing)
    encoded = tmp_path / "encoded"
    encoded_texts: list[str] = []

    class RecordingEncoder(RandomTextEncoder):
        def encode(self, texts: Sequence[str]) -> torch.Tensor:
            encoded_texts.extend(texts)
            return super().encode(texts)

    stream = iter_encode_missing_parts(missing, encoded, RecordingEncoder(4), start_row=8, start_part=4)
    first = next(stream)
    assert encoded_texts == ["one", "two"]
    assert first["input_row_start"] == 0
    assert first["part"] == 4
    assert first["rows"] == 2
    assert first["files"] == ["vectors/part-000004.safetensors", "index/part-000004.parquet"]
    assert first["checksums"] == {name: sha256_file(encoded / name) for name in first["files"]}

    second = next(stream)
    assert encoded_texts == texts
    assert second["input_row_start"] == 2
    assert second["part"] == 5
    assert second["rows"] == 1
    with pytest.raises(StopIteration):
        next(stream)


def test_modal_parts_adopt_with_row_and_key_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ced.model.encoding.PART_ROWS", 2)
    texts = ["one", "two", "three"]
    missing = tmp_path / "missing.parquet"
    pq.write_table(
        pa.table({"text_sha256": [text_key(text) for text in texts], "text": texts}), missing, compression="zstd"
    )
    report = encode_missing_to_parts(missing, tmp_path / "encoded", RandomTextEncoder(4))
    assert report["rows"] == 3
    assert report["parts"] == 2
    assert len(report["files"]) == 4
    cache = TextEmbeddingCache(tmp_path / "cache", TextEncodingSettings(provider="random", dimension=4))
    assert cache.adopt_parts(tmp_path / "encoded", missing) == 3
    assert cache.manifest()["rows"] == 3
    assert [part["name"] for part in cache.manifest()["parts"]] == [
        "part-000000.safetensors",
        "part-000001.safetensors",
    ]
    connection = duckdb.connect()
    try:
        cache.create_index(connection)
        assert connection.execute("SELECT count(*) FROM cache_index").fetchone() == (3,)
    finally:
        connection.close()
    assert [part.load(tmp_path).shape for part in cache.part_refs()] == [(2, 4), (1, 4)]


def test_modal_adoption_commits_each_validated_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad later part must not roll back an earlier validated Modal part.

    Why: durable per-part publication is what bounds cloud recompute after a
    failure. The sliced retry also proves that adoption compares the incoming
    keys with the requested input range rather than starting at row zero again.
    """
    monkeypatch.setattr("ced.model.encoding.PART_ROWS", 2)
    texts = ["one", "two", "three"]
    missing = tmp_path / "missing.parquet"
    pq.write_table(pa.table({"text_sha256": [text_key(text) for text in texts], "text": texts}), missing)
    encoded = tmp_path / "encoded"
    encode_missing_to_parts(missing, encoded, RandomTextEncoder(4))
    second_index = encoded / "index" / "part-000001.parquet"
    second = pq.read_table(second_index)
    pq.write_table(second.set_column(1, "row", pa.array([99], pa.int64())), second_index)
    cache = TextEmbeddingCache(tmp_path / "cache", TextEncodingSettings(provider="random", dimension=4))

    with pytest.raises(ValueError, match="row mismatch"):
        cache.adopt_parts(encoded, missing)
    assert cache.manifest()["rows"] == 2

    pq.write_table(second.set_column(1, "row", pa.array([2], pa.int64())), second_index)
    assert cache.adopt_parts(encoded, missing, input_row_start=2, rows=1) == 1
    assert cache.manifest()["rows"] == 3
    assert [part.load(tmp_path).shape for part in cache.part_refs()] == [(2, 4), (1, 4)]

    duplicate_missing = tmp_path / "duplicate.parquet"
    pq.write_table(pa.table({"text_sha256": [text_key("one")], "text": ["one"]}), duplicate_missing)
    duplicate = tmp_path / "duplicate-encoded"
    encode_missing_to_parts(duplicate_missing, duplicate, RandomTextEncoder(4), start_row=3, start_part=2)
    with pytest.raises(ValueError, match="already cached"):
        cache.adopt_parts(duplicate, duplicate_missing)
    assert cache.manifest()["rows"] == 3


def test_modal_adoption_rejects_invalid_slices_and_duplicate_keys(tmp_path: Path) -> None:
    """Invalid input bounds and duplicate incoming keys cannot publish vectors.

    Why: a resume descriptor is untrusted orchestration data. Bounds must fail
    before reading artifacts, and duplicate rows must fail before they create an
    ambiguous text-to-vector relation.
    """
    missing = tmp_path / "missing.parquet"
    pq.write_table(pa.table({"text_sha256": [text_key("new"), text_key("new")], "text": ["new", "new"]}), missing)
    cache = TextEmbeddingCache(tmp_path / "cache", TextEncodingSettings(provider="random", dimension=4))
    with pytest.raises(ValueError, match="invalid Modal input slice"):
        cache.adopt_parts(tmp_path / "encoded", missing, input_row_start=-1)
    with pytest.raises(ValueError, match="invalid Modal input slice"):
        cache.adopt_parts(tmp_path / "encoded", missing, rows=-1)
    with pytest.raises(ValueError, match="invalid Modal input slice"):
        cache.adopt_parts(tmp_path / "encoded", missing, input_row_start=1, rows=2)

    encoded = tmp_path / "encoded"
    encode_missing_to_parts(missing, encoded, RandomTextEncoder(4))
    with pytest.raises(ValueError, match="duplicated"):
        cache.adopt_parts(encoded, missing)
    assert cache.manifest()["rows"] == 0


def test_modal_adoption_rejects_non_finite_values(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"
    pq.write_table(pa.table({"text_sha256": [text_key("one")], "text": ["one"]}), missing)
    encoded = tmp_path / "encoded"
    encode_missing_to_parts(missing, encoded, RandomTextEncoder(4))
    vectors = encoded / "vectors" / "part-000000.safetensors"
    write_tensor(vectors, np.array([[float("nan"), 0, 0, 0]], dtype=np.float32))
    cache = TextEmbeddingCache(tmp_path / "cache", TextEncodingSettings(provider="random", dimension=4))
    with pytest.raises(ValueError, match="non-finite"):
        cache.adopt_parts(encoded, missing)
    assert cache.manifest()["rows"] == 0


def test_modal_adoption_rejects_wrong_rows_without_publishing(tmp_path: Path) -> None:
    texts = ["one"]
    missing = tmp_path / "missing.parquet"
    pq.write_table(pa.table({"text_sha256": [text_key("one")], "text": texts}), missing)
    encoded = tmp_path / "encoded"
    encode_missing_to_parts(missing, encoded, RandomTextEncoder(4))
    index = encoded / "index" / "part-000000.parquet"
    table = pq.read_table(index)
    table = table.set_column(1, "row", pa.array([99], pa.int64()))
    pq.write_table(table, index)
    cache = TextEmbeddingCache(tmp_path / "cache", TextEncodingSettings(provider="random", dimension=4))
    with pytest.raises(ValueError, match="row mismatch"):
        cache.adopt_parts(encoded, missing)
    assert cache.manifest()["rows"] == 0
