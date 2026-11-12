"""Negative and compatibility contracts for publication features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_prepare_topology import STATEMENTS, prepare
from test_publications import _fixture_embeddings, _mock_embedding_download

from ced.model.publications import encode_dataset_publications
from ced.model.training import config_hash


def test_config_hash_accepts_null_publications_and_tracks_feature_identity() -> None:
    document = {"dataset_id": "fixture", "publications": None}
    options = {"learning_rate": 0.001, "seed": 7, "auxiliary_weight": 0.1}
    plain = config_hash(document, **options)
    assert plain == config_hash({"dataset_id": "fixture"}, **options)
    assert plain != config_hash(document | {"publications": {"feature_key": "encoded"}}, **options)


def test_no_publications_produce_zero_features(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(STATEMENTS, "publications", [[] for _ in range(6)])
    _snapshot, dataset, runtime = prepare(tmp_path, monkeypatch)
    source = tmp_path / "empty-evidence.parquet"
    _fixture_embeddings(runtime, source, ["PMID1"], [[1, 0]], 2)
    _mock_embedding_download(monkeypatch, source)
    store = encode_dataset_publications(runtime, dataset.name)
    assert store.statement_publications == store.matched_publications == 0
    assert store.statements_with_publications == store.statements_with_embeddings == 0
    np.testing.assert_array_equal(store.vectors.load(dataset), np.zeros((6, 2), dtype=np.float32))
    np.testing.assert_array_equal(store.evidence.load(dataset), np.zeros(6, dtype=np.float32))


def test_duplicate_embedding_ids_fail_without_publishing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(STATEMENTS, "publications", [["PMID1"], *([[]] * 5)])
    _snapshot, dataset, runtime = prepare(tmp_path, monkeypatch)
    before = (dataset / "manifest.json").read_bytes()
    source = tmp_path / "duplicates.parquet"
    _fixture_embeddings(runtime, source, ["PMID1", "PMID1"], [[1, 0], [0, 1]], 2)
    _mock_embedding_download(monkeypatch, source)
    with pytest.raises(ValueError, match="not unique"):
        encode_dataset_publications(runtime, dataset.name)
    assert (dataset / "manifest.json").read_bytes() == before
    assert not list(dataset.glob("*.partial*"))


def test_source_checksum_mismatch_fails_before_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _snapshot, dataset, runtime = prepare(tmp_path, monkeypatch)
    source = tmp_path / "bad-checksum.parquet"
    _fixture_embeddings(runtime, source, ["PMID1"], [[1, 0]], 2)
    runtime.publication_embeddings.source_sha256 = "0" * 64
    _mock_embedding_download(monkeypatch, source)
    with pytest.raises(ValueError, match="checksum mismatch"):
        encode_dataset_publications(runtime, dataset.name)
    assert json.loads((dataset / "manifest.json").read_text())["publications"] is None
