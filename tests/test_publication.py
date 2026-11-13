from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from ced import publication
from ced.acquisition.base import atomic_write_json, sha256_file
from ced.config.models import RuntimeConfig
from ced.model.schemas import DenseFeatureStore, DensePartRef, FileArtifact, NodeSemanticArtifacts, TensorRef


class FakeApi:
    def __init__(self) -> None:
        self.repos: list[dict[str, Any]] = []
        self.folders: list[dict[str, Any]] = []
        self.files: dict[str, bytes] = {}
        self.remote_files: set[str] = set()

    def create_repo(self, *args: Any, **kwargs: Any) -> None:
        self.repos.append({"args": args, **kwargs})

    def upload_folder(self, **kwargs: Any) -> None:
        self.folders.append(kwargs)
        root = kwargs["folder_path"]
        self.remote_files.update(
            f"{kwargs['path_in_repo']}/{path.relative_to(root)}" for path in root.rglob("*") if path.is_file()
        )

    def upload_file(self, **kwargs: Any) -> None:
        value = kwargs["path_or_fileobj"]
        self.files[kwargs["path_in_repo"]] = value.read_bytes() if isinstance(value, Path) else value
        self.remote_files.add(kwargs["path_in_repo"])

    def list_repo_files(self, *args: Any, **kwargs: Any) -> list[str]:
        return sorted(self.remote_files)


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RuntimeConfig:
    monkeypatch.setattr(publication, "require_cpu_worker", lambda runtime: None)
    return RuntimeConfig(remote_project=tmp_path, artifact_root=Path("artifacts"))


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    client = FakeApi()
    monkeypatch.setattr(publication, "HfApi", lambda: client)
    return client


def write_snapshot(runtime: RuntimeConfig) -> Path:
    root = runtime.artifact_path("snapshots", "snapshot")
    (root / "nodes").mkdir(parents=True)
    pq.write_table(pa.table({"id": ["core", "isolated"]}), root / "nodes/part-00000.parquet")
    (root / "raw/source").mkdir(parents=True)
    (root / "raw/source/staging.parquet").write_bytes(b"not published")
    atomic_write_json(
        root / "manifest.json",
        {
            "snapshot_hash": "snapshot",
            "counts": {"nodes": 2, "statements": 0},
            "sources": [{"license_name": "CC0"}],
        },
    )
    atomic_write_json(root / "mapping-report.json", [{"mapping_status": "source_native", "count": 2}])
    checksums = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    atomic_write_json(root / "checksums.json", checksums)
    return root


def write_dataset(runtime: RuntimeConfig, *, text: bool = False) -> Path:
    root = runtime.artifact_path("datasets", "dataset")
    root.mkdir(parents=True)
    pq.write_table(pa.table({"head": [0], "tail": [1], "split": [0]}), root / "topology.parquet")
    checksums = {"topology.parquet": sha256_file(root / "topology.parquet")}
    if text:
        artifact = root / "text-encoder"
        (artifact / "parts").mkdir(parents=True)
        part = artifact / "parts/part-000000.safetensors"
        part.write_bytes(b"vectors")
        index = artifact / "text-index.parquet"
        pq.write_table(pa.table({"owner_index": [0], "row": [0]}), index)
        prefixes = artifact / "node-prefixes.safetensors"
        categories = artifact / "node-categories.safetensors"
        prefixes.write_bytes(b"prefixes")
        categories.write_bytes(b"categories")
        store = DenseFeatureStore(
            provider="random",
            model="fixture",
            encoder_key="encoder",
            dimension=2,
            rows=1,
            index=FileArtifact(path="text-encoder/text-index.parquet", rows=1, sha256=sha256_file(index)),
            node=NodeSemanticArtifacts(
                prefix_id=TensorRef(
                    path="text-encoder/node-prefixes.safetensors",
                    dtype="int64",
                    shape=[1],
                    sha256=sha256_file(prefixes),
                ),
                category_id=TensorRef(
                    path="text-encoder/node-categories.safetensors",
                    dtype="int64",
                    shape=[1],
                    sha256=sha256_file(categories),
                ),
                category_indptr=TensorRef(
                    path="text-encoder/node-categories.safetensors",
                    dtype="int64",
                    shape=[2],
                    sha256=sha256_file(categories),
                ),
                prefix_vocabulary_size=1,
                category_vocabulary_size=1,
            ),
            parts=[
                DensePartRef(
                    path="text-encoder/parts/part-000000.safetensors",
                    dtype="float32",
                    shape=[1, 2],
                    sha256=sha256_file(part),
                    row_start=0,
                )
            ],
            checksums={
                str(path.relative_to(root)): sha256_file(path) for path in sorted(artifact.rglob("*")) if path.is_file()
            },
        )
        atomic_write_json(root / "text-features.json", store.model_dump(mode="json"))
        checksums["text-features.json"] = sha256_file(root / "text-features.json")
        checksums.update(
            {str(path.relative_to(root)): sha256_file(path) for path in sorted(artifact.rglob("*")) if path.is_file()}
        )
    atomic_write_json(
        root / "manifest.json",
        {
            "dataset_id": "dataset",
            "snapshot_id": "snapshot",
            "entities": 7,
            "statements": 8,
            "pruning": {"pruned_nodes": 3},
            "checksums": checksums,
        },
    )
    return root


def card_config(card: str, name: str) -> dict[str, Any]:
    metadata = yaml.safe_load(card.split("---", 2)[1])
    return next(config for config in metadata["configs"] if config["config_name"] == name)


def test_snapshot_upload_is_complete_post_qc_and_card_is_dynamic(runtime: RuntimeConfig, api: FakeApi) -> None:
    root = write_snapshot(runtime)
    write_dataset(runtime)
    publication.publish_snapshot(runtime, root)

    assert api.repos == [{"args": (publication.REPO_ID,), "repo_type": "dataset", "exist_ok": True}]
    assert len(api.folders) == 1
    upload = api.folders[0]
    assert upload["folder_path"] == root
    assert upload["path_in_repo"] == "graph/snapshot"
    assert "nodes/part-00000.parquet" in upload["allow_patterns"]
    assert not any(path.startswith("raw/") for path in upload["allow_patterns"])
    checksums = json.loads(api.files["graph/snapshot/checksums.json"])
    assert "nodes/part-00000.parquet" in checksums
    assert not any(path.startswith("raw/") for path in checksums)
    card = api.files["README.md"].decode()
    assert "| `snapshot` | 2 | 0 | CC0 |" in card
    assert "including isolated trees" in card
    assert card_config(card, "graph-nodes") == {
        "config_name": "graph-nodes",
        "data_files": [{"split": "graph", "path": "graph/*/nodes/*.parquet"}],
        "default": True,
    }
    assert "config_name: graph-statements" not in card
    assert "config_name: encoded-topology" not in card
    assert "| `dataset`" not in card


def test_encoded_upload_preserves_pruning_and_portable_text_encoding(runtime: RuntimeConfig, api: FakeApi) -> None:
    root = write_dataset(runtime, text=True)
    publication.publish_dataset(runtime, root)

    assert [upload["path_in_repo"] for upload in api.folders] == ["encoded/dataset"]
    assert "encoded/dataset/text-encoder/text-index.parquet" in api.remote_files
    assert "encoded/dataset/text-encoder/parts/part-000000.safetensors" in api.remote_files
    document = json.loads(api.files["encoded/dataset/text-features.json"])
    assert document["index"]["path"] == "text-encoder/text-index.parquet"
    assert document["parts"][0]["path"] == "text-encoder/parts/part-000000.safetensors"
    assert document["checksums"] == {
        "text-encoder/node-categories.safetensors": document["node"]["category_id"]["sha256"],
        "text-encoder/node-prefixes.safetensors": document["node"]["prefix_id"]["sha256"],
        "text-encoder/parts/part-000000.safetensors": document["parts"][0]["sha256"],
        "text-encoder/text-index.parquet": document["index"]["sha256"],
    }
    assert str(root) not in json.dumps(document)
    assert api.files["encoded/dataset/text-encoder.json"] == api.files["encoded/dataset/text-features.json"]
    card = api.files["README.md"].decode()
    assert "| `dataset` | `snapshot` | 7 | 8 | 3 | encoder |" in card
    assert card_config(card, "encoded-topology")["data_files"] == [
        {"split": "encoded", "path": "encoded/*/topology.parquet"}
    ]
    assert "text-encoder/text-index.parquet" in api.folders[0]["allow_patterns"]
    assert card_config(card, "encoded-text-index")["data_files"] == [
        {"split": "encoded", "path": "encoded/*/text-*/text-index.parquet"}
    ]


def test_reencoding_keeps_previous_encoder_index_and_manifest(runtime: RuntimeConfig, api: FakeApi) -> None:
    root = write_dataset(runtime, text=True)
    publication.publish_dataset(runtime, root)
    previous_manifest = api.files["encoded/dataset/text-encoder.json"]
    document = json.loads((root / "text-features.json").read_text())
    artifact = root / "text-second"
    (artifact / "parts").mkdir(parents=True)
    (artifact / "parts/part-000000.safetensors").write_bytes(
        (root / "text-encoder/parts/part-000000.safetensors").read_bytes()
    )
    index = artifact / "text-index.parquet"
    pq.write_table(pa.table({"owner_index": [0], "row": [5]}), index)
    document["encoder_key"] = "second"
    document["index"] = {"path": "text-second/text-index.parquet", "rows": 1, "sha256": sha256_file(index)}
    document["parts"][0] = {**document["parts"][0], "path": "text-second/parts/part-000000.safetensors"}
    document["checksums"] = {
        str(path.relative_to(root)): sha256_file(path) for path in sorted(artifact.rglob("*")) if path.is_file()
    }
    atomic_write_json(root / "text-features.json", document)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["checksums"].update(document["checksums"])
    manifest["checksums"]["text-features.json"] = sha256_file(root / "text-features.json")
    atomic_write_json(root / "manifest.json", manifest)

    publication.publish_dataset(runtime, root)

    assert api.files["encoded/dataset/text-encoder.json"] == previous_manifest
    assert "encoded/dataset/text-encoder/text-index.parquet" in api.remote_files
    assert "encoded/dataset/text-second/text-index.parquet" in api.remote_files
    assert api.files["encoded/dataset/text-second.json"] != previous_manifest
    assert "encoder, second" in api.files["README.md"].decode()


def test_encoded_upload_without_text_and_incomplete_manifests(runtime: RuntimeConfig, api: FakeApi) -> None:
    root = write_dataset(runtime)
    partial = runtime.artifact_path("datasets", "broken.partial")
    partial.mkdir(parents=True)
    atomic_write_json(partial / "manifest.json", {"dataset_id": "broken"})
    publication.publish_dataset(runtime, root)

    assert len(api.folders) == 1
    assert "encoded/dataset/text-features.json" not in api.files
    assert "broken" not in api.files["README.md"].decode()


@pytest.mark.parametrize("fault", ["index", "path", "duplicate", "suffix"])
def test_invalid_text_store_rejected_before_network(runtime: RuntimeConfig, api: FakeApi, fault: str) -> None:
    root = write_dataset(runtime, text=True)
    path = root / "text-features.json"
    document = json.loads(path.read_text())
    if fault == "index":
        (root / document["index"]["path"]).write_bytes(b"tampered")
    elif fault == "path":
        document["index"]["path"] = "/outside/text-index.parquet"
    elif fault == "duplicate":
        document["parts"].append(document["parts"][0])
    else:
        document["parts"][0]["path"] += ".bin"
    atomic_write_json(path, document)
    with pytest.raises(ValueError, match=r"checksum mismatch|encoder-scoped|unique safetensors"):
        publication.publish_dataset(runtime, root)
    assert api.repos == []


def test_invalid_artifacts_fail_before_network(runtime: RuntimeConfig, api: FakeApi) -> None:
    snapshot = write_snapshot(runtime)
    (snapshot / "nodes/part-00000.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        publication.publish_snapshot(runtime, snapshot)

    dataset = write_dataset(runtime, text=True)
    part = dataset / "text-encoder/parts/part-000000.safetensors"
    part.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        publication.publish_dataset(runtime, dataset)
    assert api.repos == []
    assert api.folders == []


def test_unresolved_license_blocks_snapshot_publish(runtime: RuntimeConfig, api: FakeApi) -> None:
    """A source integrated before its redistribution terms are confirmed must not publish.

    `clinical-trials.yaml` records `license_name: Not specified by producer; ...` until terms
    are confirmed. `publish_snapshot` refuses rather than relying on a README reminder, since
    a bare `ced build` selects every enabled source and uploads by default.
    """
    root = write_snapshot(runtime)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["sources"] = [
        {"source_id": "chembl", "license_name": "CC0"},
        {"source_id": "clinical-trials", "license_name": "Not specified by producer; confirm redistribution terms"},
    ]
    atomic_write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="unresolved source licenses: clinical-trials"):
        publication.publish_snapshot(runtime, root)
    assert api.repos == []
    assert api.folders == []
