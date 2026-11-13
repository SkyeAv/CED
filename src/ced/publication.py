from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi

from ced.config.models import RuntimeConfig
from ced.kg.audit import validate_checksums
from ced.model.schemas import DenseFeatureStore
from ced.preflight import require_cpu_worker

REPO_ID = "SkyeAv/ced"
GRAPH_TABLES = (
    "nodes",
    "statements",
    "attributes",
    "node_identities",
    "predicates",
    "ontology_terms",
    "conflicts",
)
ENCODED_TABLES = (
    "entities",
    "relations",
    "types",
    "topology",
    "fitted_fields",
    "fitted_categories",
    "node-features",
    "statement-features",
    "text-index",
)


def _config(split: str, table: str, pattern: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "config_name": f"{split}-{table}",
        "data_files": [{"split": split, "path": pattern}],
    }
    if split == "graph" and table == "nodes":
        config["default"] = True
    return config


def _manifest_paths(runtime: RuntimeConfig, kind: str) -> list[Path]:
    return [
        path
        for path in sorted(runtime.artifact_path(kind).glob("*/manifest.json"))
        if not path.parent.name.endswith(".partial")
    ]


def dataset_card(runtime: RuntimeConfig, files: set[str]) -> str:
    """Generate the card from local manifests whose artifacts are actually present on the Hub."""
    configs = []
    for split, tables in (("graph", GRAPH_TABLES), ("encoded", ENCODED_TABLES)):
        for table in tables:
            suffix = f"{table}/*.parquet" if split == "graph" else f"{table}.parquet"
            if table == "text-index":
                suffix = "text-*/text-index.parquet"
            pattern = f"{split}/*/{suffix}"
            if any(fnmatch(path, pattern) for path in files):
                configs.append(_config(split, table, pattern))
    metadata = {
        "pretty_name": "CED knowledge graph",
        "license": "other",
        "tags": ["biomedical", "knowledge-graph", "parquet"],
        "configs": configs,
    }
    lines = [
        "---",
        yaml.safe_dump(metadata, sort_keys=False).strip(),
        "---",
        "",
        "# CED",
        "",
        "QC-ed canonical graph snapshots and model-ready encodings. The `graph` split retains every canonical "
        "node, including isolated trees. The `encoded` split contains post-preprocessing Parquet topology and "
        "features; safetensors preserve model-ready tensors and text embeddings. Raw staging and spill files "
        "are not published.",
        "",
        "Source-specific licenses and restrictions remain in force; see each graph's license-report.json. "
        "This compilation does not relicense its sources and is not for clinical decision-making.",
        "",
        "```python",
        "from datasets import load_dataset",
        "",
        f'nodes = load_dataset("{REPO_ID}", "graph-nodes", split="graph")',
        f'topology = load_dataset("{REPO_ID}", "encoded-topology", split="encoded")',
        "```",
        "",
        "## Graph snapshots",
        "",
        "| Snapshot | Nodes | Statements | Source licenses |",
        "| --- | ---: | ---: | --- |",
    ]
    for path in _manifest_paths(runtime, "snapshots"):
        if f"graph/{path.parent.name}/manifest.json" not in files:
            continue
        manifest = json.loads(path.read_text())
        counts = manifest["counts"]
        licenses = ", ".join(sorted({source["license_name"] for source in manifest["sources"]}))
        lines.append(f"| `{manifest['snapshot_hash']}` | {counts['nodes']} | {counts['statements']} | {licenses} |")
    lines += [
        "",
        "## Encoded datasets",
        "",
        "| Dataset | Snapshot | Entities | Statements | Pruned isolated nodes | Text encoder |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for path in _manifest_paths(runtime, "datasets"):
        if f"encoded/{path.parent.name}/manifest.json" not in files:
            continue
        manifest = json.loads(path.read_text())
        prefix = f"encoded/{path.parent.name}/"
        encoders = ", ".join(
            sorted(
                Path(remote).parent.name.removeprefix("text-")
                for remote in files
                if remote.startswith(prefix) and remote.endswith("/text-index.parquet") and remote.count("/") == 3
            )
        )
        pruning = manifest.get("pruning") or {}
        lines.append(
            f"| `{manifest['dataset_id']}` | `{manifest['snapshot_id']}` | {manifest['entities']} | "
            f"{manifest['statements']} | {pruning.get('pruned_nodes', 0)} | {encoders or 'none'} |"
        )
    return "\n".join(lines) + "\n"


def _client(runtime: RuntimeConfig) -> HfApi:
    require_cpu_worker(runtime)
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    return api


def _card(api: HfApi, runtime: RuntimeConfig, message: str) -> None:
    files = set(api.list_repo_files(REPO_ID, repo_type="dataset"))
    api.upload_file(
        path_or_fileobj=dataset_card(runtime, files).encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=message,
    )


def publish_snapshot(runtime: RuntimeConfig, root: Path) -> None:
    """Publish the complete post-QC graph, retaining isolated trees and excluding only staging data."""
    checksums = json.loads((root / "checksums.json").read_text())
    published_checksums = {
        relative: digest for relative, digest in checksums.items() if not relative.startswith(("raw/", "spill/"))
    }
    validate_checksums(root, published_checksums)
    api = _client(runtime)
    api.upload_folder(
        folder_path=root,
        path_in_repo=f"graph/{root.name}",
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=list(published_checksums),
        commit_message=f"Publish graph snapshot {root.name}",
    )
    api.upload_file(
        path_or_fileobj=json.dumps(published_checksums, indent=2, sort_keys=True).encode(),
        path_in_repo=f"graph/{root.name}/checksums.json",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Publish portable checksums for graph {root.name}",
    )
    _card(api, runtime, f"Update dataset card for graph {root.name}")


def _stores(root: Path) -> list[DenseFeatureStore]:
    path = root / "text-features.json"
    if not path.exists():
        return []
    return [DenseFeatureStore.model_validate(json.loads(path.read_text()))]


def _validate_store(root: Path, store: DenseFeatureStore) -> None:
    artifact = Path(f"text-{store.encoder_key}")
    if store.index.path != str(artifact / "text-index.parquet") or store.index.path not in store.checksums:
        raise ValueError("text index must have an encoder-scoped path and recorded checksum; rerun encode-text")
    validate_checksums(root, store.checksums)
    names = [Path(part.path).name for part in store.parts]
    if len(set(names)) != len(names) or any(
        Path(part.path).parent != artifact / "parts" or Path(part.path).suffix != ".safetensors" for part in store.parts
    ):
        raise ValueError("text tensor references must be unique safetensors inside the text artifact")


def publish_dataset(runtime: RuntimeConfig, root: Path) -> None:
    """Publish prepared tensors and Parquet, plus a portable manifest for each text encoding."""
    manifest = json.loads((root / "manifest.json").read_text())
    validate_checksums(root, manifest["checksums"])
    stores = _stores(root)
    for store in stores:
        _validate_store(root, store)
    api = _client(runtime)
    destination = f"encoded/{root.name}"
    api.upload_folder(
        folder_path=root,
        path_in_repo=destination,
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["manifest.json", *manifest["checksums"]],
        commit_message=f"Publish encoded dataset {root.name}",
    )
    for store in stores:
        payload = (root / "text-features.json").read_bytes()
        for path in ("text-features.json", f"text-{store.encoder_key}.json"):
            api.upload_file(
                path_or_fileobj=payload,
                path_in_repo=f"{destination}/{path}",
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Publish portable text manifest {store.encoder_key}",
            )
    _card(api, runtime, f"Update dataset card for encoded dataset {root.name}")
