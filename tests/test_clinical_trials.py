from __future__ import annotations

import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from ced.acquisition.pipeline import SourceArtifacts
from ced.config.sources import load_sources
from ced.kg.adapters.clinical_trials import ClinicalTrialsAdapter
from ced.kg.merge import merge_graph
from ced.kg.pipeline import CanonicalizationJob, canonicalize_source


def fixture(tmp_path: Path) -> SourceArtifacts:
    source = load_sources(Path("src/ced/config/sources"))["clinical-trials"]
    nodes = tmp_path / "nodes.jsonl.gz"
    edges = tmp_path / "edges.jsonl.gz"
    trial = {
        "id": "CLINICALTRIALS:NCT00000001",
        "name": "Trial title",
        "clinical_trial_enrollment": 0,
        "clinical_trial_child": False,
        "interventions": [{"id": "CHEBI:1", "original_interventions": ["Drug"]}],
        "conditions": [{"id": "MONDO:1", "original_conditions": ["Disease"]}],
    }
    with gzip.open(nodes, "wt") as stream:
        for row in [trial, {"id": "CHEBI:1", "name": "Drug"}, {"id": "MONDO:1", "name": "Disease"}]:
            stream.write(json.dumps(row) + "\n")
    with gzip.open(edges, "wt") as stream:
        stream.write(
            json.dumps(
                {
                    "id": "native-edge",
                    "subject": "CHEBI:1",
                    "predicate": "biolink:in_clinical_trials_for",
                    "object": "MONDO:1",
                    "intervention_boxed_warning": float("nan"),
                    "has_supporting_studies": [trial["id"]],
                    "supporting_study_mapping_sources": ["subject:text"],
                    "provenance_ref": "clinical_trials_kg_provenance_v4.2.6.tsv.gz",
                    "disease_context_qualifier": "MONDO:1",
                    "sources": [{"resource_id": "infores:clinicaltrials-kp"}],
                }
            )
            + "\n"
        )
    return SourceArtifacts(source=source, files={}, members={"nodes": [nodes], "edges": [edges]})


def test_clinical_trials_canonicalization(tmp_path: Path) -> None:
    artifacts = fixture(tmp_path)
    root = tmp_path / "snapshot"
    canonicalize_source(CanonicalizationJob(artifacts, root / "raw", root / "spill", 1, "128MB"))
    counts = merge_graph(root, memory="128MB")
    assert counts["nodes"] == 3
    assert counts["statements"] == 3
    trial = next(row for row in pq.read_table(root / "nodes").to_pylist() if row["id"].startswith("CLINICALTRIALS:"))
    assert trial["categories"] == ["biolink:ClinicalTrial"]
    assert trial["names"] == ["Trial title"]
    statements = pq.read_table(root / "statements").to_pylist()
    association = next(row for row in statements if row["native_id"] == "native-edge")
    assert association["predicate"] == "biolink:in_clinical_trials_for"
    assert association["target_eligible"]
    # `has_supporting_studies` feeds the publication pipeline as bare NCT ids; the
    # original CURIE-prefixed field remains an attribute event (asserted below).
    assert association["publications"] == ["NCT00000001"]
    membership = [row for row in statements if row["native_id"] != "native-edge"]
    assert {row["object"] for row in membership} == {"CHEBI:1", "MONDO:1"}
    assert all(row["relation_role"] == "schema" and not row["target_eligible"] for row in membership)
    events = pq.read_table(root / "attributes").to_pylist()
    assert any(row["path"] == "clinical_trial_enrollment" and row["numeric"] == 0 for row in events)
    assert any(row["path"] == "clinical_trial_child" and row["raw_json"] == "false" for row in events)
    assert any(row["path"] == "intervention_boxed_warning" and row["value_type"] == "nonfinite" for row in events)
    assert any(row["path"] == "has_supporting_studies[0]" and row["text"] == trial["id"] for row in events)
    assert any(row["path"] == "qualifier.disease_context_qualifier" and row["text"] == "MONDO:1" for row in events)
    assert any(row["path"] == "provenance_ref" for row in events)
    first = [record.id for batch in ClinicalTrialsAdapter(artifacts).batches() for record in batch.statements]
    second = [record.id for batch in ClinicalTrialsAdapter(artifacts).batches() for record in batch.statements]
    assert first == second


@pytest.mark.parametrize("content", ["[]", "not json"])
def test_invalid_record_rejected(tmp_path: Path, content: str) -> None:
    artifacts = fixture(tmp_path)
    path = tmp_path / "invalid.jsonl"
    path.write_text("\n" + content + "\n")
    artifacts.members["nodes"] = [path]
    with pytest.raises(ValueError, match=r"record must be an object|Expecting value"):
        list(ClinicalTrialsAdapter(artifacts).batches())


def test_invalid_trial_member_rejected(tmp_path: Path) -> None:
    artifacts = fixture(tmp_path)
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({"id": "CLINICALTRIALS:NCT1", "conditions": [{"id": "bad id"}]}))
    artifacts.members["nodes"] = [path]
    with pytest.raises(ValueError, match="invalid trial member"):
        list(ClinicalTrialsAdapter(artifacts).batches())


@pytest.mark.parametrize("member", ["not-a-dict", {"name": "no id field"}])
def test_malformed_trial_member_shape_rejected(tmp_path: Path, member: object) -> None:
    artifacts = fixture(tmp_path)
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({"id": "CLINICALTRIALS:NCT1", "conditions": [member]}))
    artifacts.members["nodes"] = [path]
    with pytest.raises(ValueError, match="trial member must be an object with an id"):
        list(ClinicalTrialsAdapter(artifacts).batches())


def test_membership_edges_carry_source_provenance(tmp_path: Path) -> None:
    artifacts = fixture(tmp_path)
    root = tmp_path / "snapshot"
    canonicalize_source(CanonicalizationJob(artifacts, root / "raw", root / "spill", 1, "128MB"))
    merge_graph(root, memory="128MB")
    statements = pq.read_table(root / "statements").to_pylist()
    membership = [row for row in statements if row["native_id"] != "native-edge"]
    assert membership
    for row in membership:
        assert json.loads(row["sources_json"]) == [{"resource_id": "infores:clinicaltrials-kp"}]


def test_dangling_trial_member_fails_merge(tmp_path: Path) -> None:
    """A member id absent from the node stream is a hard merge failure, not a silent drop.

    `merge.py` enforces endpoint integrity for every adapter; this documents that the
    clinical-trials membership edges are subject to the same contract.
    """
    source = load_sources(Path("src/ced/config/sources"))["clinical-trials"]
    nodes = tmp_path / "nodes.jsonl.gz"
    with gzip.open(nodes, "wt") as stream:
        stream.write(
            json.dumps(
                {
                    "id": "CLINICALTRIALS:NCT00000002",
                    "name": "Orphan trial",
                    "conditions": [{"id": "MONDO:missing"}],
                }
            )
            + "\n"
        )
    artifacts = SourceArtifacts(source=source, files={}, members={"nodes": [nodes], "edges": []})
    root = tmp_path / "snapshot"
    canonicalize_source(CanonicalizationJob(artifacts, root / "raw", root / "spill", 1, "128MB"))
    with pytest.raises(ValueError, match="statement references missing node: MONDO:missing"):
        merge_graph(root, memory="128MB")


def test_explicit_null_category_still_gets_clinical_trial_type(tmp_path: Path) -> None:
    source = load_sources(Path("src/ced/config/sources"))["clinical-trials"]
    nodes = tmp_path / "nodes.jsonl.gz"
    with gzip.open(nodes, "wt") as stream:
        stream.write(json.dumps({"id": "CLINICALTRIALS:NCT00000003", "name": "Trial", "category": None}) + "\n")
    artifacts = SourceArtifacts(source=source, files={}, members={"nodes": [nodes], "edges": []})
    root = tmp_path / "snapshot"
    canonicalize_source(CanonicalizationJob(artifacts, root / "raw", root / "spill", 1, "128MB"))
    merge_graph(root, memory="128MB")
    trial = next(row for row in pq.read_table(root / "nodes").to_pylist() if row["id"].startswith("CLINICALTRIALS:"))
    assert trial["categories"] == ["biolink:ClinicalTrial"]
