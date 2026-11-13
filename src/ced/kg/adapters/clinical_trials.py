"""ISB clinical-trials JSONL compatibility, without inferred treatment claims."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ced.kg.adapters.kgx_ndjson import KgxNdjsonAdapter
from ced.kg.base import batches

TRIAL_PREFIX = "CLINICALTRIALS:NCT"
TRIAL_CURIE = "CLINICALTRIALS:"
TRIAL_RELATIONS = {
    "interventions": "CED:trial_has_intervention",
    "conditions": "CED:trial_has_condition",
}


class ClinicalTrialsAdapter(KgxNdjsonAdapter):
    def chunks(self, role: str) -> Iterator[list[tuple[Path, int, dict[str, Any]]]]:
        def records() -> Iterator[tuple[Path, int, dict[str, Any]]]:
            for path in self.artifacts.paths(role):
                opener = gzip.open if path.suffix == ".gz" else open
                with opener(path, "rb") as stream:
                    for line, content in enumerate(stream, 1):
                        if not content.strip():
                            continue
                        # The producer emits bare NaN. stdlib JSON accepts it; canonical
                        # attribute events retain it as nonfinite, never a numeric zero.
                        raw = json.loads(content)
                        if not isinstance(raw, dict):
                            raise ValueError(f"clinical trial record must be an object: {path}:{line}")
                        if role == "nodes" and str(raw.get("id", "")).startswith(TRIAL_PREFIX):
                            # `or` guards against an explicit null/empty category silently
                            # dropping the injected type; a producer category always wins.
                            raw = {**raw, "category": raw.get("category") or ["biolink:ClinicalTrial"]}
                        if role == "edges":
                            # `has_supporting_studies` carries `CLINICALTRIALS:NCT...` node ids.
                            # Strip the dataset-local CURIE prefix (keeping `NCT...`) so ids feed
                            # the shared publication pipeline (`normalize_publication_id`, MedCPT
                            # embeddings). The original field stays untouched as an attribute.
                            studies = raw.get("has_supporting_studies")
                            if studies is not None and "publications" not in raw:
                                raw = {
                                    **raw,
                                    "publications": [
                                        str(study).removeprefix(TRIAL_CURIE) for study in studies if study is not None
                                    ],
                                }
                        yield path, line, raw
            if role == "edges":
                # Second deliberate pass over nodes: membership edges are derived from the
                # trial nodes themselves, which are only fully known after the node stream
                # is consumed. A single interleaved pass would reorder producer edges
                # relative to synthesized ones and break determinism across runs.
                for chunk in self.chunks("nodes"):
                    for path, line, raw in chunk:
                        if not str(raw["id"]).startswith(TRIAL_PREFIX):
                            continue
                        for field, predicate in TRIAL_RELATIONS.items():
                            for member in raw.get(field, []):
                                if not isinstance(member, dict) or "id" not in member:
                                    raise ValueError(f"trial member must be an object with an id: {path}:{line}")
                                identifier = member["id"]
                                if (
                                    not isinstance(identifier, str)
                                    or not identifier
                                    or any(character.isspace() for character in identifier)
                                ):
                                    raise ValueError(f"invalid trial member ID: {path}:{line}")
                                yield (
                                    path,
                                    line,
                                    {
                                        "id": f"{raw['id']}|{predicate}|{identifier}",
                                        "subject": raw["id"],
                                        "predicate": predicate,
                                        "object": identifier,
                                        "trial_membership": member,
                                        "sources": [{"resource_id": self.source.provenance.provided_by}],
                                    },
                                )

        return batches(records(), self.size, lambda chunk: chunk)
