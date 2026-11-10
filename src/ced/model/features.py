from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from ced.kg.records import AttributeEvent
from ced.model.schemas import FeatureFieldSpec


@dataclass
class FeatureFit:
    fields: list[FeatureFieldSpec] = field(default_factory=list)
    categories: dict[str, dict[str, int]] = field(default_factory=dict)


def fit_features(
    events: Iterable[AttributeEvent], training_nodes: set[str], training_statements: set[str]
) -> FeatureFit:
    """Fit only owners participating in training; text remains raw and unembedded."""
    statistics: dict[
        tuple[Literal["node", "statement"], str, Literal["numeric", "text", "categorical"]], list[float]
    ] = {}
    categories: dict[str, set[str]] = {}
    for event in events:
        owners = training_nodes if event.owner_kind == "node" else training_statements
        if event.owner_id not in owners or event.value_type in {"object", "list", "null", "nonfinite"}:
            continue
        kind = (
            "numeric"
            if event.numeric is not None
            else "text"
            if event.text is not None and len(event.text) > 128
            else "categorical"
        )
        key = (event.owner_kind, event.path, kind)
        state = statistics.setdefault(key, [0, 0, 0])
        state[0] += 1
        if event.numeric is not None:
            delta = event.numeric - state[1]
            state[1] += delta / state[0]
            state[2] += delta * (event.numeric - state[1])
        elif kind == "categorical":
            categories.setdefault(f"{event.owner_kind}\0{event.path}", set()).add(event.raw_json)
    result = FeatureFit()
    for index, ((owner, path, kind), state) in enumerate(sorted(statistics.items()), 2):
        result.fields.append(
            FeatureFieldSpec(
                index=index,
                path=path,
                owner_kind=owner,
                kind=kind,
                count=int(state[0]),
                mean=state[1],
                scale=max(math.sqrt(state[2] / state[0]), 1e-8),
            )
        )
    result.categories = {
        key: {value: index for index, value in enumerate(sorted(values), 2)} for key, values in categories.items()
    }
    return result
