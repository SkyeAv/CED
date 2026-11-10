from __future__ import annotations

import glob
import re
import threading
import unicodedata
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

from ced.config.loader import _expand_env, read_yaml
from ced.config.models import RuntimeConfig
from ced.config.sources import SourceConfig
from ced.preflight import require_cpu_worker, require_fullmap

TABLASSERT_VERSION = "22.3.0"
ADAPTER_VERSION = "1"
_RESERVED = ("ced_original_predicate", "ced_predicate", "ced_source_config", "ced_section_index")
_PATCH_LOCK = threading.RLock()


class CedIdentityFactory:
    def __init__(self, options: dict[str, Any]) -> None:
        self.namespace = str(options.get("unresolved_namespace", "CED"))
        self.family_pattern = re.compile(str(options.get("family_pattern", r"^Family ([0-9]+)$")))
        self.family_template = str(options.get("family_template", "CED:Family_{1}"))
        self.uuid_namespace = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/SkyeAv/CED/local-id")
        self.identities: dict[str, tuple[str, str]] = {}
        self.cache: dict[tuple[str, str], str] = {}

    def normalize(self, value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).strip().split())

    def make(self, value: str, category: str) -> str:
        normalized = self.normalize(value)
        key = (normalized.casefold(), category)
        if key in self.cache:
            return self.cache[key]
        family = self.family_pattern.fullmatch(normalized)
        if family:
            identifier = self.family_template.format(normalized, *family.groups())
        else:
            identifier = f"{self.namespace}:{uuid.uuid5(self.uuid_namespace, chr(31).join(key))}"
        semantic_key = (normalized.casefold(), category)
        previous = self.identities.setdefault(identifier, semantic_key)
        if previous != semantic_key:
            raise ValueError(f"local CED identifier collision: {identifier}")
        self.cache[key] = identifier
        return identifier

    def series(self, category: str):
        def convert(values: pl.Series) -> pl.Series:
            return pl.Series(
                values.name,
                [
                    self.make(str(value), category) if value is not None and str(value).strip() else None
                    for value in values
                ],
                dtype=pl.String,
            )

        return convert


@dataclass
class PreparedSection:
    tcode: Any
    source: dict[str, object]


@dataclass
class TablassertBuild:
    nodes: Path
    edges: Path
    rig: Path
    sections: int


class TablassertAdapter:
    def __init__(self, runtime: RuntimeConfig, source: SourceConfig, fullmap_path: Path | None = None) -> None:
        if version("tablassert") != TABLASSERT_VERSION:
            raise RuntimeError(f"CED adapter requires tablassert=={TABLASSERT_VERSION}")
        self.runtime = runtime
        self.source = source
        self.fullmap_override = fullmap_path
        self.identity = CedIdentityFactory(source.mapping.options)

    def graph(self) -> Any:
        from tablassert.models import Graph

        if self.source.graph is None:
            raise ValueError("Tablassert source requires graph")
        raw = _expand_env(read_yaml(self.source.graph))
        tables: list[str] = []
        for pattern in raw["tables"]:
            matches = sorted(glob.glob(str(pattern), recursive=True))
            if not matches:
                raise ValueError(f"graph table directive matched no YAML: {pattern}")
            tables.extend(matches)
        raw["tables"] = sorted(dict.fromkeys(tables))
        if self.fullmap_override:
            raw["fullmap"] = str(self.fullmap_override)
        output = self.runtime.artifact_path("kg", self.source.name)
        raw["rig"]["artifact_base_path"] = str(output)
        raw["rig"]["artifact_base_url"] = output.as_uri()
        return Graph.model_validate(raw)

    @staticmethod
    def _keyed_merge(stock: Any, a: Any, b: Any) -> Any:
        if isinstance(a, dict) and isinstance(b, dict):
            for key, value in b.items():
                a[key] = TablassertAdapter._keyed_merge(stock, a[key], value) if key in a else value
            return a
        if isinstance(a, list) and isinstance(b, list):
            key = next(
                (
                    candidate
                    for candidate in ("annotation", "qualifier")
                    if a + b and all(isinstance(x, dict) and candidate in x for x in a + b)
                ),
                None,
            )
            if key:
                merged = deepcopy(a)
                positions = {item[key]: index for index, item in enumerate(merged)}
                for item in b:
                    if item[key] not in positions:
                        positions[item[key]] = len(merged)
                        merged.append(deepcopy(item))
                    else:
                        index = positions[item[key]]
                        merged[index] = TablassertAdapter._keyed_merge(stock, merged[index], item)
                return merged
            return stock(a, b)
        return b

    @contextmanager
    def patches(self) -> Iterator[None]:
        from tablassert import fullmap, ingests, lib
        from tablassert.biolink import DISABLED_EDGE_FIELDS
        from tablassert.errors import BiolinkRelocationWarning

        def merge(a: Any, b: Any) -> Any:
            return self._keyed_merge(stock_merge, a, b)

        def resolve(lf: pl.LazyFrame, specs: list[Any], *args: Any, **kwargs: Any) -> pl.LazyFrame:
            relaxed = [spec._replace(nullable=True) for spec in specs]
            result = stock_resolve(lf, relaxed, *args, **kwargs)
            for spec in specs:
                col = spec.col
                raw = f"original_{col}"
                preferred = (spec.prioritize or ["NamedThing"])[0]
                category = str(getattr(preferred, "value", preferred))
                valid = pl.col(raw).is_not_null() & (pl.col(raw).cast(pl.String).str.strip_chars().str.len_chars() > 0)
                unresolved = pl.col(col).is_null() & valid
                result = result.with_columns(
                    pl.when(
                        pl.col(f"{col}_taxon").is_null() & pl.col(col).cast(pl.String).str.starts_with("NCBITaxon:")
                    )
                    .then(pl.col(col))
                    .otherwise(pl.col(f"{col}_taxon"))
                    .alias(f"{col}_taxon"),
                    pl.when(unresolved)
                    .then(pl.col(raw).map_batches(self.identity.series(category), return_dtype=pl.String))
                    .otherwise(pl.col(col))
                    .alias(col),
                    pl.when(unresolved).then(pl.col(raw)).otherwise(pl.col(f"{col}_name")).alias(f"{col}_name"),
                    pl.when(unresolved)
                    .then(pl.lit(category))
                    .otherwise(pl.col(f"{col}_category"))
                    .alias(f"{col}_category"),
                    pl.when(pl.col(col).is_not_null())
                    .then(pl.lit("fullmap"))
                    .when(valid)
                    .then(pl.lit("unresolved_local"))
                    .otherwise(pl.lit("missing"))
                    .alias(f"{col}_mapping_status"),
                )
            required = [spec.col for spec in specs if not spec.nullable]
            return (
                result.filter(pl.all_horizontal([pl.col(col).is_not_null() for col in required]))
                if required
                else result
            )

        def preserve_extensions(lf: pl.LazyFrame) -> pl.LazyFrame:
            disabled = [name for name in DISABLED_EDGE_FIELDS if name in lf.collect_schema()]
            return lf.drop(disabled) if disabled else lf

        with _PATCH_LOCK, warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`ced_(original_predicate|predicate|source_config|section_index)`.*",
                category=BiolinkRelocationWarning,
            )
            stock_merge = ingests.fastmerge
            stock_resolve = fullmap.resolve_batch
            stock_fold = lib.fold_unknown_to_supporting_text
            ingests.fastmerge = merge
            lib.resolve_batch = resolve
            lib.fold_unknown_to_supporting_text = preserve_extensions
            try:
                yield
            finally:
                ingests.fastmerge = stock_merge
                lib.resolve_batch = stock_resolve
                lib.fold_unknown_to_supporting_text = stock_fold

    def _normalize_source(self, value: Any) -> None:
        if not isinstance(value, dict):
            return
        source = value.get("source")
        if isinstance(source, dict):
            if isinstance(source.get("url"), str):
                source["url"] = [source["url"]]
            if source.get("delimiter") == "/t":
                source["delimiter"] = "\t"
        if isinstance(value.get("template"), dict):
            self._normalize_source(value["template"])
        for section in value.get("sections", []):
            self._normalize_source(section)

    def _canonical_predicate(self, predicate: str) -> str:
        from tablassert.biolink import Predicates

        if predicate in self.source.mapping.predicate_overrides:
            return self.source.mapping.predicate_overrides[predicate]
        if predicate in Predicates._value2member_map_:
            return f"biolink:{predicate}"
        if ":" in predicate:
            return predicate
        return f"ced:{re.sub(r'[^A-Za-z0-9_./-]+', '_', predicate)}"

    def sections(
        self, table_paths: list[Path] | None = None, section_limit: int | None = None
    ) -> tuple[Any, list[PreparedSection]]:
        from tablassert.biolink import Predicates
        from tablassert.ingests import from_yaml, to_sections
        from tablassert.lib import Tcode
        from tablassert.utils import file_content_hash, section_store_key

        graph = self.graph()
        prepared: list[PreparedSection] = []
        content_hashes: dict[Path, str] = {}
        with self.patches():
            for table in table_paths or graph.tables:
                instructions = from_yaml(table)
                if not isinstance(instructions, dict):
                    raise TypeError(f"table config must be a mapping: {table}")
                self._normalize_source(instructions)
                for section in to_sections(instructions, table)[:section_limit]:
                    statement = section["statement"]
                    original = statement["predicate"]
                    statement["predicate"] = original if original in Predicates._value2member_map_ else "related_to"
                    section["annotations"] = [
                        annotation
                        for annotation in section.get("annotations", [])
                        if "encoding" in annotation or annotation.get("method") != "value"
                    ]
                    lineage = {
                        "ced_original_predicate": original,
                        "ced_predicate": self._canonical_predicate(original),
                        "ced_source_config": str(table),
                        "ced_section_index": str(section["section_index"]),
                    }
                    section.setdefault("annotations", []).extend(
                        {"annotation": key, "method": "value", "encoding": value} for key, value in lineage.items()
                    )
                    local = Path(section["source"]["local"])
                    if local not in content_hashes:
                        content_hashes[local] = file_content_hash(local, config=table)
                    store = self.runtime.artifact_path(
                        "tablassert",
                        "store",
                        f"{ADAPTER_VERSION}-{section_store_key(section, content_digest=content_hashes[local])}.parquet",
                    )
                    store.parent.mkdir(parents=True, exist_ok=True)
                    tcode = Tcode.model_validate(
                        {
                            **section,
                            "store": store,
                            "name": graph.name,
                            "infores": graph.rig.source_info.infores_id,
                            "qc_similarity_threshold": graph.qc_similarity_threshold,
                            "qc_fuzzy_ratio_threshold": graph.qc_fuzzy_ratio_threshold,
                            "qc_fuzzy_partial_threshold": graph.qc_fuzzy_partial_threshold,
                            "qc_aliases": graph.qc_aliases,
                        }
                    )
                    prepared.append(
                        PreparedSection(tcode, {"local": str(local), "urls": section["source"].get("url") or []})
                    )
        return graph, prepared

    def build(self, table_paths: list[Path] | None = None, section_limit: int | None = None) -> TablassertBuild:
        from tablassert.lib import compile_graph, compile_subgraph

        require_cpu_worker(self.runtime)
        graph, prepared = self.sections(table_paths, section_limit)
        database = require_fullmap(graph.fullmap)
        with self.patches():
            subgraphs: list[Path] = []
            for section in prepared:
                operations = section.tcode.collect(database)
                subgraphs.append(operations if isinstance(operations, Path) else compile_subgraph(operations))
            compile_graph(
                subgraphs,
                graph.name,
                graph.version,
                graph.rig,
                [section.source for section in prepared],
                uuid_fields=graph.uuid_fields,
                uuid_domain=graph.uuid_namespace,
                uuid_on_collision=graph.uuid_on_collision,
            )
        output = Path(graph.rig.artifact_base_path)
        stem = f"{graph.name}_{graph.version}"
        return TablassertBuild(
            output / f"{stem}.nodes.ndjson", output / f"{stem}.edges.ndjson", output / f"{stem}.RIG.yaml", len(prepared)
        )
