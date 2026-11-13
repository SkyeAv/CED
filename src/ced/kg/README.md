# Canonical graph

`ced build --source ced` builds offline on wenceslaus. Omit `--source` to select enabled YAML sources; missing payloads report `ced fetch --source <name>` rather than downloading.

Snapshots live in `.ced/snapshots/<hash>/`. Partitioned Parquet is authoritative. `nodes/` merges canonical IDs; `statements/` preserves source/native evidence identity, not only SPO triples. `attributes/` stores every original attribute as typed long events, including structured containers and leaves; statement qualifiers (significance, direction, population context, and any other observed type) join the same event table under the `qualifier.` path prefix, which is how they reach the model's edge feature store. `node_identities/` retains source-native identity records. KGX shards are interoperability exports; use long attribute tables for lossless recovery.

`ontology_terms/` unifies the locked, installed Biolink vocabulary with Amalgam classes, predicates, attributes, taxa, and observed qualifiers. IDs are not restricted to Biolink enums. Explicit YAML predicate overrides are the only semantic remapping. Amalgam int32 keys remain snapshot metadata; canonical entity/taxon CURIEs and binary16 UUIDs determine stable identity. Synonyms, xrefs, and conflict records never merge entities.

Readers iterate bounded batches. DuckDB uses at most four threads, explicit memory limits, and `.ced/` spill directories for joins/sorts. The merge keeps one canonical node's metadata in memory at a time. Extremely high metadata cardinality for a single node is therefore the remaining per-identity memory bound.

Snapshots include manifests, SHA-256 checksums, mapping/predicate/ontology reports, and per-source licenses. Amalgam is noncommercial; its presence does not relicense other sources. Incomplete `.partial` snapshots are not resumed silently: inspect and explicitly repair or remove them before rebuilding.
