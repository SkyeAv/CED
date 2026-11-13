# CED

CED builds an extensible biomedical knowledge graph from declarative sources and trains a relational graph autoencoder for ordinary held-out link prediction. It preserves arbitrary source predicates and attributes, mixes Biolink and Amalgam terms verbatim, mints explicit `CED:` identifiers only for configured unresolved CED entities, and separates CPU preparation on `wenceslaus` from GPU execution on Modal.

## Safety boundary

- The local workstation edits, reads, syncs, and orchestrates. It does not run tests, builds, downloads, or CPU-heavy Python.
- `wenceslaus` runs all hermetic tests, lint/type checks, fetches, builds, canonicalization, tensor preparation, and CPU evaluation with at most four workers.
- Modal runs GPU training and inference only. It never downloads sources, builds the graph, prepares tensors, or performs CPU preprocessing.
- Routine verification uses synthetic fixtures and mocks. Large downloads, full builds, dataset preparation, Modal smoke jobs, and paid GPU runs require explicit approval.
- `ced fetch` is the only source acquisition command. Publication embeddings may be downloaded only by an explicit `ced encode-publications --download --yes`; cached artifacts are the default and builds never download implicitly.
- Never download fullmap. Use the existing read-only database through `CED_FULLMAP` or `--fullmap-path`.

## Setup

Copy `.envrc.example` to `.envrc`, adjust machine-specific values, and run `direnv allow`. direnv exports these variables into every shell in the checkout; remote commands also run under `direnv exec`, so run `direnv allow` once in the wenceslaus checkout (and again whenever its `.envrc` changes). Do not commit or print credentials.

```sh
export PATH=/home/sgoetz/.local/bin:$PATH
ced sync
ced check
```

`ced sync` rsyncs the local checkout to `CED_REMOTE_PROJECT` and installs the frozen uv environment. It excludes `.git`, `.venv`, `.ced`, `.envrc`, `.tablassert`, `plans`, and caches, so each machine keeps its own untracked `.envrc`. Run `uv run prek install` in each wenceslaus Git checkout after hook changes to install or refresh both hook stages.

Important environment variables:

| Variable | Purpose |
| --- | --- |
| `CED_REMOTE_HOST` | SSH host for CPU work; default `wenceslaus` |
| `CED_REMOTE_PROJECT` | Isolated remote checkout |
| `CED_ARTIFACT_ROOT` | Gitignored artifact directory, relative to the remote project |
| `CED_FULLMAP` | Existing fullmap database path |
| `CED_CPU_WORKERS` | Hard CPU worker cap; must remain `4` |
| `CED_DOWNLOAD_JOBS` | Artifacts downloaded concurrently by `ced fetch`; `1`-`4` |
| `CED_TEXT_GPU` | Modal text inference GPU; default `L4` |
| `CED_DOWNLOAD_CONNECTIONS` | Connections per file for `aria2`, and the segment cap for `http`; `1`-`32` |
| `CED_MEMORY_LIMIT` | Process memory budget for canonicalization, merge, and reports |
| `CED_DISK_RESERVE_GB` | Minimum free disk before downloads/builds |
| `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` | Modal credentials, only in the untracked wenceslaus `.envrc` or environment |
| `HF_TOKEN` | Hugging Face write token for `SkyeAv/ced`, only in the untracked wenceslaus `.envrc` or environment |

## Commands

```sh
ced config validate
ced sources list
ced fetch --source chembl --yes
ced fetch --group translator
ced build --source ced --source amalgam --fullmap-path /home/sgoetz/Desktop/fullmap
ced build --group translator
ced prepare <snapshot-id>
ced encode-text <dataset-id> --provider random
# Optional: encode statement attributes (e.g. effect size/type) as per-edge model inputs
# through the configured text encoder for categorical values and z-scored numerics.
ced encode-attributes <dataset-id> --provider random
ced encode-publications <dataset-id> --download --yes
ced train <dataset-id> --epochs 1 --seed 20261110 --yes
ced train <dataset-id> --resume <run-id> --yes
ced evaluate <run-id> --split test --ranking exact --yes
ced publish --snapshot <snapshot-id>
ced publish --dataset <dataset-id>
```

`ced check` runs the full remote gate: `uv lock --check`, Ruff, Pyright, pytest, and strict YAML lint. The gate runs pytest with four xdist workers and one thread per worker, excludes `network`, `gpu`, `fullmap`, and `perf` tests by default, and fails below 90% coverage of `src/ced` (`pytest-cov`, configured in `pyproject.toml`); use `uv run pytest -n 0` remotely for serial debugging. The prek pre-push hook runs this gate on wenceslaus. On the laptop, run `ced check` first and use `git push --no-verify` so no check executes locally.

## Acquisition and concurrency

Sources declare one of four backends: `local`, `huggingface`, `aria2`, and `http`. The `http` backend is dependency-free and segmented: it probes with `Range`, splits each artifact into `segment_bytes` chunks (default 16 MiB) up to `CED_DOWNLOAD_CONNECTIONS`, retries transient failures with exponential backoff, resumes an interrupted partial, and verifies the declared size and SHA-256 before an atomic rename into the store. Pre-signed query strings never reach logs, exceptions, or manifests. `src/ced/acquisition/README.md` documents the boundary in full.

`ced build` canonicalizes each selected source in its own worker process, writing `raw/<source>/` partitions under the snapshot staging directory, with the biolink vocabulary in `raw/_ontology`. `CED_MEMORY_LIMIT` is divided across those workers so the pool stays inside the configured ceiling. The serial merge and report stages give DuckDB 75% of that budget, leaving 25% for Python, Arrow, and allocations outside DuckDB's buffer manager. Merge reads the sorted union of every partition, so counts and merged output do not depend on the worker count. A source that fails canonicalization raises an error naming that source, and no snapshot is promoted.

The 13 Translator KGX releases are declared as the `translator` group (623 MiB total). Fetch them, then build them:

```sh
ced fetch --group translator
ced build --group translator
```

Every artifact in those sources is pinned to a concrete release directory with the exact byte size reported by the release host, so `/latest/` can never move underneath a build. Groups are declarative (`groups:` in the source YAML); an unknown group name fails loudly instead of selecting nothing. `--source`, `--group`, and `--all-enabled` are mutually exclusive, `fetch` requires exactly one of them, and a selection above 1 GiB needs `--yes`.

## Artifact layout

Generated data stays under `.ced/` on the remote worker:

```text
.ced/
  downloads/         # fetched archives and member stores
  snapshots/         # canonical Parquet/KGX source snapshots and reports
  graphs/            # merged canonical graph snapshots
  datasets/          # split tensors, feature stores, batch shards, manifests
  runs/              # Modal checkpoints, metrics, progress, and recovered artifacts
  text-embeddings/   # append-only Sentence Transformers caches: parts/ vectors plus index/ sha->row maps
```

Every snapshot, graph, and dataset writes checksums. Preparation and Modal planning validate them before use.

`scripts/bench_prepare.py` measures preparation on a synthetic snapshot (no download, no fullmap) and reports wall seconds plus peak RSS. It runs only on `wenceslaus` and is covered by `pytest -k bench`.

## Adding a source

Adding a compatible KGX, Translator-release, Hugging Face, direct-URL, archive, or local source should require YAML only.

1. Create `src/ced/config/sources/<name>.yaml`.
2. Pin the exact revision, release, file, archive member, checksum, size, license, and provenance.
3. Select the existing adapter (`tablassert`, `kgx_ndjson`, or `amalgam_compact`) or add a narrow adapter only for a genuinely new canonical format.
4. Preserve source predicates and attributes. Do not force Amalgam terms into Biolink or fabricate ontology mappings.
5. Run `ced config validate` and `ced sources list` remotely.
6. Fetch only the new source explicitly after approval: `ced fetch --source <name> --yes`.
7. Build or prepare only when the user explicitly approves that operation.

Source YAML owns URLs, artifact selection, field mappings, licenses, and source-specific semantics. Python owns reusable acquisition, canonical records, merging, tensors, and models.

## Hugging Face publication

`ced build`, `ced prepare`, and `ced encode-text` publish to `SkyeAv/ced` by default; pass `--no-upload` to keep artifacts local. Build uploads the complete post-QC snapshot to `graph/<snapshot-id>`, including isolated trees, canonical tables, KGX exports, provenance, checksums, and reports, while excluding raw staging and spill files. Preparation and text encoding upload to `encoded/<dataset-id>`, including the isolated-tree pruning report, Parquet topology/features, safetensors, batch shards, and the portable `text-<encoder>/` artifact, whose manifest is retained per encoder as an immutable copy.

`src/ced/publication.py` regenerates `README.md` from completed local manifests on every upload. Its YAML maps the Parquet tables to `graph` and `encoded` splits, for example `load_dataset("SkyeAv/ced", "graph-nodes", split="graph")` and `load_dataset("SkyeAv/ced", "encoded-topology", split="encoded")`. Use `ced publish --snapshot <snapshot-id>` or `ced publish --dataset <dataset-id>` to resume a failed upload without rebuilding. Tests mock the Hub and never upload artifacts.

## Modal workflow

Modal jobs use an ephemeral anonymous volume, not a persistent named dataset volume. The CPU worker:

1. Validates the dataset manifest and checksums.
2. Creates a run directory under `.ced/runs/`.
3. Plans bounded batch groups from `modal.max_batch_group_bytes`.
4. Requires interactive `RUN` confirmation or explicit `--yes`.
5. Uploads shared topology/features once, then one bounded batch group at a time.
6. Runs GPU training or inference in a single-use container.
7. Downloads checkpoints, metrics, and progress after each group.
8. Removes uploaded batch files and remote outputs, then destroys the ephemeral volume in `finally`.

Resume works by copying `checkpoint.pt`, `checkpoint.json`, and `progress.json` into a new run directory. The planner skips completed batches and passes only the remaining shard cursor to Modal.

Evaluation supports exact filtered ranking and explicitly labeled sampled ranking:

```sh
ced evaluate <run-id> --ranking sampled --yes
```

Use sampled ranking for very large candidate spaces; exact ranking remains the default for CED-scale evaluation.

## Text encoding

Sentence Transformers support is optional and centralized in `src/ced/model/encoding.py`. The default provider is `none`; tests use deterministic random encoders or mocked models and never download weights. Configure `text_encoding.provider=sentence_transformers` and a locally available model before running `ced encode-text` or `ced encode-attributes`. The two commands share one append-only embedding cache keyed by `text_sha256(text)`; `encode-attributes` additionally materializes the statement-attribute tensors consumed by `StatementAttributeProvider` in `src/ced/model/statement_features.py`.

Tensor artifacts use dataset schema 2. After upgrading, run `ced prepare` and then `ced encode-text` again; Parquet-backed datasets and embedding caches are not migrated or deleted automatically.

### Modal ensemble and semantic nodes

`ced encode-text <dataset-id> --provider modal_ensemble --yes` runs one encoding call on `CED_TEXT_GPU` (`L4` by default), independent of training's `L40S`. Models load once with BF16, SDPA, and length-grouped dynamic padding. Each block is L2-normalized before concatenation:

- `RenaissanceComputing/SapBERT-PubMedBERT-BabelSynonyms-v2025-01-23`: CLS, 768 dimensions.
- `google/embeddinggemma-2`: text-only documents, 768 dimensions.
- `Qwen/Qwen3-Embedding-0.6B`: documents without a query prompt, 1024 dimensions.

The 2560-dimensional vectors cover names, class labels, predicates, qualifiers, and long attributes. All missing unique texts share one Modal call and the append-only cache. Returned immutable safetensors parts and Parquet indices are validated before publication. Dataset manifests include all referenced artifacts for training and evaluation.

Six node tokens - name text, BERT class text, learned CURIE prefix, learned category, topology, and training-edge-only `log1p` degree - pass through attention and a residual 32-dimensional bottleneck autoencoder. Identical class labels share cached vectors. The node head composes with the predicate/qualifier autoencoder above, preserving learned direction features and statement reconstruction loss. Heads process at most 4096 owners per chunk.

Prepared schema-2 topology remains reusable; run `ced encode-text` again to create dense schema-1 semantic node artifacts. Legacy sidecar-only dense artifacts require re-encoding.

## Publication encoding

`ced encode-publications` attaches precomputed document embeddings to prepared statements without changing graph topology or splits. The default artifact is the pinned private `glusman-team/medcpt-ae` Parquet at revision `3c923644c682a9664f1409ce07ae4e78b622bcf5`: 128-dimensional, L2-normalized vectors for `PMID`, `PMC`, and `NCT` identifiers. The command uses the local Hugging Face cache by default. `--download --yes` explicitly permits the 20,449,885,518-byte download on wenceslaus and verifies the pinned size and SHA-256 before use.

For each statement, CED mean-pools resolved publication vectors and computes `log1p` of the distinct cited-publication count as evidence strength. Missing IDs and unsupported sources, including DailyMed, contribute a zero vector while retaining their citation in evidence strength. A bias-free learned projection adds these features to the existing relation vector, so statements with no publication evidence retain the original statement-feature path. Publication tensors are non-persistent model buffers: they move to the training device but are not duplicated into checkpoints. Modal transfers them as shared dataset artifacts for training and evaluation.

Publication features change the training configuration hash. Prepare and encode the dataset before starting a new run; checkpoints created before publication encoding are not resumed into the new model contract.

## Development

Keep code typed, concise, Ruff-formatted, and Pyright-clean. Use PyTorch for model and matrix math. NumPy and Arrow handle in-memory arrays and DuckDB handoff, with conversion to Torch tensors before computation. Persist topology, feature, batch, and embedding tensors as `.safetensors` files read through `TensorRef`; reserve Parquet for tabular KG data, vocabularies, and indexes. Preparation streams DuckDB results and never iterates per statement. Tests must remain offline and hermetic unless explicitly marked and approved.
