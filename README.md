# CED

CED builds an extensible biomedical knowledge graph from declarative sources and trains a relational graph autoencoder for ordinary held-out link prediction. It preserves arbitrary source predicates and attributes, mixes Biolink and Amalgam terms verbatim, mints explicit `CED:` identifiers only for configured unresolved CED entities, and separates CPU preparation on `wenceslaus` from GPU execution on Modal.

## Safety boundary

- The local workstation edits, reads, syncs, and orchestrates. It does not run tests, builds, downloads, or CPU-heavy Python.
- `wenceslaus` runs all hermetic tests, lint/type checks, fetches, builds, canonicalization, tensor preparation, and CPU evaluation with at most four workers.
- Modal runs GPU training and inference only. It never downloads sources, builds the graph, prepares tensors, or performs CPU preprocessing.
- Routine verification uses synthetic fixtures and mocks. Large downloads, full builds, dataset preparation, Modal smoke jobs, and paid GPU runs require explicit approval.
- `ced fetch` is the only command permitted to download payloads. Builds must not implicitly download.
- Never download fullmap. Use the existing read-only database through `CED_FULLMAP` or `--fullmap-path`.

## Setup

Copy `.env.example` to `.env` and adjust machine-specific values. Do not commit or print credentials.

```sh
export PATH=/home/sgoetz/.local/bin:$PATH
ced sync
ced check
```

`ced sync` rsyncs the local checkout to `CED_REMOTE_PROJECT` and installs the frozen uv environment. It excludes `.git`, `.venv`, `.ced`, `.env`, `.tablassert`, `plans`, and caches.

Important environment variables:

| Variable | Purpose |
| --- | --- |
| `CED_REMOTE_HOST` | SSH host for CPU work; default `wenceslaus` |
| `CED_REMOTE_PROJECT` | Isolated remote checkout |
| `CED_ARTIFACT_ROOT` | Gitignored artifact directory, relative to the remote project |
| `CED_FULLMAP` | Existing fullmap database path |
| `CED_CPU_WORKERS` | Hard CPU worker cap; must remain `4` |
| `CED_MEMORY_LIMIT` | DuckDB memory cap |
| `CED_DISK_RESERVE_GB` | Minimum free disk before downloads/builds |
| `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` | Modal credentials, only in the untracked remote environment |

## Commands

```sh
ced config validate
ced sources list
ced fetch --source chembl --yes
ced build --source ced --source amalgam --fullmap-path /home/sgoetz/Desktop/fullmap
ced prepare <snapshot-id>
ced encode-text <dataset-id> --provider random
ced train <dataset-id> --epochs 1 --seed 20261110 --yes
ced train <dataset-id> --resume <run-id> --yes
ced evaluate <run-id> --split test --ranking exact --yes
```

`ced check` runs the full remote gate: `uv lock --check`, Ruff, Pyright, pytest, and strict YAML lint. Default pytest excludes `network`, `gpu`, and `fullmap` tests.

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

Sentence Transformers support is optional and centralized in `src/ced/model/encoding.py`. The default provider is `none`; tests use deterministic random encoders or mocked models and never download weights. Configure `text_encoding.provider=sentence_transformers` and a locally available model before running `ced encode-text`.

## Development

Keep code typed, concise, Ruff-formatted, and Pyright-clean. Use PyTorch for model and matrix math. NumPy and Arrow handle in-memory arrays and DuckDB handoff, with conversion to Torch tensors before computation. Every persisted tensor is a Parquet column written by a streaming DuckDB `COPY` and read back through `TensorRef`; preparation never memory-maps or iterates per statement. Tests must remain offline and hermetic unless explicitly marked and approved.
