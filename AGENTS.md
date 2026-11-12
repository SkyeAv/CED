# CED agent guide

CED builds an extensible multi-source KGX graph and trains a relational graph autoencoder for held-out link prediction. Python 3.13, uv, Cyclopts, Pydantic, Tablassert, PyTorch, PyG, and Modal.

## Execution safety

- The local workstation is for editing, reading, synchronization, and CLI orchestration only. Do not run tests, lint, type checks, builds, downloads, Python pipelines, or other CPU-heavy work locally, including tiny tests.
- Run every test, quality check, build, download, CPU preprocessing step, and CPU evaluation on `wenceslaus`, capped at four CPU workers. Set the process/thread limits together; do not assume application workers alone cap BLAS, Rayon, Polars, or DuckDB.
- Use `/local_raid1/sgoetz/Code/ISB/CED` as the default isolated remote checkout through `CED_REMOTE_PROJECT`. Do not disturb other remote checkouts or their uncommitted work.
- Put long remote jobs in tmux. Local SSH sessions are orchestration, not permission to run a job interactively for hours.
- Run Modal for GPU training/inference only. Never use Modal for tests, downloads, fullmap work, KG construction, feature preparation, negative sampling, or other CPU preprocessing.
- Use the existing wenceslaus fullmap through `CED_FULLMAP` or `--fullmap-path`; the documented machine default is `/home/sgoetz/Desktop/fullmap`. Never download or build fullmap in this repo.
- Check memory and disk before remote work. Keep downloads, caches, generated graphs, tensors, checkpoints, and reports under the gitignored `.ced/` directory.

## Commands

The local `ced` process only syncs code and delegates work. `ced` is not on the laptop PATH; invoke it through uv from a checkout with a synced environment:

```sh
uv sync                                  # one-time per checkout
uv run --no-sync ced sync                # sync source/config/lockfiles; never delete or expose .ced or credentials
uv run --no-sync ced check               # remote Ruff, Pyright, four-worker pytest-xdist, uv lock, and YAML gate
uv run --no-sync ced config validate
uv run --no-sync ced sources list        # inspect declarative sources without downloading them
uv run --no-sync ced fetch --source chembl
uv run --no-sync ced fetch --group translator
uv run --no-sync ced build --source ced --fullmap-path /home/sgoetz/Desktop/fullmap
uv run --no-sync ced build --group translator
uv run --no-sync ced prepare <snapshot-id>
uv run --no-sync ced encode-text <dataset-id> --provider random
uv run --no-sync ced train <dataset-id> --epochs 1 --seed 20261110 --yes
uv run --no-sync ced train <dataset-id> --resume <run-id> --yes
uv run --no-sync ced evaluate <run-id> --split test --ranking exact --yes
```

For direct debugging, SSH to wenceslaus first and use the remote checkout:

```sh
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv lock --check
uv run prek run --all-files
```

Do not run those remote debugging commands locally. Do not let a local commit hook launch lint or tests; run hooks and commits that need hooks in the remote checkout after synchronization. Use `git push --no-verify` on the laptop so the pre-push hook never launches checks locally; run `uv run --no-sync ced check` first. A normal `git push` on wenceslaus runs the same full gate through prek. Run `uv run prek install` in each wenceslaus Git checkout after hook changes; `ced sync` excludes `.git` and cannot install hooks. Tests are offline and hermetic by default. Network integration, large downloads, full builds, dataset preparations, and paid GPU jobs require explicit selection and approval; do not run them as routine implementation verification.

## Layout

| Path | Responsibility |
| --- | --- |
| `src/ced/config/` | typed runtime and declarative source YAML; source specifics belong here |
| `src/ced/acquisition/` | HF, aria2, segmented HTTP, local, and safe archive acquisition behind shared protocols |
| `src/ced/graphs/` | CED graph config and curated tables, recursively including `families/` and `books/` |
| `src/ced/kg/` | narrow Tablassert compatibility adapter, source adapters, canonical records, streaming merge |
| `src/ced/model/` | raw/numeric tensor contracts, features, splits, encoder, decoder, training, evaluation |
| `src/ced/modal_app.py` | GPU-only Modal entry points and ephemeral artifact exchange |
| `src/ced/modal_plan.py` | Modal-independent transfer planning, bounded rotation, resume, and artifact recovery |
| `tests/` | small offline fixtures and contract tests, executed on wenceslaus |

## Conventions

- Keep code uniform and concise. Ruff owns formatting: 120 columns, double quotes, organized imports, and the shared lint rules in `pyproject.toml`. Pyright and pytest are mandatory remote gates.
- Use small typed modules and functions, Pydantic tagged unions for configuration, protocols for backends/adapters/feature providers, and one canonical representation per concept. Avoid duplicate per-source pipelines, parallel schemas, speculative wrappers, and repeated conversion code.
- Keep source-specific URLs, artifacts, field mappings, licenses, and predicate semantics in `src/ced/config/sources/*.yaml`. Adding a compatible dataset must not edit Python. Keep genuinely new formats inside a narrow adapter.
- Preserve the three scoped Tablassert monkeypatches behind exact version guards and restoration. Do not patch across other modules when the public API or a small pre/post-processing step suffices.
- Use environment variables or untracked machine config for host-specific paths and credentials. Configure them with direnv in the untracked `.envrc` (template: `.envrc.example`); never commit `.envrc`, print secrets, or synchronize credentials. Keep checked-in code free of machine-specific constants.
- Preserve arbitrary predicates and attributes without fabricating ontology mappings. Only unresolved CED values receive deterministic, explicitly marked `CED:` local IDs. Do not merge identities through names, synonyms, or cross-references.
- Use conventional commits (`feat:`, `fix:`, `refactor:`, `perf:`, `test:`, `docs:`, `chore:`). Keep scientific table content unchanged during infrastructure work.

## Git / PR workflow

- Squash-merge PRs into `main` with conventional-commit subjects; the squash commit carries the PR number.  (evidence: history x4)
- Rebase a feature branch onto `main` and fix conflicts before the PR; do not merge locally.  (evidence: sessions x1)
- No tags or releases exist; versioning is not yet in use.  (evidence: history, git tag empty)

## Gotchas

- A stale checkout venv fails with `ModuleNotFoundError` (e.g. `cyclopts`); rerun `uv sync` in that checkout. [verified]
- Non-interactive SSH to wenceslaus can pick up the snap `uv` with the wrong `HOME` and fail with "cannot create user data directory"; prefix `HOME=/home/sgoetz` on remote `uv run` one-liners. [verified]

## Verification boundary

`ced check` and all smoke tests run on wenceslaus using synthetic fixtures. The `ced check` gate uses four xdist workers with one thread per worker; use `uv run pytest -n 0` remotely for serial debugging. The `network`, `gpu`, `fullmap`, and `perf` markers are excluded from the default gate; run performance benchmarks only on wenceslaus with `uv run --no-sync pytest -m perf` (protocol in PERFORMANCE.md). Verify bounded memory, deterministic artifacts, split isolation, global typed negatives, finite gradients, checkpoint resume, and artifact checksums remotely. Do not run actual graph builds, dataset preparations, downloads, or paid Modal jobs during routine verification. Modal smoke jobs use tiny synthetic tensors and ephemeral storage, require explicit approval, and clean up in `finally`. Never download unnamed datasets or fullmap during verification.
