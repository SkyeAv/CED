# Acquisition boundary

All implementations run on wenceslaus. `fetch` is the only command permitted to download. Builds call `locate_source`, which uses local paths, verified aria2 stores, segmented HTTP stores, or Hugging Face `local_files_only=True` and emits an exact fetch command when an artifact is absent. No stage downloads implicitly: a missing artifact fails with the `ced fetch --source <name>` command that would repair it.

`fetch_sources` materializes the artifacts of all selected sources concurrently, bounded by `runtime.download_jobs` (at most four). Concurrency never widens the boundary: every artifact still verifies its declared size and checksum, each source records its own `fetch-manifest.json`, and a selection whose total exceeds 1 GiB still refuses to run without `--yes`. When several artifacts fail, the first error is raised and the rest are logged, so one broken source cannot hide another.

Backends:

| `backend` | Transport | Knob |
| --- | --- | --- |
| `local` | Existing files under a configured root | none |
| `huggingface` | `huggingface_hub` with `local_files_only=True` at build time | `revision` |
| `aria2` | `aria2c` subprocess for very large payloads | `connections`, `revision` |
| `http` | Dependency-free segmented `Range` downloads in `http.py` | `segment_bytes`, `connections` |

The `http` backend probes each artifact with `Range: bytes=0-0`. A `206` answer yields the exact size from `Content-Range` and enables segmentation into at most `runtime.download_connections` chunks of `segment_bytes` (default 16 MiB), downloaded concurrently into one file at their offsets; a `200` answer falls back to a single stream, and `416` means the object is empty. Transient failures (408, 429, 5xx, network errors) are retried with exponential backoff up to `max_tries`, while other 4xx answers fail immediately. An interrupted transfer keeps its `.part` file and `.part.state.json` cursor for resume; the declared size and SHA-256 are verified before an atomic rename into the store, and a verification failure discards the partial and starts clean. Pre-signed query strings are stripped by `redact_url()` from every log message, exception, origin record, and manifest field; the unredacted URL is used for the request only.

Extraction stays behind `archive.py`: member patterns, member counts, and expanded byte caps are declared per artifact, and a member that exceeds its cap fails the fetch instead of filling the disk.
