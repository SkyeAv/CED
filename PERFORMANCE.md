# Performance verification

## Scope and protocol

This report covers the optimization branch rebased onto `3b62f75`, which uses
streamed-Parquet preparation. Earlier `.npy` topology/feature-write optimizations
were dropped during conflict resolution because that implementation superseded
them. Preparation remains covered by `scripts/bench_prepare.py` and the existing
preparation tests; this report contains the 13 remaining applicable hotspots.

All benchmarks ran on `wenceslaus.systemsbiology.net` with `CED_REMOTE_WORKER=1`,
`CED_CPU_WORKERS=4`, and the repository's thread-pool caps. Final records were
written between 2026-11-12T19:53:07Z and 2026-11-12T19:53:13Z.

Each cell is `mean [min, max]` wall time. Speedup is baseline mean divided by
final mean; values below 1.0x are regressions. For cumulative benchmark series,
the baseline is the first record and the final is the latest record. Rows backed
by a frozen reference compare the latest reference and optimized pair. Workload
JSON and per-run records are stored under the gitignored `.ced/perf/` directory.

Reproduce with:

```sh
ssh wenceslaus 'cd /local_raid1/sgoetz/Code/ISB/CED-optimizations && export CED_REMOTE_WORKER=1 CED_CPU_WORKERS=4 PYTHONUTF8=1 && uv run --no-sync pytest -m perf --disable-warnings -q'
```

Final result: `9 passed, 102 deselected, 6 warnings in 9.42s`.

## Before/after results

| Hotspot | Baseline mean [min, max] | Final mean [min, max] | Speedup | Iterations | Workload |
| --- | --- | --- | --- | --- | --- |
| typed_pools | 4.287 ms [4.261, 4.307] | 3.387 ms [3.380, 3.400] | 1.27x | 3 | {"nodes":2000,"edges":20000,"relations":8,"type_sets":4,"triples":200,"count":8} |
| sample_negatives | 7.560 ms [7.534, 7.584] | 1.454 ms [1.440, 1.480] | 5.20x | 3 | {"nodes":2000,"edges":20000,"relations":8,"type_sets":4,"triples":200,"count":8} |
| sample_neighborhood | 1132.177 ms [1129.579, 1136.075] | 56.685 ms [56.492, 56.969] | 19.97x | 3 | {"nodes":5000,"edges":50000,"relations":8,"seeds":100,"fanout":[15,10]} |
| shared adjacency | 1132.177 ms [1129.579, 1136.075] | 46.441 ms [46.187, 46.688] | 24.38x | 3 | {"nodes":5000,"edges":50000,"relations":8,"seeds":100,"fanout":[15,10]} |
| encoder forward | 288.082 ms [286.973, 288.748] | 187.836 ms [187.770, 187.926] | 1.53x | 3 | {"nodes":5000,"edges":20000,"edge_types":128,"layers":2,"dimension":32} |
| auxiliary heads | 23.967 ms [20.685, 28.955] | 21.862 ms [21.677, 22.077] | 1.10x | 5 | {"states":5000,"fields":8,"dimension":32} |
| train_step | 72.442 ms [72.119, 72.648] | 81.006 ms [77.282, 86.460] | 0.89x | 3 | {"nodes":2998,"edges":12000,"positives":500,"negatives":4,"features":"none"} |
| evaluation messages | 656.565 ms [654.491, 659.568] | 37.884 ms [37.479, 38.103] | 17.33x | 3 | {"nodes":5000,"edges":20000,"relations":8} |
| evaluation positives | 86.465 ms [25.842, 207.555] | 127.818 ms [59.943, 263.174] | 0.68x | 3 | {"nodes":5000,"edges":20000,"relations":8} |
| evaluation scoring | 63.290 ms [61.865, 64.192] | 16.879 ms [16.553, 17.076] | 3.75x | 3 | {"nodes":5000,"targets":32,"relations_per_batch":1,"dimension":32} |
| merge_nodes | 235.165 ms [227.538, 248.221] | 173.658 ms [169.012, 182.949] | 1.35x | 3 | {"identities":1000,"duplicates":8,"rows":8000} |
| shard edge types | 20.158 ms [19.604, 20.439] | 1.901 ms [1.896, 1.911] | 10.60x | 3 | {"nodes":2000,"edges":20000,"train_edges":14060,"triples":2048} |
| shard triples | 1.712 ms [1.709, 1.715] | 0.310 ms [0.303, 0.315] | 5.53x | 3 | {"nodes":2000,"edges":20000,"train_edges":14060,"triples":2048} |

The shared host produced visible timing variance, so the min/max ranges are part
of the result. The final `train_step` run was 0.89x on CPU, and the
evaluation-positive index did not show a stable mean speedup. CUDA host-sync
removal and fused AdamW were not GPU-timed.

## Correctness evidence

Frozen-reference tests cover negative sampling, neighborhood sampling, encoder
forward/backward, training auxiliary tensors and losses, evaluation
messages/positives/scores/reports, merged rows/KGX bytes, and
in-memory/DiskGraph batch Parquet bytes. Model-facing comparisons are bitwise on
CPU; report and artifact comparisons use exact equality. Parameter names,
checkpoint compatibility, and `model_version` remain unchanged because model
numerics did not change. Streamed-Parquet preparation is additionally covered
by `test_prepare_topology.py` and `test_bench_prepare.py`.

These claims are based on synthetic fixtures, not a full production KG build.
No scientific source tables, split formulas, split ratios, canonical JSON
ordering, or KGX semantics were changed. No paid Modal GPU job was run, so CUDA
host-sync removal and fused AdamW have correctness coverage but no measured GPU
wall-time result.

## Quality gate

The final `uv run --no-sync ced check` on wenceslaus exited 0. It resolved 115
locked packages, passed Ruff, verified 78 formatted files, reported 0 Pyright
errors/warnings/informations, and completed pytest with `101 passed, 10
deselected, 17 warnings in 17.50s`. Strict YAML lint also exited 0 over all
source and graph YAML files.
