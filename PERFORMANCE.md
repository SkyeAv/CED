# Performance verification

## Scope and protocol

This report covers the optimization stories in commits `cb032d2` through
`f9f00bc`. All benchmarks ran on `wenceslaus.systemsbiology.net` with
`CED_REMOTE_WORKER=1`, `CED_CPU_WORKERS=4`, and the repository's thread-pool
caps. Final records were written between 2026-11-12T18:57:22Z and
2026-11-12T18:59:50Z.

Each cell is `mean [min, max]` wall time. Speedup is baseline mean divided by
final mean; values below 1.0x are regressions. For cumulative benchmark series,
the baseline is the first record and the final is the latest record. Rows backed
by a frozen reference compare the latest reference and optimized pair. Workload
JSON and per-run records are stored under the gitignored `.ced/perf/` directory.

Reproduce with:

```sh
ssh wenceslaus 'cd /local_raid1/sgoetz/Code/ISB/CED-optimizations && export CED_REMOTE_WORKER=1 CED_CPU_WORKERS=4 PYTHONUTF8=1 && uv run --no-sync pytest -m perf --disable-warnings -q'
```

Final result: `12 passed, 84 deselected, 22 warnings in 148.69s`.

## Before/after results

| Hotspot | Baseline mean [min, max] | Final mean [min, max] | Speedup | Iterations | Workload |
| --- | --- | --- | --- | --- | --- |
| typed_pools | 4.287 ms [4.261, 4.307] | 3.390 ms [3.384, 3.401] | 1.26x | 3 | {"nodes":2000,"edges":20000,"relations":8,"type_sets":4,"triples":200,"count":8} |
| sample_negatives | 7.560 ms [7.534, 7.584] | 1.454 ms [1.429, 1.488] | 5.20x | 3 | {"nodes":2000,"edges":20000,"relations":8,"type_sets":4,"triples":200,"count":8} |
| sample_neighborhood | 1132.177 ms [1129.579, 1136.075] | 57.224 ms [57.043, 57.411] | 19.79x | 3 | {"nodes":5000,"edges":50000,"relations":8,"seeds":100,"fanout":[15,10]} |
| shared adjacency | 1132.177 ms [1129.579, 1136.075] | 47.285 ms [47.230, 47.385] | 23.94x | 3 | {"nodes":5000,"edges":50000,"relations":8,"seeds":100,"fanout":[15,10]} |
| prepare topology | 258.735 ms [249.464, 275.634] | 216.083 ms [200.653, 236.962] | 1.20x | 3 | {"entities":500,"predicates":20,"statements":20000} |
| prepare topology (memmap) | 483.239 ms [441.610, 524.869] | 211.163 ms [205.572, 221.571] | 2.29x | 3 | {"statements":20000,"backing":"memmap"} |
| write_ordered categorical | 227.697 ms [216.228, 233.558] | 209.661 ms [207.608, 212.549] | 1.09x | 3 | {"owners":2000,"fields":6} |
| write_ordered numeric | 211.044 ms [199.923, 216.609] | 205.474 ms [199.953, 212.365] | 1.03x | 3 | {"owners":2000,"fields":6} |
| encoder forward | 288.082 ms [286.973, 288.748] | 193.187 ms [191.341, 194.775] | 1.49x | 3 | {"nodes":5000,"edges":20000,"edge_types":128,"layers":2,"dimension":32} |
| auxiliary heads | 23.967 ms [20.685, 28.955] | 33.395 ms [21.007, 65.717] | 0.72x | 5 | {"states":5000,"fields":8,"dimension":32} |
| train_step | 72.442 ms [72.119, 72.648] | 64.643 ms [64.522, 64.810] | 1.12x | 3 | {"nodes":2998,"edges":12000,"positives":500,"negatives":4,"features":"none"} |
| evaluation messages | 656.565 ms [654.491, 659.568] | 38.785 ms [38.577, 38.912] | 16.93x | 3 | {"nodes":5000,"edges":20000,"relations":8} |
| evaluation positives | 86.465 ms [25.842, 207.555] | 113.531 ms [61.352, 216.852] | 0.76x | 3 | {"nodes":5000,"edges":20000,"relations":8} |
| evaluation scoring | 43.082 ms [43.000, 43.175] | 18.793 ms [18.649, 19.048] | 2.29x | 3 | {"nodes":5000,"targets":32,"relations_per_batch":1,"dimension":32} |
| merge_nodes | 239.351 ms [228.657, 260.451] | 169.772 ms [169.099, 170.861] | 1.41x | 3 | {"identities":1000,"duplicates":8,"rows":8000} |
| shard edge types | 19.952 ms [19.513, 20.172] | 1.905 ms [1.901, 1.910] | 10.48x | 3 | {"nodes":2000,"edges":20000,"train_edges":14060,"triples":2048} |
| shard triples | 1.715 ms [1.706, 1.731] | 0.304 ms [0.302, 0.307] | 5.64x | 3 | {"nodes":2000,"edges":20000,"train_edges":14060,"triples":2048} |

The shared host produced visible timing variance, so the min/max ranges are part
of the result. The final auxiliary-head run was slower on CPU; that change
removes CUDA host-device synchronization and is not a CPU optimization. The
evaluation-positive index did not show a stable mean speedup in this run. No
GPU timing claim is made for either case.

## Correctness evidence

Frozen-reference tests cover negative sampling, neighborhood sampling, prepare
and feature writes, encoder forward/backward, training auxiliary tensors and
losses, evaluation messages/positives/scores/reports, merged rows/KGX bytes, and
in-memory/DiskGraph batch tensor bytes. Model-facing comparisons are bitwise on
CPU; report and artifact comparisons use exact equality. Parameter names,
checkpoint compatibility, and `model_version` remain unchanged because model
numerics did not change.

These claims are based on synthetic fixtures, not a full production KG build.
No scientific source tables, split formulas, split ratios, canonical JSON
ordering, or KGX semantics were changed. No paid Modal GPU job was run, so CUDA
host-sync removal and fused AdamW have correctness coverage but no measured GPU
wall-time result.

## Quality gate

The final `uv run --no-sync ced check` on wenceslaus exited 0. It resolved 115
locked packages, passed Ruff, verified 75 formatted files, reported 0 Pyright
errors/warnings/informations, and completed pytest with `83 passed, 13
deselected, 25 warnings in 24.72s`. Strict YAML lint also exited 0 over all
source and graph YAML files.
