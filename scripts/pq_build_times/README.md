# Local build-time tracking

A small toolkit for analysing and improving PQ's build times step by step, on **your** machine.
Standard library only. It is deliberately **uncoupled from the CI metrics** (`.github/ci-metrics`):
it has its own parsers and schema, and its data never leaves your machine.

```text
python3 scripts/pq_build_times/pqbt.py snapshot --note "what changed"   # measure and store
python3 scripts/pq_build_times/pqbt.py baseline set                     # pin the newest snapshot as the baseline
python3 scripts/pq_build_times/pqbt.py report                           # HTML graph + terminal table
python3 scripts/pq_build_times/pqbt.py list                             # stored fingerprints
```

## What a snapshot measures

It configures and builds in a directory **it owns** (default `build-times/` in the repository, ignored by git; it
is deleted and recreated for the cold build, and the tool refuses to touch a directory it did not create) and runs
these scenarios:

| Scenario | What it answers |
| --- | --- |
| `cold` | configure + full build from nothing (dependencies come from a cache, so the network is not timed) |
| `noop` | cost of "nothing to do" |
| `touch_leaf` | a median source file changed: the everyday edit |
| `touch_header_top` | the most widely included project header changed: the worst case |
| `touch_header_median` | a median shared header changed |

Besides wall time, each scenario stores the **steps rebuilt** (deterministic: it does not depend on machine
speed), CPU time, parallelism, link time and the run-to-run spread. The snapshot also stores the exact include graph
(objects, files, include pairs and a digest). Cheap scenarios are repeated (`--repeat`, default 3, median stored).
ccache is off unless you pass `--ccache`, because cache hits hide compile time.

The touched files are chosen deterministically from the include graph and then **pinned** to the first snapshot
(or the baseline) of the series, so the numbers stay comparable. Override with `--leaf`, `--header-top`,
`--header-median`.

## Fingerprint, baseline, storage

Every snapshot carries a fingerprint: architecture, CPU model, logical cores, compiler, build type, ninja target,
CMake arguments, `-j`, ccache. Its id is a hash of exactly those, and **only snapshots with the same id are compared
or plotted together**. Change the compiler or a flag and you start a new series (and pin a new baseline); kernel, RAM,
tool versions and the git commit (with a dirty flag and the submodule commits) are recorded for reading only.

Data lives in `$PQ_BUILD_TIMES_DIR`, else `$XDG_DATA_HOME/pq-build-times`, else `~/.local/share/pq-build-times`
(`--data-dir` overrides all). The baseline is pinned explicitly and never overwritten by itself.

## The report

`report` prints a table (the change against the baseline in brackets) and writes one self-contained HTML file with
SVG charts: wall time per scenario over the snapshots, the rebuild scope (steps) and include-graph size, the
baseline as a dashed line and every `--note` as a marker. It needs no network and no packages.

## Habits that keep the numbers honest

- Measure on an idle machine; the tool refuses above a load of 25% of the threads unless you pass `--force`.
- Use `--target` for a part of the project while developing the tool or testing an idea, but know that it is a
  different series (the target is part of the fingerprint).
- To try clang: `--cmake-arg -DCMAKE_CXX_COMPILER=clang++ --cmake-arg -DCMAKE_C_COMPILER=clang`. It becomes its
  own series; `-ftime-trace` data is a planned next step.
- One change per snapshot, with a `--note`; that is what makes the markers in the graph useful.
