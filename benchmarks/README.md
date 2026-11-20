# Benchmarks

The Google Benchmark suite measures core kernels and algorithms with
statistical wall-clock sampling. It is separate from the fixed-work callgrind
benchmarks in `benchmarks/perf`.

Configure and build a release suite:

```sh
cmake -S . -B build-benchmark \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_WITH_BENCHMARKING=On \
    -DBUILD_WITH_TESTS=Off \
    -DBUILD_WITH_ASE=Off
cmake --build build-benchmark --target google_benchmarks
```

Run the short smoke set:

```sh
ctest --test-dir build-benchmark -L benchmark --output-on-failure
```

Run the full local suite from the repository root:

```sh
./benchmarks/run_benchmarks.sh build-benchmark
```

The runner measures every registered case for at least three seconds and writes
one JSON result per executable to `benchmark-results`. Set
`PQ_BENCHMARK_MIN_TIME` to adjust the sampling time.

The benchmarks are standalone executables under
`build-benchmark/benchmarks/src`; they are not invoked through the `pq`
executable. Google Benchmark options such as `--benchmark_filter` and
`--benchmark_out` can narrow or save the results:

```sh
./build-benchmark/benchmarks/src/benchmark_forceCalculation
```

## Checking a change locally

`scripts/perf_local.py` runs the checks of the CI performance gate on your own
machine, for your working tree (uncommitted changes included) against a base
commit. It builds the fixed-work benchmarks of `benchmarks/perf` (portable
Release, like CI) in `build-perf-local/` and runs both CI tools:

- `scripts/perf_gate.sh`: **instruction counts** (callgrind) and heap
  allocations. Exact, the same on every machine and independent of what else
  runs. This is the check that decides whether code got more expensive.
- `scripts/perf_timing.py`: **CPU time** of the measured loop, paired and
  interleaved with the base, with a 95 % interval. It sees what counts cannot
  (memory, system calls, file I/O) but needs a quiet machine.

```sh
scripts/perf_local.py                       # all benchmarks, base = merge-base with origin/dev
scripts/perf_local.py --only interWater     # benchmarks whose name contains it (several allowed)
scripts/perf_local.py --counts-only         # no timing: seconds instead of minutes
scripts/perf_local.py --base HEAD           # what did my uncommitted change do?
scripts/perf_local.py --native              # build with -march=native, as a user would
scripts/perf_local.py --list                # the benchmark names
```

While you work, `--only <name> --counts-only` is the quick loop; run the whole
thing before you open a pull request. The exit code is 1 if the counts show a
regression (`--report-only` always exits 0). Reports are written to
`build-perf-local/report-counts.md` and `report-timing.md`.

The first run builds your tree and the base (a few minutes with a cold cache).
The base is exported from git, never taken from your working tree, built once
per commit and its counts are cached, so later runs only measure your side. If
the build downloads Eigen on every configure, point `PQ_EIGEN_SRC` at an
existing Eigen checkout. `--rebuild-base` and deleting `build-perf-local/` start
over.

For timing the tool pins the process to the CPU with the least load whose
hyper-thread sibling is idle and warns if other programs are running or the
CPU governor is not `performance`. Close other programs for a timing run. On a
laptop, or in a container or virtual machine, timings are only good for a
comparison within one run (which is what this tool does), not as absolute
numbers. Compare `--native` runs with `--native` runs only: the counts of a
portable and a native build differ.

### End-to-end scenarios

The benchmarks measure parts of the code. `--suite scenarios` (or `--suite all`)
instead runs the whole `PQ` program, as a user would, on inputs of the examples
and of the integration tests listed in `benchmarks/scenarios.json`, for your
tree and for the base:

```sh
scripts/perf_local.py --suite scenarios --counts-only        # minutes
scripts/perf_local.py --suite scenarios --only cgo           # scenarios whose name contains it
scripts/perf_local.py --suite all --base HEAD                # benchmarks and scenarios
```

- **Counts**: instructions of a short run (`count_steps`) under callgrind. A
  change that makes any part of a whole run more expensive shows up here. Unlike
  the benchmarks the counts are only reproducible to about 1e-7, so the
  threshold is 1 %, not exact.
- **Timing**: CPU time of a run of about a second (`steps`), paired and
  interleaved with the base (7 rounds, 2 copies per side by default, see
  `--scenario-rounds` and `--scenario-copies`). It sees file output and
  startup, but with so few rounds only effects of a few percent.

Both sides run the same input files from your working tree with only `nstep`
and a fixed `random_seed` changed; if the base cannot read an input, the
scenario fails for the base. QM inputs are not scenarios (they need external
programs), and runs of a few milliseconds are too short to time. To add a
scenario, add an entry to `benchmarks/scenarios.json`. CI does not run the
scenarios. A full run (counts and timing, seven scenarios) takes about seven
minutes; the reports are in `build-perf-local/report-scenarios.md`.
