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

For measurements, run a benchmark executable directly. Google Benchmark
options such as `--benchmark_filter` and `--benchmark_out` can narrow or save
the results:

```sh
build-benchmark/benchmarks/src/benchmark_forceCalculation
```
