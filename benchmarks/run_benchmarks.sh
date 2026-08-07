#!/usr/bin/env bash

set -euo pipefail

build_dir="${1:-build-benchmark}"
results_dir="${2:-benchmark-results}"
min_time="${PQ_BENCHMARK_MIN_TIME:-3s}"
benchmark_dir="${build_dir}/benchmarks/src"

benchmarks=(
    benchmark_linearAlgebra
    benchmark_box
    benchmark_pairPotentials
    benchmark_cellList
    benchmark_forceCalculation
    benchmark_integrator
    benchmark_kinetics
)

mkdir -p "${results_dir}"

for target in "${benchmarks[@]}"; do
    executable="${benchmark_dir}/${target}"
    if [[ ! -x "${executable}" ]]; then
        printf 'Missing benchmark executable: %s\n' "${executable}" >&2
        exit 1
    fi

    printf '\n== %s ==\n' "${target}"
    "${executable}" \
        --benchmark_min_time="${min_time}" \
        --benchmark_out="${results_dir}/${target}.json" \
        --benchmark_out_format=json
done
