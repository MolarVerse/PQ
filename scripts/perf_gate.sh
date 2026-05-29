#!/usr/bin/env bash
#
# Performance-regression gate.
#
# Runs every perf_* benchmark in two build directories (the PR build and the
# base-branch build) under callgrind and compares their instruction counts.
# Exits non-zero if any benchmark regressed by more than the threshold.
#
# Instruction counts are deterministic (independent of CI runner load), so this
# is not flaky like wall-clock timing. Each benchmark zeroes the counter after
# setup (CALLGRIND_ZERO_STATS), so the count is pure loop work.
#
# Usage: perf_gate.sh <pr_bench_dir> <base_bench_dir> [threshold_percent]
#
set -uo pipefail

PR_DIR="${1:?usage: perf_gate.sh <pr_bench_dir> <base_bench_dir> [threshold%]}"
BASE_DIR="${2:?usage: perf_gate.sh <pr_bench_dir> <base_bench_dir> [threshold%]}"
THRESHOLD="${3:-2}"

# run a binary under callgrind and echo its instruction count (Ir)
measure() {
    local log
    log="$(mktemp)"
    valgrind --tool=callgrind --callgrind-out-file="$(mktemp)" "$1" \
        >/dev/null 2>"$log"
    grep -oE 'I +refs: +[0-9,]+' "$log" | grep -oE '[0-9,]+' | tr -d ,
    rm -f "$log"
}

status=0
found=0
shopt -s nullglob

for pr_bin in "$PR_DIR"/perf_*; do
    [ -x "$pr_bin" ] || continue
    found=1
    name="$(basename "$pr_bin")"
    base_bin="$BASE_DIR/$name"

    pr_ir="$(measure "$pr_bin")"

    if [ ! -x "$base_bin" ]; then
        echo "$name: no baseline on base branch (new benchmark) - pr Ir=$pr_ir"
        continue
    fi

    base_ir="$(measure "$base_bin")"
    pct="$(awk -v p="$pr_ir" -v b="$base_ir" \
        'BEGIN{ if (b==0) print "nan"; else printf "%.2f", (p-b)/b*100 }')"
    regressed="$(awk -v p="$pr_ir" -v b="$base_ir" -v t="$THRESHOLD" \
        'BEGIN{ print (b>0 && p > b*(1+t/100)) ? 1 : 0 }')"

    if [ "$regressed" = "1" ]; then
        echo "::error::$name regressed ${pct}% (Ir base=$base_ir pr=$pr_ir, threshold ${THRESHOLD}%)"
        status=1
    else
        echo "$name: ${pct}% (Ir base=$base_ir pr=$pr_ir, threshold ${THRESHOLD}%)"
    fi
done

if [ "$found" = "0" ]; then
    echo "no perf_* benchmarks found in $PR_DIR"
fi

exit $status
