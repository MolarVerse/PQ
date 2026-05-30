#!/usr/bin/env bash
#
# Performance-regression gate.
#
# Runs every perf_* benchmark in two build directories (the PR build and the
# base-branch build) under callgrind and compares their instruction counts.
# Instruction counts are deterministic (independent of CI runner load), so this
# is not flaky like wall-clock timing. Each benchmark zeroes the counter after
# setup (CALLGRIND_ZERO_STATS), so the count is pure loop work.
#
# Writes a markdown report (headline + collapsible per-benchmark table) to
# $PERF_REPORT_FILE (default perf_report.md) and, if set, appends it to
# $GITHUB_STEP_SUMMARY. Exits non-zero if any benchmark regressed past the
# threshold.
#
# Usage: perf_gate.sh <pr_bench_dir> <base_bench_dir> [threshold_percent]
#
set -uo pipefail

PR_DIR="${1:?usage: perf_gate.sh <pr_bench_dir> <base_bench_dir> [threshold%]}"
BASE_DIR="${2:?usage: perf_gate.sh <pr_bench_dir> <base_bench_dir> [threshold%]}"
THRESHOLD="${3:-2}"
REPORT="${PERF_REPORT_FILE:-perf_report.md}"

# run a binary under callgrind and echo its instruction count (Ir)
measure() {
    local log
    log="$(mktemp)"
    valgrind --tool=callgrind --callgrind-out-file="$(mktemp)" "$1" \
        >/dev/null 2>"$log"
    grep -oE 'I +refs: +[0-9,]+' "$log" | grep -oE '[0-9,]+' | tr -d ,
    rm -f "$log"
}

fmt() {   # raw instruction count -> "x.xxM"
    awk -v n="$1" 'BEGIN{ printf "%.2fM", n/1e6 }'
}

status=0
rows=""
regressed=""
improved=""
shopt -s nullglob

for pr_bin in "$PR_DIR"/perf_*; do
    [ -x "$pr_bin" ] || continue
    name="$(basename "$pr_bin" | sed 's/^perf_//')"
    base_bin="$BASE_DIR/$(basename "$pr_bin")"

    pr_ir="$(measure "$pr_bin")"

    if [ ! -x "$base_bin" ]; then
        echo "$name: new benchmark (no baseline) - pr Ir=$pr_ir"
        rows+="| \`$name\` | – | $(fmt "$pr_ir") | new | ➕ |"$'\n'
        continue
    fi

    base_ir="$(measure "$base_bin")"
    pct="$(awk -v p="$pr_ir" -v b="$base_ir" \
        'BEGIN{ if (b==0) print "0.00"; else printf "%+.2f", (p-b)/b*100 }')"
    verdict="$(awk -v p="$pr_ir" -v b="$base_ir" -v t="$THRESHOLD" \
        'BEGIN{ if (b>0 && p>b*(1+t/100)) print "regress";
                else if (b>0 && p<b*(1-t/100)) print "improve";
                else print "ok" }')"

    case "$verdict" in
        regress) emoji="❌"; status=1; regressed+="\`$name\` ${pct}% "
                 echo "::error::$name regressed ${pct}% (Ir base=$base_ir pr=$pr_ir, threshold ${THRESHOLD}%)" ;;
        improve) emoji="🎉"; improved+="\`$name\` ${pct}% "
                 echo "$name: ${pct}% (improved)" ;;
        *)       emoji="✅"; echo "$name: ${pct}%" ;;
    esac
    rows+="| \`$name\` | $(fmt "$base_ir") | $(fmt "$pr_ir") | ${pct}% | $emoji |"$'\n'
done

# ---- markdown report (headline + collapsible table + footer) ----
if [ -n "$regressed" ]; then
    headline="❌ regression: $regressed"
    details_open=" open"
elif [ -n "$improved" ]; then
    headline="🎉 improvement: $improved"
    details_open=""
else
    headline="✅ no regressions"
    details_open=""
fi

{
    echo "<!-- perf-gate-comment -->"
    echo "### ⚡ Performance (instruction count) — $headline"
    echo ""
    echo "<details${details_open}><summary>per-benchmark breakdown</summary>"
    echo ""
    echo "| benchmark | base Ir | PR Ir | Δ | |"
    echo "|---|---|---|---|---|"
    printf '%s' "$rows"
    echo "</details>"
    echo ""
    echo "<sub><em>Deterministic callgrind instruction counts vs the base branch; gated at ±${THRESHOLD}%. Not wall-clock.</em></sub>"
} > "$REPORT"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    cat "$REPORT" >> "$GITHUB_STEP_SUMMARY"
fi

exit $status
