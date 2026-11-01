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
# Besides the instruction count, every benchmark is run once more, outside
# valgrind, with libperf_alloc_counter.so preloaded (benchmarks/perf): it
# reports the heap allocations between the benchmark's perfAllocBegin() and its
# end. Allocation counts are exact; a rise beyond PERF_ALLOC_THRESHOLD_PERCENT
# (default 2, and any rise from 0) fails like an instruction count regression.
# Without the library, or for a binary that has no marker, nothing is compared.
#
# The callgrind run also simulates the caches and the branch predictor (L1 and
# last level geometry fixed below, so the numbers do not depend on the machine).
# The binary is run through a symlink of the same path length and with an empty
# environment: the program path and environment change the stack addresses and
# with them the simulated cache misses by several percent, which would show up
# as a difference between the base and the PR binary. The simulated counts are
# reported next to the instruction counts but never fail the check; the
# instruction count of the same run is the one that is gated.
#
# Writes a markdown report (headline + collapsible per-benchmark table) to
# $PERF_REPORT_FILE (default perf_report.md) and, if set, appends it to
# $GITHUB_STEP_SUMMARY. Exits non-zero if any benchmark regressed past the
# threshold.
#
# Usage: perf_gate.sh <pr_bench_dir> <base_bench_dir> [threshold_percent]
#
# If PERF_BASE_IR_FILE is set, base instruction counts are loaded from that
# file (one "<name>\t<ir>" line per benchmark) instead of being re-measured.
# After running, the freshly-known base counts are (re-)written to the same
# path so the workflow can persist them as a cache keyed on the base SHA.
# A miss falls through to measuring the base binaries — identical to the
# pre-cache behavior.
#
set -uo pipefail

PR_DIR="${1:?usage: perf_gate.sh <pr_bench_dir> <base_bench_dir> [threshold%]}"
BASE_DIR="${2:?usage: perf_gate.sh <pr_bench_dir> <base_bench_dir> [threshold%]}"
THRESHOLD="${3:-2}"
# An improvement at least this large is flagged: a benchmark that silently
# stopped doing its work also looks like a big speed-up. Never fails the check.
LARGE_DROP="${PERF_LARGE_DROP_PERCENT:-30}"
ALLOC_LIB="${PERF_ALLOC_LIB:-$PR_DIR/libperf_alloc_counter.so}"
ALLOC_THRESHOLD="${PERF_ALLOC_THRESHOLD_PERCENT:-2}"
VALGRIND_BIN="$(command -v valgrind || echo valgrind)"
SIM_OPTIONS=(--cache-sim=yes --branch-sim=yes
    --I1=32768,8,64 --D1=32768,8,64 --LL=8388608,16,64)
SIM_WARN_PERCENT="${PERF_SIM_WARN_PERCENT:-10}"
SIM_WARN_MIN_EVENTS="${PERF_SIM_WARN_MIN_EVENTS:-5000}"
REPORT="${PERF_REPORT_FILE:-perf_report.md}"
BASE_IR_FILE="${PERF_BASE_IR_FILE:-}"

# Load cached base Ir counts (if any). The file format is one tab-separated
# "<benchmark name>\t<integer>" line per benchmark. Strict parse: any line
# that doesn't match is ignored.
declare -A base_ir_cache
declare -A base_alloc_cache
declare -A base_sim_cache
if [ -n "$BASE_IR_FILE" ] && [ -s "$BASE_IR_FILE" ]; then
    # columns: name, Ir, allocations, D1 misses, LL misses, mispredicts
    # ("-" for an unknown value)
    while IFS=$'\t' read -r cached_name cached_ir cached_alloc cached_d1 cached_ll cached_br; do
        [[ "$cached_ir" =~ ^[0-9]+$ ]] || continue
        base_ir_cache["$cached_name"]="$cached_ir"
        [[ "${cached_alloc:-}" =~ ^[0-9]+$ ]] &&
            base_alloc_cache["$cached_name"]="$cached_alloc"
        if [[ "${cached_d1:-}" =~ ^[0-9]+$ && "${cached_ll:-}" =~ ^[0-9]+$ &&
            "${cached_br:-}" =~ ^[0-9]+$ ]]; then
            base_sim_cache["$cached_name"]="$cached_d1 $cached_ll $cached_br"
        fi
    done < "$BASE_IR_FILE"
fi
# Collect freshly-known base counts (cached + measured) for write-back.
declare -A base_ir_fresh
declare -A base_alloc_fresh
declare -A base_sim_fresh

# Runs a binary under callgrind with cache and branch simulation and echoes
# "<Ir> <D1 misses> <LL misses> <branch mispredicts>" (the last three are
# missing if valgrind did not print them). Prints nothing if the benchmark
# exits non-zero or no count was reported, so a crashing benchmark can never
# look like a speed-up. See the header for why a symlink and an empty
# environment are used.
measure() {
    local log rc dir
    log="$(mktemp)"
    dir="$(mktemp -d)"
    ln -s "$(readlink -f "$1")" "$dir/b"
    env -i "$VALGRIND_BIN" --tool=callgrind --callgrind-out-file=/dev/null \
        "${SIM_OPTIONS[@]}" "$dir/b" >/dev/null 2>"$log"
    rc=$?
    rm -rf "$dir"
    if [ "$rc" -eq 0 ]; then
        awk '
            function number(line) { sub(/^[^:]*:[ ]*/, "", line); sub(/[ (].*$/, "", line); gsub(/,/, "", line); return line }
            /I +refs:/                { ir = number($0) }
            /D1 +misses:/             { d1 = number($0) }
            /^(==[0-9]+== )?LL misses:/ { ll = number($0) }
            /Mispredicts:/            { br = number($0) }
            END { if (ir != "") { out = ir; if (d1 != "" && ll != "" && br != "") out = out " " d1 " " ll " " br; print out } }
        ' "$log"
    fi
    rm -f "$log"
}

fmt() {   # raw instruction count -> "x.xxM"
    awk -v n="$1" 'BEGIN{ printf "%.2fM", n/1e6 }'
}

# Heap allocations of the measured loop of a benchmark (empty if the library is
# missing, the benchmark has no marker or exits non-zero).
measure_allocs() {
    local log rc
    [ -f "$ALLOC_LIB" ] || return 0
    log="$(mktemp)"
    LD_PRELOAD="$ALLOC_LIB" "$1" >/dev/null 2>"$log"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        awk '$1 == "PQ_PERF_ALLOCS" && $2 ~ /^[0-9]+$/ { n = $2 } END { if (n != "") print n }' "$log"
    fi
    rm -f "$log"
}

status=0
alloc_rows=""
alloc_regressed=""

# <name> <base allocations or empty> <PR allocations or empty>
add_alloc_row() {
    local name="$1" base="${2:-}" pr="${3:-}" verdict emoji delta
    [[ "$pr" =~ ^[0-9]+$ ]] || return 0
    if ! [[ "$base" =~ ^[0-9]+$ ]]; then
        alloc_rows+="| \`$name\` | – | $pr | – | |"$'\n'
        return 0
    fi
    verdict="$(awk -v p="$pr" -v b="$base" -v t="$ALLOC_THRESHOLD" \
        'BEGIN{ if ((b==0 && p>0) || (b>0 && p>b*(1+t/100))) print "regress";
                else if (p<b) print "improve"; else print "ok" }')"
    delta="$(awk -v p="$pr" -v b="$base" \
        'BEGIN{ if (b==0) printf "%+d", p-b; else printf "%+.2f%%", (p-b)/b*100 }')"
    case "$verdict" in
        regress) emoji="❌"; status=1; alloc_regressed+="\`$name\` $base to $pr "
                 echo "::error::$name allocates more: $base to $pr (threshold ${ALLOC_THRESHOLD}%)" ;;
        improve) emoji="🎉" ;;
        *)       emoji="✅" ;;
    esac
    alloc_rows+="| \`$name\` | $base | $pr | $delta | $emoji |"$'\n'
}

sim_rows=""

# <name> <base "d1 ll br" or empty> <PR "d1 ll br" or empty>; report only
add_sim_row() {
    local name="$1" base="${2:-}" pr="${3:-}" i delta cells="" warn=""
    local -a b=() p=()
    read -r -a p <<< "$pr"
    read -r -a b <<< "$base"
    [ "${#p[@]}" -eq 3 ] || return 0
    for i in 0 1 2; do
        if [ "${#b[@]}" -eq 3 ]; then
            delta="$(awk -v p="${p[$i]}" -v b="${b[$i]}" \
                'BEGIN{ if (b==0) printf "%+d", p-b; else printf "%+.1f%%", (p-b)/b*100 }')"
            if awk -v p="${p[$i]}" -v b="${b[$i]}" -v w="$SIM_WARN_PERCENT" -v m="$SIM_WARN_MIN_EVENTS" \
                'BEGIN{ exit !((p-b) >= m && (b==0 || (p-b)/b*100 >= w)) }'; then
                warn="⚠️"
            fi
            cells+=" ${p[$i]} ($delta) |"
        else
            cells+=" ${p[$i]} (–) |"
        fi
    done
    sim_rows+="| \`$name\` |$cells $warn |"$'\n'
}

rows=""
regressed=""
improved=""
unmeasured=""
large_drops=""
declare -A pr_names
declare -A pr_ir_all
shopt -s nullglob

for pr_bin in "$PR_DIR"/perf_*; do
    [ -x "$pr_bin" ] || continue
    name="$(basename "$pr_bin" | sed 's/^perf_//')"
    base_bin="$BASE_DIR/$(basename "$pr_bin")"

    pr_names["$name"]=1
    pr_ir="" pr_d1="" pr_ll="" pr_br=""
    read -r pr_ir pr_d1 pr_ll pr_br <<< "$(measure "$pr_bin")"
    pr_sim=""
    [ -n "$pr_br" ] && pr_sim="$pr_d1 $pr_ll $pr_br"
    if ! [[ "$pr_ir" =~ ^[0-9]+$ ]]; then
        echo "::error::$name: no instruction count (the benchmark failed or valgrind reported none)"
        rows+="| \`$name\` | – | – | no count | ⚠️ |"$'\n'
        unmeasured+="\`$name\` "
        status=1
        continue
    fi
    pr_ir_all["$name"]="$pr_ir"
    pr_alloc="$(measure_allocs "$pr_bin")"
    base_alloc=""
    base_sim=""

    if [ -n "${base_ir_cache[$name]:-}" ]; then
        base_ir="${base_ir_cache[$name]}"
        base_alloc="${base_alloc_cache[$name]:-}"
        base_sim="${base_sim_cache[$name]:-}"
    elif [ -x "$base_bin" ]; then
        base_ir="" b_d1="" b_ll="" b_br=""
        read -r base_ir b_d1 b_ll b_br <<< "$(measure "$base_bin")"
        if ! [[ "$base_ir" =~ ^[0-9]+$ ]]; then
            echo "::warning::$name: the base binary gave no instruction count - not compared"
            rows+="| \`$name\` | – | $(fmt "$pr_ir") | base: no count | ⚠️ |"$'\n'
            add_alloc_row "$name" "" "$pr_alloc"
            add_sim_row "$name" "" "$pr_sim"
            continue
        fi
        [ -n "$b_br" ] && base_sim="$b_d1 $b_ll $b_br"
        base_alloc="$(measure_allocs "$base_bin")"
    else
        echo "$name: new benchmark (no baseline) - pr Ir=$pr_ir"
        rows+="| \`$name\` | – | $(fmt "$pr_ir") | new | ➕ |"$'\n'
        add_alloc_row "$name" "" "$pr_alloc"
        add_sim_row "$name" "" "$pr_sim"
        continue
    fi
    base_ir_fresh["$name"]="$base_ir"
    base_alloc_fresh["$name"]="$base_alloc"
    base_sim_fresh["$name"]="$base_sim"
    add_alloc_row "$name" "$base_alloc" "$pr_alloc"
    add_sim_row "$name" "$base_sim" "$pr_sim"
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
                 echo "$name: ${pct}% (improved)"
                 if awk -v p="$pct" -v t="$LARGE_DROP" 'BEGIN{ exit !(-p >= t) }'; then
                     emoji="⚠️"
                     large_drops+="\`$name\` ${pct}% "
                     echo "::warning::$name dropped ${pct}% - check that the benchmark still does its work"
                 fi ;;
        *)       emoji="✅"; echo "$name: ${pct}%" ;;
    esac
    rows+="| \`$name\` | $(fmt "$base_ir") | $(fmt "$pr_ir") | ${pct}% | $emoji |"$'\n'
done

# ---- scaling: benchmarks named <group>N<molecules> form a size series ----
# Exponent = ln(Ir_large / Ir_small) / ln(n_large / n_small); 1.00 is linear.
# Informational: a worse exponent also shows up as a regression of the largest
# benchmark, which the per-benchmark gate above already catches.
declare -A scaling_groups
for name in "${!pr_ir_all[@]}"; do
    if [[ "$name" =~ ^(.+)N([0-9]+)$ ]]; then
        scaling_groups["${BASH_REMATCH[1]}"]+="${BASH_REMATCH[2]} "
    fi
done
scaling_rows=""
for group in "${!scaling_groups[@]}"; do
    # shellcheck disable=SC2086
    read -r -a sizes <<< "$(printf '%s\n' ${scaling_groups[$group]} | sort -n | tr '\n' ' ')"
    [ "${#sizes[@]}" -ge 2 ] || continue
    small="${sizes[0]}"
    large="${sizes[${#sizes[@]} - 1]}"
    exponent() {   # <ir small> <ir large>
        awk -v a="$1" -v b="$2" -v s="$small" -v l="$large" \
            'BEGIN{ if (a>0 && b>0) printf "%.2f", log(b/a)/log(l/s); else print "-" }'
    }
    pr_exp="$(exponent "${pr_ir_all[${group}N$small]}" "${pr_ir_all[${group}N$large]}")"
    base_exp="–"
    if [ -n "${base_ir_fresh[${group}N$small]:-}" ] && [ -n "${base_ir_fresh[${group}N$large]:-}" ]; then
        base_exp="$(exponent "${base_ir_fresh[${group}N$small]}" "${base_ir_fresh[${group}N$large]}")"
    fi
    scaling_rows+="| \`$group\` | $small to $large | $base_exp | $pr_exp |"$'\n'
done

# Benchmarks that exist on the base but not in the PR (deleted or renamed):
# not a failure, but never silent.
declare -A base_names
for base_bin in "$BASE_DIR"/perf_*; do
    [ -x "$base_bin" ] && base_names["$(basename "$base_bin" | sed 's/^perf_//')"]=1
done
for name in "${!base_ir_cache[@]}"; do base_names["$name"]=1; done
for name in "${!base_names[@]}"; do
    [ -n "${pr_names[$name]:-}" ] && continue
    echo "$name: present on the base branch but not in the PR (removed or renamed)"
    base_col="–"
    [ -n "${base_ir_cache[$name]:-}" ] && base_col="$(fmt "${base_ir_cache[$name]}")"
    rows+="| \`$name\` | $base_col | – | removed | ➖ |"$'\n'
done

# ---- markdown report (headline + collapsible table + footer) ----
if [ -n "$unmeasured" ]; then
    headline="❌ no instruction count: $unmeasured"
    details_open=" open"
elif [ -n "$regressed" ]; then
    headline="❌ regression: $regressed"
    details_open=" open"
elif [ -n "$alloc_regressed" ]; then
    headline="❌ more heap allocations: $alloc_regressed"
    details_open=" open"
elif [ -n "$large_drops" ]; then
    headline="⚠️ large drop, check that the benchmark still does its work: $large_drops"
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
    if [ -n "$scaling_rows" ]; then
        echo ""
        echo "Scaling of the size series (exponent of Ir over the number of molecules, 1.00 = linear):"
        echo ""
        echo "| series | sizes | base | PR |"
        echo "|---|---|---|---|"
        printf '%s' "$scaling_rows"
    fi
    if [ -n "$alloc_rows" ]; then
        echo ""
        echo "Heap allocations in the measured loop (exact, independent of the CPU):"
        echo ""
        echo "| benchmark | base | PR | Δ | |"
        echo "|---|---|---|---|---|"
        printf '%s' "$alloc_rows"
    fi
    if [ -n "$sim_rows" ]; then
        echo ""
        echo "Simulated cache and branch behaviour (callgrind with a 32 KiB 8-way L1 data cache, an 8 MiB 16-way last level cache and its branch predictor; value of the PR and change against the base; report only, ⚠️ = at least +${SIM_WARN_PERCENT}% and +${SIM_WARN_MIN_EVENTS} events):"
        echo ""
        echo "| benchmark | D1 misses | LL misses | branch mispredicts | |"
        echo "|---|---|---|---|---|"
        printf '%s' "$sim_rows"
    fi
    echo "</details>"
    echo ""
    echo "<sub><em>Deterministic callgrind instruction counts vs the base branch; gated at ±${THRESHOLD}%. Not wall-clock.</em></sub>"
} > "$REPORT"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    cat "$REPORT" >> "$GITHUB_STEP_SUMMARY"
fi

# Persist the freshly-known base Ir counts so the workflow's cache (keyed on
# the base SHA) carries them into the next run. Always rewrite — picks up any
# newly added benchmarks while preserving counts loaded from a prior cache.
if [ -n "$BASE_IR_FILE" ] && [ "${#base_ir_fresh[@]}" -gt 0 ]; then
    : > "$BASE_IR_FILE"
    for name in "${!base_ir_fresh[@]}"; do
        sim_cols="- - -"
        [ -n "${base_sim_fresh[$name]:-}" ] && sim_cols="${base_sim_fresh[$name]}"
        read -r d1 ll br <<< "$sim_cols"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "${base_ir_fresh[$name]}" \
            "${base_alloc_fresh[$name]:--}" "$d1" "$ll" "$br" >> "$BASE_IR_FILE"
    done
fi

exit $status
