#!/usr/bin/env bash
set -o pipefail

LOGFILE="clangd-tidy-report.log"

all_files=false
recheck=false
base_ref=""
head_ref=""
compile_db="."
# Every batch is a fresh clangd-tidy process. One process over all the files of
# a large change keeps growing (the headers are parsed again and again) and has
# been killed on the CI runner; 0 checks everything in one process.
batch_size="${CLANG_TIDY_BATCH_SIZE:-30}"
while [[ $# -gt 0 ]]; do
    case "$1" in
    --all)
        all_files=true
        shift
        ;;
    --recheck)
        recheck=true
        shift
        ;;
    --base)
        base_ref="$2"
        shift 2
        ;;
    --head)
        head_ref="$2"
        shift 2
        ;;
    -p | --build-dir)
        compile_db="$2"
        shift 2
        ;;
    --batch-size)
        batch_size="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

if ! [[ "$batch_size" =~ ^[0-9]+$ ]]; then
    echo "--batch-size needs a number of files (0 = all in one process), got: $batch_size"
    exit 1
fi

if $all_files && $recheck; then
    echo "--all and --recheck are mutually exclusive"
    exit 1
fi

if { $all_files || $recheck; } && { [[ -n "$base_ref" ]] || [[ -n "$head_ref" ]]; }; then
    echo "--base/--head only apply to the default changed-files mode"
    exit 1
fi

# Capture the previous run's log before it gets rotated to .bak, so
# --recheck can extract failing files from it.
prev_log=""
if [[ -f "$LOGFILE" ]]; then
    prev_log="$(cat "$LOGFILE")"
    mv "$LOGFILE" "${LOGFILE}.bak"
fi

exec > >(tee "$LOGFILE")

echo "Clangd-Tidy:"

files=()
if $recheck; then
    echo "  Mode: recheck files with prior diagnostics"
    if [[ -z "$prev_log" ]]; then
        echo "  No previous log found (${LOGFILE})."
        exit 0
    fi

    # Strip ANSI escape codes in case the log contains colored output.
    clean_log="$(sed -E 's/\x1b\[[0-9;]*m//g' <<<"$prev_log")"

    while IFS= read -r f; do
        [[ -f "$f" ]] && files+=("$f")
    done < <(grep -oiE '^[^:]+:[0-9]+:[0-9]+: (error|warning|hint):' <<<"$clean_log" |
        cut -d: -f1 | sort -u)

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "  No diagnostic lines matched in previous log."
        echo "  --- last 20 lines of previous log for inspection ---"
        tail -n 20 <<<"$clean_log"
        echo "  -----------------------------------------------------"
    fi
elif $all_files; then
    echo "  Mode: all tracked C++ files"
    while IFS= read -r f; do
        [[ -f "$f" ]] && files+=("$f")
    done < <(git ls-files '*.cpp' '*.cxx' '*.cc' '*.c' '*.h' '*.hpp' '*.hxx' -- ':!external')
else
    diff_base="$base_ref"
    [[ -z "$diff_base" ]] && diff_base="$(git merge-base HEAD origin/dev)"
    diff_refs=("$diff_base")
    [[ -n "$head_ref" ]] && diff_refs+=("$head_ref")

    if [[ -n "$base_ref" ]]; then
        echo "  Mode: changed files from $diff_base to ${head_ref:-working tree}"
    else
        echo "  Mode: changed files since origin/dev"
    fi
    while IFS=$'\t' read -r status old new; do
        case "$status" in
        D) ;;
        R*) [[ -f "$new" ]] && files+=("$new") ;;
        *) [[ -f "$old" ]] && files+=("$old") ;;
        esac
    done < <(git diff --name-status "${diff_refs[@]}")

    # Filter to C++ files only (changed mode may include non-source files)
    # and exclude anything under external/
    cpp_files=()
    for f in "${files[@]}"; do
        [[ "$f" == external/* ]] && continue
        [[ "$f" =~ \.(cpp|cxx|cc|c|h|hpp|hxx)$ ]] && cpp_files+=("$f")
    done
    files=("${cpp_files[@]}")
fi

if [[ ${#files[@]} -eq 0 ]]; then
    echo "  No files to check."
    exit 0
fi

echo "  Files: ${#files[@]}"

run_clangd_tidy() {
    clangd-tidy "$@" -p="$compile_db" --fail-on-severity=hint --tqdm -j1
}

total=${#files[@]}
if ((batch_size == 0 || batch_size >= total)); then
    run_clangd_tidy "${files[@]}"
    exit $?
fi

# All batches run even if one finds something, so that the report is complete;
# the exit status is the first failing one.
batches=$(((total + batch_size - 1) / batch_size))
echo "  Batches: $batches of up to $batch_size files"
status=0
for ((start = 0; start < total; start += batch_size)); do
    last=$((start + batch_size))
    ((last > total)) && last=$total
    echo "  Batch $((start / batch_size + 1)) of $batches: files $((start + 1))-$last"
    run_clangd_tidy "${files[@]:start:batch_size}"
    code=$?
    ((status == 0)) && status=$code
done
exit $status
