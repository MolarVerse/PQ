#!/usr/bin/env bash
set -o pipefail

LOGFILE="clangd-tidy-report.log"

all_files=false
recheck=false
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
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

if $all_files && $recheck; then
    echo "--all and --recheck are mutually exclusive"
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
    echo "  Mode: changed files since origin/dev"
    while IFS=$'\t' read -r status old new; do
        case "$status" in
        D) ;;
        R*) [[ -f "$new" ]] && files+=("$new") ;;
        *) [[ -f "$old" ]] && files+=("$old") ;;
        esac
    done < <(git diff --name-status "$(git merge-base HEAD origin/dev)")

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
clangd-tidy "${files[@]}" -p=. --fail-on-severity=hint --tqdm -j1
