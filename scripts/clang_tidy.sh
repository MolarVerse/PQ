#!/usr/bin/env bash
set -o pipefail

LOGFILE="clangd-tidy-report.log"

# check if log file exists and make a backup if it does
if [[ -f "$LOGFILE" ]]; then
    mv "$LOGFILE" "${LOGFILE}.bak"
fi

# Only stdout goes to the log file; stderr (where --tqdm draws its
# progress bar via carriage returns) stays on the terminal only, so
# the log file doesn't fill up with \r-based redraw noise.
exec > >(tee "$LOGFILE")

echo "Clangd-Tidy:"

all_files=false
while [[ $# -gt 0 ]]; do
    case "$1" in
    --all)
        all_files=true
        shift
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

files=()
if $all_files; then
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
