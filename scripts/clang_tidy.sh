#!/usr/bin/env bash
set -eo pipefail

all_files=false
base_ref="origin/dev"
build_dir="."
jobs=1

usage() {
    cat <<'EOF'
Usage: scripts/clang_tidy.sh [options]

Options:
  --all                 Check every tracked C/C++ file outside external/.
  --base <revision>     Compare HEAD with this revision (default: origin/dev).
  --build-dir <path>    Directory containing compile_commands.json (default: .).
  --jobs <count>        Number of concurrent clangd-tidy workers (default: 1).
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --all)
        all_files=true
        shift
        ;;
    --base)
        [[ $# -ge 2 ]] || { echo "--base requires a revision" >&2; exit 2; }
        base_ref="$2"
        shift 2
        ;;
    --build-dir)
        [[ $# -ge 2 ]] || { echo "--build-dir requires a path" >&2; exit 2; }
        build_dir="$2"
        shift 2
        ;;
    --jobs)
        [[ $# -ge 2 ]] || { echo "--jobs requires a count" >&2; exit 2; }
        jobs="$2"
        shift 2
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "--jobs must be a positive integer" >&2
    exit 2
fi

if [[ ! -f "$build_dir/compile_commands.json" ]]; then
    echo "Missing $build_dir/compile_commands.json; configure CMake first." >&2
    exit 2
fi

LOGFILE="clangd-tidy-report.log"
if [[ -f "$LOGFILE" ]]; then
    mv "$LOGFILE" "${LOGFILE}.bak"
fi

# Keep tqdm redraws on stderr instead of filling the report with carriage returns.
exec > >(tee "$LOGFILE")

echo "Clangd-Tidy:"

files=()
if $all_files; then
    echo "  Mode: all tracked C++ files"
    while IFS= read -r f; do
        [[ -f "$f" ]] && files+=("$f")
    done < <(
        git ls-files \
            '*.cpp' '*.cxx' '*.cc' '*.c' \
            '*.h' '*.hpp' '*.hxx' '*.tpp' \
            -- ':!external/**'
    )
else
    merge_base="$(git merge-base HEAD "$base_ref")"
    echo "  Mode: changed files since $base_ref"
    while IFS=$'\t' read -r status first second; do
        case "$status" in
        D) ;;
        R*) [[ -f "$second" ]] && files+=("$second") ;;
        *) [[ -f "$first" ]] && files+=("$first") ;;
        esac
    done < <(git diff --name-status --find-renames "$merge_base...HEAD")

    # Filter to C++ files only (changed mode may include non-source files)
    # and exclude anything under external/
    cpp_files=()
    for f in "${files[@]}"; do
        [[ "$f" == external/* ]] && continue
        [[ "$f" =~ \.(cpp|cxx|cc|c|h|hpp|hxx|tpp)$ ]] && cpp_files+=("$f")
    done
    files=("${cpp_files[@]}")
fi

if [[ ${#files[@]} -eq 0 ]]; then
    echo "  No files to check."
    exit 0
fi

echo "  Files: ${#files[@]}"
clangd-tidy "${files[@]}" -p="$build_dir" --tqdm -j"$jobs"
