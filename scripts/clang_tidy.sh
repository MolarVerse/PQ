#!/usr/bin/env bash
set -euo pipefail

# scripts/clangd_tidy.sh
#
# Runs clangd-tidy either on all source files, or only on files changed
# relative to HEAD.
#
# Usage:
#   scripts/clangd_tidy.sh          # only changed .cpp files
#   scripts/clangd_tidy.sh --all    # all .cpp files under src/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/.build"

cd "${ROOT_DIR}"

MODE="changed"
if [[ "${1:-}" == "--all" ]]; then
    MODE="all"
fi

if [[ "${MODE}" == "all" ]]; then
    mapfile -t FILES < <(find "${ROOT_DIR}/src" -name '*.cpp' ! -name 'moc_*.cpp')
else
    mapfile -t FILES < <(git diff --name-only --diff-filter=ACMR HEAD -- '*.cpp' ':!*moc_*.cpp')
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No files to lint."
    exit 0
fi

clangd-tidy "${FILES[@]}" -p="${BUILD_DIR}" --tqdm
