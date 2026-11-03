#!/usr/bin/env bash
set -euo pipefail

ENFORCED_PATHS=(
    "apps/"
    "src/"
    "include/"
)

LOG=$1
if [[ ! -f "$LOG" ]]; then
    echo "warning log not found at $LOG" >&2
    exit 1
fi

fail=0
for p in "${ENFORCED_PATHS[@]}"; do
    hits=$(grep "$p" "$LOG" || true)
    if [[ -n "$hits" ]]; then
        echo "$hits"
        fail=1
    fi
done
exit $fail
