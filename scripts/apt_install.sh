#!/usr/bin/env bash
# apt-get update and install with a bounded time per try, apt network timeouts
# and retries: a stalled mirror used to hold a CI step for 13 to 34+ minutes
# (normally 6 to 20 s, #1069).
#
# Usage: apt_install.sh <package>...
# APT_TRY_TIMEOUT (seconds per apt-get call, default 240), APT_TRIES (default 3),
# APT_RETRY_PAUSE (seconds between tries, default 5)
set -uo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: apt_install.sh <package>..." >&2
    exit 2
fi

limit="${APT_TRY_TIMEOUT:-240}"
tries="${APT_TRIES:-3}"
options=(-o Acquire::Retries=3 -o Acquire::http::Timeout=20
    -o Acquire::https::Timeout=20 -o DPkg::Lock::Timeout=120)

for ((try = 1; try <= tries; try++)); do
    if sudo timeout -k 20 "$limit" apt-get "${options[@]}" update &&
        sudo timeout -k 20 "$limit" apt-get "${options[@]}" install -y "$@"; then
        exit 0
    fi
    echo "::warning::apt-get try $try of $tries failed or timed out after ${limit}s"
    # a killed dpkg may have left a package half configured
    sudo dpkg --configure -a || true
    sleep "${APT_RETRY_PAUSE:-5}"
done

echo "::error::apt-get did not finish in $tries tries"
exit 1
