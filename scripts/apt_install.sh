#!/usr/bin/env bash
# apt-get update and install that survives a stalled mirror (#1069).
#
# A stall used to hold a CI step for 13 to 34+ minutes (normally 6 to 20 s). A
# plain time limit is wrong though: on a slow mirror an install that is still
# receiving data finishes after 13 minutes, and killing it fails the job. So an
# apt-get call is stopped when nothing happens for APT_STALL seconds (no growth
# of the downloaded files, no new package unpacked) or after APT_CAP seconds,
# and then tried again, up to APT_TRIES times.
#
# Usage: apt_install.sh <package>...
# APT_STALL (default 120), APT_CAP (1200), APT_TRIES (3), APT_POLL (5),
# APT_RETRY_PAUSE (5), APT_PROGRESS_DIRS (the apt directories) are for tuning
# and for the tests.
set -uo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: apt_install.sh <package>..." >&2
    exit 2
fi

# everything below runs as root, so a stuck apt-get can be killed
if [ "$(id -u)" -ne 0 ] && [ -z "${APT_INSTALL_AS_ROOT:-}" ]; then
    APT_INSTALL_AS_ROOT=1 exec sudo --preserve-env "$0" "$@"
fi

stall="${APT_STALL:-120}"
cap="${APT_CAP:-1200}"
tries="${APT_TRIES:-3}"
poll="${APT_POLL:-5}"
options=(-o Acquire::Retries=3 -o Acquire::http::Timeout=20
    -o Acquire::https::Timeout=20 -o DPkg::Lock::Timeout=120)
read -r -a progress_dirs <<< "${APT_PROGRESS_DIRS:-/var/cache/apt/archives /var/lib/apt/lists /var/lib/dpkg/info}"

# a number that changes while apt downloads or unpacks something
progress() {
    find "${progress_dirs[@]}" -type f -printf '%s %p\n' 2>/dev/null |
        awk '{ bytes += $1; files++ } END { print bytes + 0, files + 0 }'
}

# run a command, stop it if it makes no progress or takes too long (status 124)
watched() {
    local pid last now idle=0 elapsed=0
    setsid "$@" &
    pid=$!
    last="$(progress)"
    while kill -0 "$pid" 2> /dev/null; do
        sleep "$poll"
        elapsed="$(awk -v e="$elapsed" -v p="$poll" 'BEGIN { print e + p }')"
        now="$(progress)"
        if [ "$now" != "$last" ]; then
            last="$now"
            idle=0
        else
            idle="$(awk -v i="$idle" -v p="$poll" 'BEGIN { print i + p }')"
        fi
        if awk -v i="$idle" -v s="$stall" -v e="$elapsed" -v c="$cap" \
            'BEGIN { exit !(i >= s || e >= c) }'; then
            echo "::warning::stopping '$*': idle ${idle}s, running ${elapsed}s"
            kill -TERM -- "-$pid" 2> /dev/null
            sleep 2
            kill -KILL -- "-$pid" 2> /dev/null
            wait "$pid" 2> /dev/null
            return 124
        fi
    done
    wait "$pid"
}

for ((try = 1; try <= tries; try++)); do
    if watched apt-get "${options[@]}" update &&
        watched apt-get "${options[@]}" install -y "$@"; then
        exit 0
    fi
    echo "::warning::apt-get try $try of $tries failed or stalled"
    # a killed dpkg may have left a package half configured
    dpkg --configure -a || true
    sleep "${APT_RETRY_PAUSE:-5}"
done

echo "::error::apt-get did not finish in $tries tries"
exit 1
