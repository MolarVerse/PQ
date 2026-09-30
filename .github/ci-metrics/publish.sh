#!/usr/bin/env bash
# Commit new CI timing data and push it to a branch.
#
# usage: publish.sh <branch> [--dry-run]     (run from the repository root)
#
# Used by the CI Metrics workflow. It only ever commits files under
# .github/ci-metrics/data/. Before pushing it rebases onto the current tip of
# <branch> and then checks that the push would change nothing outside that
# directory, so it cannot publish anything else by accident (for example if it
# was started on a feature branch). If the push is rejected because the branch
# moved, it fetches, rebases and tries again, up to 5 attempts.
#
# Environment (all optional):
#   CI_METRICS_GIT_NAME / CI_METRICS_GIT_EMAIL   commit identity
#   CI_METRICS_PUSH_SLEEP                        seconds to sleep per failed attempt
#                                                (default 3; the tests set 0)
set -euo pipefail

branch=${1:?usage: publish.sh <branch> [--dry-run]}
dry_run=false
if [[ ${2:-} == "--dry-run" ]]; then
  dry_run=true
elif [[ -n ${2:-} ]]; then
  echo "unknown argument: $2" >&2
  exit 2
fi

data_dir=.github/ci-metrics/data
name=${CI_METRICS_GIT_NAME:-github-actions[bot]}
email=${CI_METRICS_GIT_EMAIL:-41898282+github-actions[bot]@users.noreply.github.com}
pause=${CI_METRICS_PUSH_SLEEP:-3}
max_attempts=5

git add -- "$data_dir"
if git diff --cached --quiet -- "$data_dir"; then
  echo "No new CI timing data."
  exit 0
fi

added=$(git diff --cached --numstat -- "$data_dir" | awk '{n += $1} END {print n + 0}')
files=$(git diff --cached --name-only -- "$data_dir" | wc -l | tr -d ' ')
git diff --cached --stat -- "$data_dir"

if [[ $dry_run == true ]]; then
  echo "Dry run: would commit $added records in $files file(s) to $branch."
  git reset -q -- "$data_dir"
  exit 0
fi

git -c user.name="$name" -c user.email="$email" commit -q \
  -m "ci: update CI timing data (+$added records)" \
  -m "Collected by the CI Metrics workflow (.github/workflows/ci_metrics.yml)." \
  -- "$data_dir"
echo "Committed $added records in $files file(s)."

for attempt in $(seq 1 "$max_attempts"); do
  git fetch -q origin "$branch"

  if ! git -c user.name="$name" -c user.email="$email" rebase -q "origin/$branch"; then
    git rebase --abort || true
    echo "Could not rebase onto origin/$branch (conflicting data changes); not pushing." >&2
    exit 1
  fi

  outside=$(git diff --name-only "origin/$branch" HEAD | grep -v "^$data_dir/" || true)
  if [[ -n $outside ]]; then
    echo "Refusing to push: this would change files outside $data_dir:" >&2
    echo "$outside" >&2
    exit 1
  fi

  if git push -q origin "HEAD:$branch"; then
    echo "Pushed to $branch (attempt $attempt)."
    exit 0
  fi

  echo "Push to $branch was rejected (attempt $attempt of $max_attempts); retrying." >&2
  sleep $((attempt * pause))
done

echo "Could not push to $branch after $max_attempts attempts." >&2
exit 1
