"""GitHub Actions runs per day, workflow, event and conclusion, and per pull request (optional, needs `gh`).

The runs API returns at most 1,000 results per query with a date filter, so it is queried one day at a time.
GitHub only keeps runs for a limited time, so this reaches back as far as the API still has them.
"""

import subprocess
from collections import defaultdict
from datetime import date, timedelta

JQ = ('.workflow_runs[] | [.id, .name, .event, (.conclusion // ""), .status, .created_at, .run_attempt, '
      '(.pull_requests[0].number // ""), (.head_branch // "")] | @tsv')


class CiError(RuntimeError):
    pass


def gh_api(path):
    try:
        result = subprocess.run(["gh", "api", "--paginate", path, "--jq", JQ], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as error:
        raise CiError(f"cannot run gh: {error}")
    if result.returncode != 0:
        raise CiError(f"gh api failed: {result.stderr.strip()[:200]}")
    return result.stdout


def parse_runs(text):
    runs = []
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) == 9:
            runs.append({"id": fields[0], "workflow": fields[1], "event": fields[2], "conclusion": fields[3] or "none",
                         "status": fields[4], "created": fields[5], "attempt": int(fields[6] or 1),
                         "pr": int(fields[7]) if fields[7] else None, "branch": fields[8]})
    return runs


def fetch_runs(repo, first_day, last_day, api=gh_api, log=print):
    """All runs created from first_day to last_day (inclusive dates), one API query per day."""
    runs, day = {}, first_day
    while day <= last_day:
        for run in parse_runs(api(f"repos/{repo}/actions/runs?per_page=100&created={day}..{day}")):
            runs[run["id"]] = run
        day += timedelta(days=1)
    log(f"{len(runs)} workflow runs from {first_day} to {last_day}")
    return list(runs.values())


def pr_of_run(run, pr_branches):
    """The PR a pull_request run belongs to: the number the API gave, else by head branch.

    GitHub drops the pull request association of a run once its branch is deleted, so the branch named in the
    merge commit ("Merge pull request #N from owner/branch") is used. pr_branches is {branch: [(merge time, pr)]};
    if a branch name was used for several PRs, the one merged next after the run wins.
    """
    if run["pr"]:
        return run["pr"]
    if run["event"] != "pull_request":
        return None
    candidates = sorted((when, pr) for when, pr in (pr_branches or {}).get(run["branch"], []) if when >= run["created"])
    return candidates[0][1] if candidates else None


def aggregate(runs, pr_branches=None):
    """(daily rows, {pr: {"runs", "attempts", "failed"}})."""
    daily = defaultdict(lambda: [0, 0])
    per_pr = defaultdict(lambda: {"runs": 0, "attempts": 0, "failed": 0})
    for run in runs:
        run = dict(run, pr=pr_of_run(run, pr_branches))
        key = (run["created"][:10], run["workflow"], run["event"], run["conclusion"])
        daily[key][0] += 1
        daily[key][1] += run["attempt"]
        if run["pr"]:
            entry = per_pr[run["pr"]]
            entry["runs"] += 1
            entry["attempts"] += run["attempt"]
            entry["failed"] += 1 if run["conclusion"] == "failure" else 0
    rows = [{"date": d, "workflow": w, "event": e, "conclusion": c, "runs": n, "attempts": a}
            for (d, w, e, c), (n, a) in sorted(daily.items())]
    return rows, dict(per_pr)


def window(days, today=None):
    today = today or date.today()
    return today - timedelta(days=days - 1), today
