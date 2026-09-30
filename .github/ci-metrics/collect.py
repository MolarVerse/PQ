#!/usr/bin/env python3
"""Collect finished CI job timings from the GitHub Actions API into JSONL.

Writes one record per job to data/<ISO year>-W<week>.jsonl following
schema.json (field meanings and derivation rules: SCHEMA.md).

Stateless and idempotent: a job whose job_id is already in its shard is never
written twice, so the script can be re-run over overlapping windows (a daily
incremental run, or a one-off backfill). Standard library only; the API is
reached through the `gh` CLI, which uses GH_TOKEN / GITHUB_TOKEN in Actions and
the local login elsewhere.

Examples:
  collect.py                       # last 3 days (incremental)
  collect.py --since-days 90       # backfill, resumable: re-run to continue
  collect.py --run-id 36693119228  # one specific run
  collect.py --dry-run             # show what would be written

Exit status: 0 ok, 1 some runs could not be collected, 2 stopped early because
the API rate limit was hit (data collected so far is still written).
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1

EVENTS = {"pull_request", "push", "schedule", "workflow_dispatch"}
JOB_CONCLUSIONS = {
    "success",
    "failure",
    "cancelled",
    "timed_out",
    "neutral",
    "action_required",
}
STEP_CONCLUSIONS = {"success", "failure", "cancelled"}

# eigen_cache_hit is derived from these two step names (see SCHEMA.md).
EIGEN_CACHE_STEP = "Cache Eigen source"
EIGEN_CLONE_STEP = "Clone Eigen (cache miss)"

WORKERS = 8


class ApiError(Exception):
    def __init__(self, message, status=None, rate_limited=False):
        super().__init__(message)
        self.status = status
        self.rate_limited = rate_limited


class GhApi:
    """Read-only GitHub API access through `gh api`, with retries."""

    def __init__(
        self,
        repo,
        timeout=60,
        retries=3,
        run=subprocess.run,
        sleep=time.sleep,
    ):
        self.repo = repo
        self.timeout = timeout
        self.retries = retries
        self._run = run
        self._sleep = sleep

    def _call(self, args):
        for attempt in range(self.retries):
            try:
                proc = self._run(
                    ["gh", "api", *args],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                error = ApiError("timed out")
            else:
                if proc.returncode == 0:
                    return proc.stdout
                error = _classify_error(proc.stderr)

            final = attempt == self.retries - 1
            if final or error.status == 404 or error.rate_limited:
                raise error
            self._sleep(2**attempt)
        raise AssertionError("unreachable")

    def get_json(self, path):
        return json.loads(self._call([f"repos/{self.repo}/{path}"]))

    def get_text(self, path):
        return self._call(["--allow-escape-sequences", f"repos/{self.repo}/{path}"])


def _classify_error(stderr):
    text = stderr.strip()
    status = re.search(r"HTTP (\d{3})", text)
    return ApiError(
        text or "gh api failed",
        status=int(status.group(1)) if status else None,
        rate_limited="rate limit" in text.lower(),
    )


def paged(api, path, key):
    """Yield every item of a paginated list endpoint."""
    separator = "&" if "?" in path else "?"
    page = 1
    while True:
        data = api.get_json(f"{path}{separator}per_page=100&page={page}")
        items = data[key]
        yield from items
        if len(items) < 100:
            return
        page += 1


def parse_time(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def seconds_between(start, end):
    return max(0, int((parse_time(end) - parse_time(start)).total_seconds()))


def shard_name(created_at):
    """Shard of a run: the ISO year and week (UTC) of its creation time."""
    iso = parse_time(created_at).isocalendar()
    return f"{iso.year}-W{iso.week:02d}.jsonl"


def day_windows(today, days):
    """One `created` range per day, oldest first.

    The runs endpoint returns at most 1000 results per query, so a long
    backfill has to be split into small windows.
    """
    windows = []
    for offset in range(days - 1, -1, -1):
        day = (today - timedelta(days=offset)).isoformat()
        windows.append(f"{day}T00:00:00Z..{day}T23:59:59Z")
    return windows


def list_runs(api, windows, workflows):
    """Completed runs of the recorded workflows created in the windows."""
    runs = []
    for window in windows:
        path = f"actions/runs?status=completed&created={window}"
        runs.extend(r for r in paged(api, path, "workflow_runs") if r["name"] in workflows)
    return runs


def eigen_cache_hit(raw_steps):
    """True/False if the Eigen cache was restored/missed, None if unknown.

    The clone step is skipped on a cache hit, but it is also "skipped" when
    the job failed before reaching it, so a hit needs the preceding cache step
    to have succeeded.
    """
    by_name = {step["name"]: step for step in raw_steps}
    cache = by_name.get(EIGEN_CACHE_STEP)
    clone = by_name.get(EIGEN_CLONE_STEP)
    if cache is None or clone is None or cache.get("conclusion") != "success":
        return None
    if clone.get("conclusion") == "skipped":
        return True
    if clone.get("conclusion") in ("success", "failure"):
        return False
    return None


def build_records(run, jobs, resolve_pr, fetch_log, signatures):
    """Turn the jobs of one run into records; also count what was dropped."""
    dropped = Counter()
    if run["event"] not in EVENTS or not run.get("head_branch"):
        dropped["unsupported run"] += len(jobs)
        return [], dropped

    pr_number = None
    records = []
    for job in jobs:
        if not job.get("started_at") or not job.get("completed_at"):
            dropped["no start/end"] += 1
            continue
        conclusion = job.get("conclusion")
        if conclusion == "skipped":
            dropped["skipped"] += 1
            continue
        if conclusion not in JOB_CONCLUSIONS:
            dropped[f"conclusion {conclusion}"] += 1
            continue

        if run["event"] == "pull_request" and pr_number is None:
            pr_number = resolve_pr(run["head_sha"], run["head_branch"])

        raw_steps = job.get("steps") or []
        steps = [
            {
                "name": step["name"],
                "seconds": seconds_between(step["started_at"], step["completed_at"]),
                "conclusion": step["conclusion"],
            }
            for step in raw_steps
            if step.get("started_at")
            and step.get("completed_at")
            and step.get("conclusion") in STEP_CONCLUSIONS
        ]

        infra = None
        if conclusion == "failure":
            log = fetch_log(job["id"])
            if log is not None:
                infra = any(signature in log for signature in signatures)

        labels = job.get("labels") or []
        created = job.get("created_at") or job["started_at"]
        attempt = job.get("run_attempt") or run["run_attempt"]
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "job",
                "workflow": run["name"],
                "run_id": run["id"],
                "run_number": run["run_number"],
                "run_attempt": attempt,
                "event": run["event"],
                "branch": run["head_branch"],
                "head_sha": run["head_sha"],
                "pr_number": pr_number,
                "job_id": job["id"],
                "job": job["name"],
                "runner_labels": labels,
                "arch": "arm64" if any("arm" in label.lower() for label in labels) else "x86_64",
                "created_at": created,
                "started_at": job["started_at"],
                "completed_at": job["completed_at"],
                "queue_s": seconds_between(created, job["started_at"]),
                "duration_s": seconds_between(job["started_at"], job["completed_at"]),
                "conclusion": conclusion,
                "steps": steps,
                "flags": {
                    "eigen_cache_hit": eigen_cache_hit(raw_steps),
                    "is_rerun": attempt > 1,
                    "infra_failure": infra,
                },
            }
        )
    return records, dropped


class ShardIndex:
    """Which jobs and run attempts each shard already contains (lazy)."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self._jobs = {}
        self._attempts = {}

    def _load(self, shard):
        if shard in self._jobs:
            return
        jobs, attempts = set(), set()
        path = self.data_dir / shard
        if path.exists():
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    # Carrying on would risk writing duplicates.
                    raise ValueError(f"{path}:{number}: not valid JSON ({error})") from error
                jobs.add(record["job_id"])
                attempts.add((record["run_id"], record["run_attempt"]))
        self._jobs[shard], self._attempts[shard] = jobs, attempts

    def has_attempt(self, shard, run_id, attempt):
        self._load(shard)
        return (run_id, attempt) in self._attempts[shard]

    def has_job(self, shard, job_id):
        self._load(shard)
        return job_id in self._jobs[shard]

    def add(self, shard, record):
        self._load(shard)
        self._jobs[shard].add(record["job_id"])
        self._attempts[shard].add((record["run_id"], record["run_attempt"]))


def append_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    with path.open("a") as handle:
        if needs_newline:
            handle.write("\n")
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


class Result:
    def __init__(self):
        self.runs_seen = 0
        self.runs_already_collected = 0
        self.runs_collected = 0
        self.records_written = 0
        self.per_shard = Counter()
        self.dropped = Counter()
        self.errors = []
        self.rate_limited = False

    def exit_code(self):
        if self.rate_limited:
            return 2
        return 1 if self.errors else 0


def collect(api, config, data_dir, runs, *, workers=WORKERS, dry_run=False, max_runs=None, out=print):
    result = Result()
    index = ShardIndex(data_dir)
    signatures = config["infra_failure_signatures"]

    pr_cache, pr_lock = {}, threading.Lock()

    def resolve_pr(sha, branch):
        key = (sha, branch)
        with pr_lock:
            if key in pr_cache:
                return pr_cache[key]
        try:
            pulls = api.get_json(f"commits/{sha}/pulls?per_page=100")
        except ApiError as error:
            if error.status != 404:  # 404: commit is gone (force-push)
                raise
            pulls = []
        number = next((p["number"] for p in pulls if p["head"]["ref"] == branch), None)
        with pr_lock:
            pr_cache[key] = number
        return number

    def fetch_log(job_id):
        try:
            return api.get_text(f"actions/jobs/{job_id}/logs")
        except ApiError as error:
            if error.status in (404, 410):  # logs expire after 90 days
                return None
            raise

    result.runs_seen = len(runs)
    todo = []
    for run in sorted(runs, key=lambda r: r["created_at"], reverse=True):
        if index.has_attempt(shard_name(run["created_at"]), run["id"], run["run_attempt"]):
            result.runs_already_collected += 1
        else:
            todo.append(run)
    if max_runs is not None and len(todo) > max_runs:
        out(f"Limiting to {max_runs} of {len(todo)} runs; re-run to continue.")
        todo = todo[:max_runs]

    stop = threading.Event()

    def work(run):
        if stop.is_set():
            return run, None, None, None
        try:
            jobs = list(paged(api, f"actions/runs/{run['id']}/jobs?filter=all", "jobs"))
            records, dropped = build_records(run, jobs, resolve_pr, fetch_log, signatures)
            return run, records, dropped, None
        except Exception as error:  # one bad run must not abort the others
            if getattr(error, "rate_limited", False):
                stop.set()
            return run, None, None, error

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, (run, records, dropped, error) in enumerate(pool.map(work, todo), 1):
            if error is not None:
                if getattr(error, "rate_limited", False):
                    result.rate_limited = True
                else:
                    result.errors.append(f"run {run['id']}: {error}")
                continue
            if records is None:  # skipped after a rate limit
                continue
            result.runs_collected += 1
            result.dropped.update(dropped)
            shard = shard_name(run["created_at"])
            fresh = [r for r in records if not index.has_job(shard, r["job_id"])]
            fresh.sort(key=lambda r: (r["created_at"], r["job_id"]))
            if fresh and not dry_run:
                append_records(Path(data_dir) / shard, fresh)
            for record in fresh:
                index.add(shard, record)
            result.records_written += len(fresh)
            result.per_shard[shard] += len(fresh)
            if number % 25 == 0:
                out(f"  {number}/{len(todo)} runs processed")
    return result


def summarize(result, dry_run, out=print):
    verb = "would write" if dry_run else "wrote"
    out(
        f"{result.runs_seen} runs listed, {result.runs_already_collected} already collected, "
        f"{result.runs_collected} processed; {verb} {result.records_written} records"
    )
    for shard, count in sorted(result.per_shard.items()):
        out(f"  {shard}: +{count}")
    if result.dropped:
        out("dropped jobs: " + ", ".join(f"{k}={v}" for k, v in sorted(result.dropped.items())))
    if result.rate_limited:
        out("STOPPED EARLY: API rate limit reached; re-run later to continue.")
    for error in result.errors:
        out(f"ERROR {error}")


def main(argv=None, api=None, today=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--data-dir", default=str(HERE / "data"))
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--since-days", type=int, default=3, help="look back this many days (default 3)")
    which.add_argument("--run-id", type=int, action="append", help="collect this run only (repeatable)")
    parser.add_argument("--max-runs", type=int, help="process at most this many runs per invocation")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    api = api or GhApi(config["repo"])
    today = today or datetime.now(timezone.utc).date()

    try:
        if args.run_id:
            runs = [api.get_json(f"actions/runs/{run_id}") for run_id in args.run_id]
            runs = [r for r in runs if r["status"] == "completed"]
        else:
            if args.since_days < 1:
                parser.error("--since-days must be at least 1")
            windows = day_windows(today, args.since_days)
            runs = list_runs(api, windows, set(config["workflows"]))
    except ApiError as error:
        out(f"ERROR could not list runs: {error}")
        return 2 if error.rate_limited else 1

    result = collect(
        api,
        config,
        args.data_dir,
        runs,
        workers=args.workers,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
        out=out,
    )
    summarize(result, args.dry_run, out)
    return result.exit_code()


if __name__ == "__main__":
    sys.exit(main())
