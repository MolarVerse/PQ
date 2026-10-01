#!/usr/bin/env python3
"""Collect finished CI job timings from the GitHub Actions API into JSONL.

Writes one record per job to data/<ISO year>-W<week>.jsonl following
schema.json (field meanings and derivation rules: SCHEMA.md), plus one
`build-analysis` record per job of the workflows in config.json
(`build_analysis_workflows`) that uploaded a build-timings artifact.

Stateless and idempotent: a job whose job_id is already in its shard is never
written twice, so the script can be re-run over overlapping windows (a daily
incremental run, or a one-off backfill). Standard library only; the API is
reached through the `gh` CLI, which uses GH_TOKEN / GITHUB_TOKEN in Actions and
the local login elsewhere.

Examples:
  collect.py                       # last 3 days (incremental)
  collect.py --since-days 90       # backfill, resumable: re-run to continue
  collect.py --since-days 90 --rate-limit-wait 90  # ... waiting out rate limits
  collect.py --run-id 36693119228  # one specific run
  collect.py --dry-run             # show what would be written

Exit status: 0 ok, 1 some runs could not be collected, 2 stopped early because
the API rate limit was hit (data collected so far is still written).
"""

import argparse
import io
import json
import re
import subprocess
import sys
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

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

# Build-analysis artifacts (see SCHEMA.md): written by .github/ci-metrics/summarise_build.py
# inside the jobs. Artifacts of pull request runs are produced by code from the pull
# request, so nothing from them is copied unchecked: sanitise_summary() rebuilds
# every field from a whitelist and the join keys come from the API.
ARTIFACT_PREFIX = "build-timings-"
SUMMARY_NAME = "build-analysis.json"
MAX_ARTIFACT_BYTES = 1_000_000
MAX_TARGET_CHARS = 300
MAX_SLOWEST = 20
MAX_COUNTERS = 200
STEP_KINDS = ("compile", "archive", "link", "other")
COUNTER_NAME = re.compile(r"^[a-z0-9_]{1,64}$")


class ApiError(Exception):
    def __init__(self, message, status=None, rate_limited=False):
        super().__init__(message)
        self.status = status
        self.rate_limited = rate_limited


RATE_LIMIT_FIRST_PAUSE = 60
RATE_LIMIT_MAX_PAUSE = 600


class GhApi:
    """Read-only GitHub API access through `gh api`, with retries.

    Transient failures are retried a few times with a short backoff. A rate
    limit stops the call immediately unless `rate_limit_wait` (seconds, per
    call) is set: then the call is retried after 1, 2, 4, 8, 10, 10 ... minutes
    until that budget is used up. It deliberately does not compute when the
    limit resets: neither `gh api rate_limit` nor the X-RateLimit-* headers
    predicted the real reset reliably during the first backfill.
    """

    def __init__(
        self,
        repo,
        timeout=60,
        retries=3,
        rate_limit_wait=0,
        notice=None,
        run=subprocess.run,
        sleep=time.sleep,
    ):
        self.repo = repo
        self.timeout = timeout
        self.retries = retries
        self.rate_limit_wait = rate_limit_wait
        self._notice = notice or (lambda _message: None)
        self._run = run
        self._sleep = sleep

    def _call(self, args, binary=False):
        failures = 0
        waited = 0
        pause = RATE_LIMIT_FIRST_PAUSE
        while True:
            try:
                proc = self._run(
                    ["gh", "api", *args],
                    capture_output=True,
                    text=not binary,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                error = ApiError("timed out")
            else:
                if proc.returncode == 0:
                    return proc.stdout
                stderr = proc.stderr if isinstance(proc.stderr, str) else proc.stderr.decode(errors="replace")
                error = _classify_error(stderr)

            if error.rate_limited:
                delay = min(pause, self.rate_limit_wait - waited)
                if delay <= 0:
                    raise error
                self._notice(f"rate limited ({short_message(error)}); retrying in {delay}s")
                self._sleep(delay)
                waited += delay
                pause = min(pause * 2, RATE_LIMIT_MAX_PAUSE)
                continue

            failures += 1
            if failures >= self.retries or error.status == 404:
                raise error
            self._sleep(2 ** (failures - 1))

    def get_json(self, path):
        return json.loads(self._call([f"repos/{self.repo}/{path}"]))

    def get_text(self, path):
        return self._call(["--allow-escape-sequences", f"repos/{self.repo}/{path}"])

    def get_bytes(self, path):
        return self._call([f"repos/{self.repo}/{path}"], binary=True)


def short_message(error):
    """The error text without gh's prefix and GitHub's support boilerplate."""
    text = str(error).strip().removeprefix("gh:").strip()
    return text.split(" If you reach out")[0].strip()


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


class PrResolver:
    """Finds the pull request a pull_request run belongs to.

    1. `commits/{sha}/pulls`, keeping the PR whose head branch is the run's.
       That endpoint returns the PR that *merged* a commit, not the PR whose
       head it is, so it misses runs whose head commit is a merge commit of
       another PR (a stacked PR), and also others; all 450 historical misses of
       the first backfill were of this kind.
    2. Fallback: the PRs from that branch (`pulls?head=owner:branch`), keeping
       the most recently created one that was open when the run was created. A
       branch name can be reused by several PRs over time, hence the time check.

    PRs from forks are not found (their head is `fork-owner:branch`).
    """

    def __init__(self, api, owner):
        self._api = api
        self._owner = owner
        self._lock = threading.Lock()
        self._by_commit = {}
        self._by_branch = {}

    def resolve(self, sha, branch, created_at):
        key = (sha, branch)
        with self._lock:
            if key in self._by_commit:
                return self._by_commit[key]
        number = self._from_commit(sha, branch)
        if number is None:
            number = self._from_branch(branch, created_at)
        with self._lock:
            self._by_commit[key] = number
        return number

    def _from_commit(self, sha, branch):
        try:
            pulls = self._api.get_json(f"commits/{sha}/pulls?per_page=100")
        except ApiError as error:
            if error.status != 404:  # 404: commit is gone (force-push)
                raise
            pulls = []
        return next((p["number"] for p in pulls if p["head"]["ref"] == branch), None)

    def _from_branch(self, branch, created_at):
        with self._lock:
            pulls = self._by_branch.get(branch)
        if pulls is None:
            head = quote(f"{self._owner}:{branch}", safe=":/")
            try:
                pulls = self._api.get_json(f"pulls?state=all&head={head}&per_page=100")
            except ApiError as error:
                if error.status != 404:
                    raise
                pulls = []
            with self._lock:
                self._by_branch[branch] = pulls

        when = parse_time(created_at)
        open_then = [
            p
            for p in pulls
            if parse_time(p["created_at"]) <= when
            and (p.get("closed_at") is None or parse_time(p["closed_at"]) >= when)
        ]
        if not open_then:
            return None
        return max(open_then, key=lambda p: p["created_at"])["number"]


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
            pr_number = resolve_pr(run["head_sha"], run["head_branch"], run["created_at"])

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


def _integer(value, what, *, low=0, high=10**12):
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise ValueError(f"{what}: expected an integer from {low} to {high}")
    return value


def _number(value, what, *, high=10**7, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what}: expected a number")
    if not 0 <= value <= high:  # also false for NaN
        raise ValueError(f"{what}: out of range")
    return round(float(value), 4)


def _flag(value, what):
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{what}: expected true, false or null")
    return value


def _object(value, what):
    if not isinstance(value, dict):
        raise ValueError(f"{what}: expected an object")
    return value


def _sanitise_ninja(raw):
    raw = _object(raw, "ninja")
    version = raw.get("log_version")
    ninja = {
        "log_version": None if version is None else _integer(version, "ninja.log_version", high=1000),
        "complete": _flag(raw.get("complete"), "ninja.complete"),
        "steps": _integer(raw.get("steps"), "ninja.steps"),
    }
    if "error" in raw:
        ninja["error"] = _text(raw["error"], "ninja.error")
        return ninja
    ninja["wall_s"] = _number(raw.get("wall_s"), "ninja.wall_s")
    ninja["cpu_s"] = _number(raw.get("cpu_s"), "ninja.cpu_s")
    ninja["parallelism"] = _number(raw.get("parallelism"), "ninja.parallelism", high=10**4, nullable=True)
    ninja["tail_after_compile_s"] = _number(
        raw.get("tail_after_compile_s"), "ninja.tail_after_compile_s", nullable=True
    )
    by_kind = _object(raw.get("by_kind"), "ninja.by_kind")
    if set(by_kind) != set(STEP_KINDS):
        raise ValueError("ninja.by_kind: unexpected kinds")
    ninja["by_kind"] = {}
    for kind in STEP_KINDS:
        entry = _object(by_kind[kind], f"ninja.by_kind.{kind}")
        ninja["by_kind"][kind] = {
            "steps": _integer(entry.get("steps"), f"ninja.by_kind.{kind}.steps"),
            "cpu_s": _number(entry.get("cpu_s"), f"ninja.by_kind.{kind}.cpu_s"),
        }
    slowest = raw.get("slowest")
    if not isinstance(slowest, list) or len(slowest) > MAX_SLOWEST:
        raise ValueError(f"ninja.slowest: expected a list of at most {MAX_SLOWEST}")
    ninja["slowest"] = []
    for number, entry in enumerate(slowest):
        entry = _object(entry, f"ninja.slowest[{number}]")
        if entry.get("kind") not in STEP_KINDS:
            raise ValueError(f"ninja.slowest[{number}].kind: unknown")
        ninja["slowest"].append(
            {
                "target": _text(entry.get("target"), f"ninja.slowest[{number}].target", limit=MAX_TARGET_CHARS),
                "kind": entry["kind"],
                "seconds": _number(entry.get("seconds"), f"ninja.slowest[{number}].seconds"),
            }
        )
    if "ignored_lines" in raw:
        ninja["ignored_lines"] = _integer(raw["ignored_lines"], "ninja.ignored_lines")
    return ninja


def _text(value, what, *, limit=200):
    if not isinstance(value, str) or not value or len(value) > limit or not value.isprintable():
        raise ValueError(f"{what}: expected short printable text")
    return value


def _sanitise_ccache(raw):
    raw = _object(raw, "ccache")
    counters = _object(raw.get("counters"), "ccache.counters")
    if len(counters) > MAX_COUNTERS:
        raise ValueError("ccache.counters: too many entries")
    clean = {}
    for name, value in counters.items():
        if not COUNTER_NAME.match(name):
            raise ValueError(f"ccache.counters: bad name {name!r}")
        clean[name] = _integer(value, f"ccache.counters.{name}")
    return {
        "hits": _integer(raw.get("hits"), "ccache.hits"),
        "misses": _integer(raw.get("misses"), "ccache.misses"),
        "hit_rate": _number(raw.get("hit_rate"), "ccache.hit_rate", high=1, nullable=True),
        "counters": dict(sorted(clean.items())),
    }


def sanitise_summary(raw):
    """Validate the JSON a job uploaded and return only whitelisted parts.

    Returns {"run_id", "run_attempt", "job_id", "ninja", "ccache"}; raises
    ValueError on anything unexpected.
    """
    raw = _object(raw, "summary")
    if raw.get("schema_version") != SCHEMA_VERSION or raw.get("kind") != "build-analysis":
        raise ValueError("not a version 1 build-analysis summary")
    return {
        "run_id": _integer(raw.get("run_id"), "run_id", high=10**15),
        "run_attempt": _integer(raw.get("run_attempt"), "run_attempt", low=1, high=10**4),
        "job_id": _integer(raw.get("job_id"), "job_id", high=10**15),
        "ninja": None if raw.get("ninja") is None else _sanitise_ninja(raw["ninja"]),
        "ccache": None if raw.get("ccache") is None else _sanitise_ccache(raw["ccache"]),
    }


def read_summary(blob):
    """The parsed build-analysis.json inside an artifact zip."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        info = archive.getinfo(SUMMARY_NAME)
        if info.file_size > MAX_ARTIFACT_BYTES:
            raise ValueError("summary too large")
        with archive.open(info) as handle:
            return json.loads(handle.read(MAX_ARTIFACT_BYTES + 1))


def build_analysis_records(api, run, job_records):
    """One `build-analysis` record per job of the run that uploaded a summary.

    Join keys come from the job records (which come from the API), never from the
    artifact; an artifact only counts if it names the same run, attempt and job.
    Returns (records, dropped counter). API errors other than an expired
    artifact propagate, so that the whole run is retried later.
    """
    dropped = Counter()
    by_job = {record["job_id"]: record for record in job_records}
    found = {}
    for artifact in paged(api, f"actions/runs/{run['id']}/artifacts", "artifacts"):
        if not artifact["name"].startswith(ARTIFACT_PREFIX):
            continue
        if artifact.get("expired"):
            dropped["build-analysis artifact expired"] += 1
            continue
        if artifact.get("size_in_bytes", 0) > MAX_ARTIFACT_BYTES:
            dropped["build-analysis artifact too large"] += 1
            continue
        try:
            blob = api.get_bytes(f"actions/artifacts/{artifact['id']}/zip")
        except ApiError as error:
            if error.status in (404, 410):
                dropped["build-analysis artifact expired"] += 1
                continue
            raise
        try:
            summary = sanitise_summary(read_summary(blob))
        except (ValueError, KeyError, zipfile.BadZipFile, json.JSONDecodeError):
            dropped["invalid build-analysis artifact"] += 1
            continue
        job = by_job.get(summary["job_id"])
        if job is None or summary["run_id"] != run["id"] or summary["run_attempt"] != job["run_attempt"]:
            dropped["build-analysis artifact without a matching job"] += 1
            continue
        found.setdefault(job["job_id"], (job, summary))

    records = []
    for job, summary in found.values():
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "build-analysis",
                "workflow": job["workflow"],
                "run_id": job["run_id"],
                "run_attempt": job["run_attempt"],
                "event": job["event"],
                "branch": job["branch"],
                "head_sha": job["head_sha"],
                "job_id": job["job_id"],
                "job": job["job"],
                "created_at": job["created_at"],
                "conclusion": job["conclusion"],
                "ninja": summary["ninja"],
                "ccache": summary["ccache"],
            }
        )
    return records, dropped


class ShardIndex:
    """Which jobs, build analyses and run attempts each shard already holds (lazy)."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self._jobs = {}
        self._analyses = {}
        self._attempts = {}

    def _load(self, shard):
        if shard in self._jobs:
            return
        jobs, analyses, attempts = set(), set(), set()
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
                if record.get("kind", "job") == "job":
                    jobs.add(record["job_id"])
                    attempts.add((record["run_id"], record["run_attempt"]))
                elif record["kind"] == "build-analysis":
                    analyses.add(record["job_id"])
        self._jobs[shard], self._analyses[shard], self._attempts[shard] = jobs, analyses, attempts

    def has_attempt(self, shard, run_id, attempt):
        self._load(shard)
        return (run_id, attempt) in self._attempts[shard]

    def has_record(self, shard, record):
        self._load(shard)
        known = self._jobs if record["kind"] == "job" else self._analyses
        return record["job_id"] in known[shard]

    def add(self, shard, record):
        self._load(shard)
        if record["kind"] == "job":
            self._jobs[shard].add(record["job_id"])
            self._attempts[shard].add((record["run_id"], record["run_attempt"]))
        else:
            self._analyses[shard].add(record["job_id"])


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
        self.analyses_written = 0
        self.per_shard = Counter()
        self.dropped = Counter()
        self.errors = []
        self.rate_limited = False
        self.rate_limit_message = None  # the first one seen, for the summary

    def exit_code(self):
        if self.rate_limited:
            return 2
        return 1 if self.errors else 0


def collect(api, config, data_dir, runs, *, workers=WORKERS, dry_run=False, max_runs=None, out=print):
    result = Result()
    index = ShardIndex(data_dir)
    signatures = config["infra_failure_signatures"]
    analysis_workflows = set(config.get("build_analysis_workflows", []))

    resolve_pr = PrResolver(api, config["repo"].split("/")[0]).resolve

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
            if records and run["name"] in analysis_workflows:
                analyses, analysis_dropped = build_analysis_records(api, run, records)
                records = records + analyses
                dropped.update(analysis_dropped)
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
                    if result.rate_limit_message is None:
                        result.rate_limit_message = short_message(error)
                else:
                    result.errors.append(f"run {run['id']}: {error}")
                continue
            if records is None:  # skipped after a rate limit
                continue
            result.runs_collected += 1
            result.dropped.update(dropped)
            shard = shard_name(run["created_at"])
            fresh = [r for r in records if not index.has_record(shard, r)]
            fresh.sort(key=lambda r: (r["created_at"], r["job_id"], r["kind"] != "job"))
            if fresh and not dry_run:
                append_records(Path(data_dir) / shard, fresh)
            for record in fresh:
                index.add(shard, record)
            result.records_written += len(fresh)
            result.analyses_written += sum(1 for r in fresh if r["kind"] != "job")
            result.per_shard[shard] += len(fresh)
            if number % 25 == 0:
                out(f"  {number}/{len(todo)} runs processed")
    return result


def fix_pr_numbers(api, config, data_dir, *, dry_run=False, out=print):
    """Fill in `pr_number` where it is null on pull_request records.

    The only change ever made to existing data: a one-off repair for records
    written before the branch-based fallback existed. Only that value changes;
    every other byte of the affected lines and all other lines are kept.
    Returns (null records found, records fixed).
    """
    resolver = PrResolver(api, config["repo"].split("/")[0])
    found = fixed = 0
    for path in sorted(Path(data_dir).glob("*.jsonl")):
        lines, changed = [], 0
        for line in path.read_text().splitlines():
            record = json.loads(line) if line.strip() else None
            if (
                record
                and record.get("kind", "job") == "job"
                and record["event"] == "pull_request"
                and record["pr_number"] is None
            ):
                found += 1
                number = resolver.resolve(record["head_sha"], record["branch"], record["created_at"])
                if number is not None:
                    record["pr_number"] = number
                    line = json.dumps(record, separators=(",", ":"))
                    changed += 1
            lines.append(line)
        if changed:
            fixed += changed
            out(f"  {path.name}: {changed} fixed")
            if not dry_run:
                temporary = path.with_suffix(".jsonl.tmp")
                temporary.write_text("\n".join(lines) + "\n")
                temporary.replace(path)
    verb = "would fix" if dry_run else "fixed"
    out(f"{found} pull_request records without pr_number; {verb} {fixed}, {found - fixed} still unresolved")
    return found, fixed


def summarize(result, dry_run, out=print):
    verb = "would write" if dry_run else "wrote"
    out(
        f"{result.runs_seen} runs listed, {result.runs_already_collected} already collected, "
        f"{result.runs_collected} processed; {verb} {result.records_written} records"
        + (f" (incl. {result.analyses_written} build analyses)" if result.analyses_written else "")
    )
    for shard, count in sorted(result.per_shard.items()):
        out(f"  {shard}: +{count}")
    if result.dropped:
        out("dropped jobs: " + ", ".join(f"{k}={v}" for k, v in sorted(result.dropped.items())))
    if result.rate_limited:
        detail = f" ({result.rate_limit_message})" if result.rate_limit_message else ""
        out(
            f"STOPPED EARLY: API rate limit reached{detail}; "
            "re-run later to continue, or use --rate-limit-wait to wait and retry."
        )
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
    parser.add_argument(
        "--rate-limit-wait",
        type=int,
        default=0,
        metavar="MINUTES",
        help="on a rate limit, retry each call with backoff (1, 2, 4, 8, 10, 10 ... min) "
        "for up to this many minutes instead of stopping (default 0: stop)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fix-pr-numbers",
        action="store_true",
        help="instead of collecting, fill in null pr_number values of pull_request records "
        "already in the data directory (a one-off repair; combine with --dry-run to preview)",
    )
    args = parser.parse_args(argv)
    if args.rate_limit_wait < 0:
        parser.error("--rate-limit-wait must not be negative")

    config = json.loads(Path(args.config).read_text())
    api = api or GhApi(
        config["repo"],
        rate_limit_wait=args.rate_limit_wait * 60,
        notice=lambda message: out(f"  {message}"),
    )
    today = today or datetime.now(timezone.utc).date()

    if args.fix_pr_numbers:
        try:
            fix_pr_numbers(api, config, args.data_dir, dry_run=args.dry_run, out=out)
        except ApiError as error:
            message = short_message(error) if error.rate_limited else error
            out(f"ERROR could not look up pull requests: {message}")
            return 2 if error.rate_limited else 1
        return 0

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
        message = short_message(error) if error.rate_limited else error
        out(f"ERROR could not list runs: {message}")
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
