import contextlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import zipfile
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / ".github" / "ci-metrics"

SPEC = importlib.util.spec_from_file_location("ci_metrics_collect", METRICS / "collect.py")
collect = importlib.util.module_from_spec(SPEC)
sys.modules["ci_metrics_collect"] = collect
SPEC.loader.exec_module(collect)

SCHEMA = json.loads((METRICS / "schema.json").read_text())
ANALYSIS_SCHEMA = json.loads((METRICS / "schema-build-analysis.json").read_text())
CONFIG = json.loads((METRICS / "config.json").read_text())


# --- a small JSON Schema validator -----------------------------------------
# jsonschema is not installed where these tests run, so this implements just
# the keywords schema.json uses. MiniValidatorTests keep it honest.


def _is_type(value, name):
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return {
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "null": value is None,
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }[name]


def validation_errors(value, schema=SCHEMA, path="$"):
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        return validation_errors(value, SCHEMA["definitions"][name], path)

    errors = []
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")

    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, name) for name in names):
            return errors + [f"{path}: expected type {names}"]

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: too short")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match {schema['pattern']}")
    if _is_type(value, "number") and value < schema.get("minimum", value):
        errors.append(f"{path}: below minimum")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing {name}")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}: unexpected {name}")
        for name, sub in properties.items():
            if name in value:
                errors += validation_errors(value[name], sub, f"{path}.{name}")
    if isinstance(value, list) and "items" in schema:
        for number, item in enumerate(value):
            errors += validation_errors(item, schema["items"], f"{path}[{number}]")
    return errors


# --- builders for fake API objects ------------------------------------------

BASE = datetime(2026, 9, 30, 7, 11, 5)
SHA = "a" * 40


def at(offset):
    return (BASE + timedelta(seconds=offset)).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_step(name, conclusion="success", start=0, seconds=1):
    return {
        "name": name,
        "conclusion": conclusion,
        "started_at": at(start),
        "completed_at": at(start + seconds),
    }


def default_steps():
    return [
        make_step("Set up job", start=23, seconds=2),
        make_step("Cache Eigen source", start=25, seconds=1),
        make_step("Clone Eigen (cache miss)", "skipped", start=26, seconds=0),
        make_step("Build and Test Project", start=26, seconds=580),
        make_step("Post Cache Eigen source", start=606, seconds=1),
    ]


def make_run(run_id=100, name="BUILD", event="push", branch="dev", attempt=1, created=0):
    return {
        "id": run_id,
        "name": name,
        "event": event,
        "head_branch": branch,
        "head_sha": SHA,
        "run_attempt": attempt,
        "run_number": 7,
        "status": "completed",
        "conclusion": "success",
        "created_at": at(created),
    }


def make_job(
    job_id=1000,
    name="build (ubuntu-24.04, Debug)",
    attempt=1,
    conclusion="success",
    created=20,
    started=23,
    completed=623,
    labels=("ubuntu-24.04",),
    steps=None,
):
    return {
        "id": job_id,
        "name": name,
        "run_attempt": attempt,
        "conclusion": conclusion,
        "created_at": at(created),
        "started_at": at(started) if started is not None else None,
        "completed_at": at(completed) if completed is not None else None,
        "labels": list(labels),
        "steps": default_steps() if steps is None else steps,
    }


def records_for(run, jobs, pr=None, logs=None, signatures=("GitLab is currently",)):
    pr_calls = []

    def resolve_pr(sha, branch, created_at):
        pr_calls.append((sha, branch, created_at))
        return pr

    def fetch_log(job_id):
        return (logs or {}).get(job_id)

    records, dropped = collect.build_records(run, jobs, resolve_pr, fetch_log, list(signatures))
    return records, dropped, pr_calls


class FakeApi:
    """Stands in for GhApi: serves runs, jobs, PRs and logs from dicts."""

    def __init__(
        self, runs=(), jobs=None, pulls=None, logs=None, job_errors=None, branch_prs=None, artifacts=None, blobs=None
    ):
        self.runs = list(runs)
        self.artifacts = artifacts or {}  # run id -> artifact dicts
        self.blobs = blobs or {}  # artifact id -> zip bytes (or an exception to raise)
        self.jobs = jobs or {}
        self.pulls = pulls or {}  # commit sha -> PRs that GitHub associates with the commit
        self.branch_prs = branch_prs or {}  # head branch -> all PRs from that branch
        self.logs = logs or {}
        self.job_errors = job_errors or {}
        self.calls = []

    def get_json(self, path):
        self.calls.append(path)
        parts = urlsplit(path)
        query = parse_qs(parts.query)
        segments = parts.path.split("/")

        if parts.path == "actions/runs":
            start, end = query["created"][0].split("..")
            matching = [r for r in self.runs if start <= r["created_at"] <= end]
            return {"workflow_runs": self._page(matching, query)}
        if segments[:2] == ["actions", "runs"] and segments[-1] == "artifacts":
            return {"artifacts": self._page(self.artifacts.get(int(segments[2]), []), query)}
        if segments[:2] == ["actions", "runs"] and segments[-1] == "jobs":
            run_id = int(segments[2])
            if run_id in self.job_errors:
                raise self.job_errors[run_id]
            return {"jobs": self._page(self.jobs.get(run_id, []), query)}
        if segments[:2] == ["actions", "runs"]:
            return next(r for r in self.runs if r["id"] == int(segments[2]))
        if segments[0] == "commits" and segments[-1] == "pulls":
            result = self.pulls.get(segments[1], [])
            if isinstance(result, Exception):
                raise result
            return result
        if parts.path == "pulls":
            owner, branch = query["head"][0].split(":", 1)
            assert owner == "MolarVerse", owner
            result = self.branch_prs.get(branch, [])
            if isinstance(result, Exception):
                raise result
            return result
        raise AssertionError(f"unexpected API path {path}")

    @staticmethod
    def _page(items, query):
        size, page = int(query["per_page"][0]), int(query["page"][0])
        return items[(page - 1) * size : page * size]

    def get_bytes(self, path):
        self.calls.append(path)
        result = self.blobs[int(path.split("/")[2])]
        if isinstance(result, Exception):
            raise result
        return result

    def get_text(self, path):
        self.calls.append(path)
        result = self.logs.get(int(path.split("/")[2]))
        if isinstance(result, Exception):
            raise result
        return result


def collect_into(api, runs, data_dir, **kwargs):
    kwargs.setdefault("workers", 1)
    return collect.collect(api, CONFIG, data_dir, runs, out=lambda *_: None, **kwargs)


def read_shard(data_dir, shard="2026-W40.jsonl"):
    path = Path(data_dir) / shard
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- tests -------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_recorded_workflows_are_documented(self):
        # Compare with whitespace collapsed: the README wraps long lines.
        readme = " ".join((METRICS / "README.md").read_text().split())
        for name in CONFIG["workflows"]:
            self.assertIn(name, readme)

    def test_config_has_repo_and_signatures(self):
        self.assertEqual("MolarVerse/PQ", CONFIG["repo"])
        self.assertTrue(CONFIG["infra_failure_signatures"])


class ShardNameTests(unittest.TestCase):
    def test_iso_week_of_a_normal_day(self):
        self.assertEqual("2026-W40.jsonl", collect.shard_name("2026-09-30T07:11:05Z"))

    def test_iso_year_differs_from_calendar_year_at_boundaries(self):
        self.assertEqual("2025-W01.jsonl", collect.shard_name("2024-12-30T10:00:00Z"))
        self.assertEqual("2020-W53.jsonl", collect.shard_name("2021-01-03T23:59:59Z"))

    def test_day_windows_are_per_day_and_oldest_first(self):
        windows = collect.day_windows(date(2026, 9, 30), 3)
        self.assertEqual(
            [
                "2026-09-28T00:00:00Z..2026-09-28T23:59:59Z",
                "2026-09-29T00:00:00Z..2026-09-29T23:59:59Z",
                "2026-09-30T00:00:00Z..2026-09-30T23:59:59Z",
            ],
            windows,
        )


class BuildRecordsTests(unittest.TestCase):
    def test_record_fields_and_schema(self):
        records, dropped, _ = records_for(make_run(), [make_job()])
        self.assertEqual(0, sum(dropped.values()))
        self.assertEqual(1, len(records))
        record = records[0]
        self.assertEqual([], validation_errors(record))
        self.assertEqual("BUILD", record["workflow"])
        self.assertEqual("build (ubuntu-24.04, Debug)", record["job"])
        self.assertEqual(3, record["queue_s"])
        self.assertEqual(600, record["duration_s"])
        self.assertEqual("x86_64", record["arch"])
        self.assertFalse(record["flags"]["is_rerun"])

    def test_arch_is_arm64_for_arm_runner_labels(self):
        job = make_job(labels=("ubuntu-24.04-arm",))
        record = records_for(make_run(), [job])[0][0]
        self.assertEqual("arm64", record["arch"])

    def test_skipped_steps_are_omitted_from_the_record(self):
        record = records_for(make_run(), [make_job()])[0][0]
        names = [step["name"] for step in record["steps"]]
        self.assertNotIn("Clone Eigen (cache miss)", names)
        self.assertIn("Build and Test Project", names)
        build = next(s for s in record["steps"] if s["name"] == "Build and Test Project")
        self.assertEqual(580, build["seconds"])

    def test_eigen_cache_hit_is_derived_before_skipped_steps_are_dropped(self):
        record = records_for(make_run(), [make_job()])[0][0]
        self.assertIs(True, record["flags"]["eigen_cache_hit"])

    def test_eigen_cache_miss(self):
        steps = default_steps()
        steps[2] = make_step("Clone Eigen (cache miss)", "success", start=26, seconds=1)
        record = records_for(make_run(), [make_job(steps=steps)])[0][0]
        self.assertIs(False, record["flags"]["eigen_cache_hit"])

    def test_eigen_cache_unknown_without_the_steps(self):
        steps = [make_step("Build and Test Project", start=26, seconds=580)]
        record = records_for(make_run(), [make_job(steps=steps)])[0][0]
        self.assertIsNone(record["flags"]["eigen_cache_hit"])

    def test_eigen_cache_unknown_if_job_died_before_the_cache_step(self):
        # The clone step also reads "skipped" when an earlier step failed.
        steps = default_steps()
        steps[1] = make_step("Cache Eigen source", "failure", start=25, seconds=1)
        record = records_for(make_run(), [make_job(conclusion="failure", steps=steps)])[0][0]
        self.assertIsNone(record["flags"]["eigen_cache_hit"])

    def test_skipped_unfinished_and_unknown_conclusions_are_dropped(self):
        jobs = [
            make_job(1, conclusion="skipped"),
            make_job(2, started=None, completed=None, conclusion=None),
            make_job(3, conclusion="startup_failure"),
            make_job(4),
        ]
        records, dropped, _ = records_for(make_run(), jobs)
        self.assertEqual([4], [r["job_id"] for r in records])
        self.assertEqual(1, dropped["skipped"])
        self.assertEqual(1, dropped["no start/end"])
        self.assertEqual(1, dropped["conclusion startup_failure"])

    def test_cancelled_job_that_started_is_recorded(self):
        records = records_for(make_run(), [make_job(conclusion="cancelled")])[0]
        self.assertEqual("cancelled", records[0]["conclusion"])
        self.assertEqual([], validation_errors(records[0]))

    def test_rerun_attempt_is_flagged(self):
        jobs = [make_job(1, attempt=1), make_job(2, attempt=2)]
        records = records_for(make_run(attempt=2), jobs)[0]
        self.assertEqual([False, True], [r["flags"]["is_rerun"] for r in records])
        self.assertEqual([1, 2], [r["run_attempt"] for r in records])

    def test_pr_number_is_resolved_only_for_pull_request_events(self):
        run = make_run(event="pull_request", branch="feature/x")
        records, _, calls = records_for(run, [make_job(1), make_job(2)], pr=721)
        self.assertEqual([721, 721], [r["pr_number"] for r in records])
        self.assertEqual([(SHA, "feature/x", at(0))], calls)  # once per run, with the run's creation time

        records, _, calls = records_for(make_run(event="push"), [make_job()], pr=721)
        self.assertIsNone(records[0]["pr_number"])
        self.assertEqual([], calls)

    def test_infra_failure_states(self):
        failed = make_job(1, conclusion="failure")
        logs = {1: "fatal: remote error: GitLab is currently unable to handle this request"}
        self.assertIs(True, records_for(make_run(), [failed], logs=logs)[0][0]["flags"]["infra_failure"])

        logs = {1: "error: compilation terminated"}
        self.assertIs(False, records_for(make_run(), [failed], logs=logs)[0][0]["flags"]["infra_failure"])

        # log unavailable (expired)
        self.assertIsNone(records_for(make_run(), [failed], logs={})[0][0]["flags"]["infra_failure"])

    def test_infra_failure_is_null_and_log_not_fetched_for_successful_jobs(self):
        fetched = []
        run, jobs = make_run(), [make_job(1)]

        def fetch_log(job_id):
            fetched.append(job_id)

        records, _ = collect.build_records(run, jobs, lambda *_: None, fetch_log, ["x"])
        self.assertIsNone(records[0]["flags"]["infra_failure"])
        self.assertEqual([], fetched)

    def test_unsupported_event_or_missing_branch_drops_the_run(self):
        for run in (make_run(event="merge_group"), make_run(branch=None)):
            records, dropped, _ = records_for(run, [make_job()])
            self.assertEqual([], records)
            self.assertEqual(1, dropped["unsupported run"])


class MiniValidatorTests(unittest.TestCase):
    """The validator above must reject bad records, or the tests prove nothing."""

    def good(self):
        return json.loads(json.dumps(records_for(make_run(), [make_job()])[0][0]))

    def test_accepts_a_good_record(self):
        self.assertEqual([], validation_errors(self.good()))

    def test_rejects_unknown_event(self):
        record = self.good()
        record["event"] = "cron"
        self.assertTrue(validation_errors(record))

    def test_rejects_missing_and_extra_fields(self):
        record = self.good()
        del record["flags"]
        self.assertTrue(validation_errors(record))
        record = self.good()
        record["surprise"] = 1
        self.assertTrue(validation_errors(record))

    def test_rejects_bad_types_patterns_and_minimums(self):
        for field, value in (
            ("run_id", "100"),
            ("started_at", "2026-09-30 07:11:28"),
            ("head_sha", "abc"),
            ("duration_s", -1),
            ("pr_number", 0),
        ):
            record = self.good()
            record[field] = value
            self.assertTrue(validation_errors(record), field)

    def test_rejects_bad_nested_step_and_flag(self):
        record = self.good()
        record["steps"][0]["conclusion"] = "skipped"
        self.assertTrue(validation_errors(record))
        record = self.good()
        record["flags"]["is_rerun"] = "no"
        self.assertTrue(validation_errors(record))


class GhApiTests(unittest.TestCase):
    def make(self, outcomes, retries=3, **options):
        calls, sleeps = [], []

        def run(command, **_):
            calls.append(command)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        api = collect.GhApi("o/r", retries=retries, run=run, sleep=sleeps.append, **options)
        return api, calls, sleeps

    @staticmethod
    def process(returncode=0, stdout="", stderr=""):
        return subprocess.CompletedProcess([], returncode, stdout, stderr)

    def test_returns_parsed_json_and_targets_the_repo(self):
        api, calls, _ = self.make([self.process(stdout='{"a": 1}')])
        self.assertEqual({"a": 1}, api.get_json("actions/runs/5"))
        self.assertEqual(["gh", "api", "repos/o/r/actions/runs/5"], calls[0])

    def test_retries_transient_failures_with_backoff(self):
        api, calls, sleeps = self.make(
            [self.process(1, stderr="gh: Bad Gateway (HTTP 502)"), self.process(stdout="[]")]
        )
        self.assertEqual([], api.get_json("x"))
        self.assertEqual(2, len(calls))
        self.assertEqual([1], sleeps)

    def test_retries_a_timeout(self):
        api, calls, _ = self.make([subprocess.TimeoutExpired("gh", 60), self.process(stdout="{}")])
        self.assertEqual({}, api.get_json("x"))
        self.assertEqual(2, len(calls))

    def test_gives_up_after_the_retry_budget(self):
        failure = self.process(1, stderr="gh: Server Error (HTTP 500)")
        api, calls, _ = self.make([failure, failure, failure])
        with self.assertRaises(collect.ApiError) as context:
            api.get_json("x")
        self.assertEqual(500, context.exception.status)
        self.assertEqual(3, len(calls))

    def test_does_not_retry_a_404(self):
        api, calls, _ = self.make([self.process(1, stderr="gh: Not Found (HTTP 404)")])
        with self.assertRaises(collect.ApiError) as context:
            api.get_json("x")
        self.assertEqual(404, context.exception.status)
        self.assertEqual(1, len(calls))

    def test_flags_rate_limits_and_does_not_retry_them(self):
        api, calls, _ = self.make([self.process(1, stderr="gh: API rate limit exceeded (HTTP 403)")])
        with self.assertRaises(collect.ApiError) as context:
            api.get_json("x")
        self.assertTrue(context.exception.rate_limited)
        self.assertEqual(1, len(calls))

    def test_logs_are_requested_with_escape_sequences_allowed(self):
        api, calls, _ = self.make([self.process(stdout="log text")])
        self.assertEqual("log text", api.get_text("actions/jobs/9/logs"))
        self.assertIn("--allow-escape-sequences", calls[0])

    RATE_LIMITED = "gh: API rate limit exceeded for user ID 1. If you reach out to GitHub Support (HTTP 403)"

    def test_waits_and_retries_a_rate_limit_with_growing_pauses_when_allowed(self):
        limited = self.process(1, stderr=self.RATE_LIMITED)
        api, calls, sleeps = self.make([limited, limited, limited, self.process(stdout="{}")], rate_limit_wait=3600)
        self.assertEqual({}, api.get_json("x"))
        self.assertEqual(4, len(calls))
        self.assertEqual([60, 120, 240], sleeps)

    def test_rate_limit_pauses_are_capped_at_ten_minutes(self):
        limited = self.process(1, stderr=self.RATE_LIMITED)
        api, _, sleeps = self.make([limited] * 7 + [self.process(stdout="{}")], rate_limit_wait=10**6)
        api.get_json("x")
        self.assertEqual([60, 120, 240, 480, 600, 600, 600], sleeps)

    def test_rate_limit_wait_budget_is_not_exceeded(self):
        limited = self.process(1, stderr=self.RATE_LIMITED)
        api, calls, sleeps = self.make([limited, limited, limited], rate_limit_wait=90)
        with self.assertRaises(collect.ApiError) as context:
            api.get_json("x")
        self.assertTrue(context.exception.rate_limited)
        self.assertEqual([60, 30], sleeps)  # 90 s in total, then it gives up
        self.assertEqual(3, len(calls))

    def test_rate_limit_waits_do_not_use_up_the_transient_retry_budget(self):
        limited = self.process(1, stderr=self.RATE_LIMITED)
        server_error = self.process(1, stderr="gh: Server Error (HTTP 500)")
        outcomes = [limited, server_error, limited, server_error, self.process(stdout="{}")]
        api, calls, _ = self.make(outcomes, rate_limit_wait=3600)
        self.assertEqual({}, api.get_json("x"))
        self.assertEqual(5, len(calls))

    def test_notice_reports_each_wait_with_the_shortened_message(self):
        messages = []
        limited = self.process(1, stderr=self.RATE_LIMITED)
        api, _, _ = self.make([limited, self.process(stdout="{}")], rate_limit_wait=3600, notice=messages.append)
        api.get_json("x")
        self.assertEqual(["rate limited (API rate limit exceeded for user ID 1.); retrying in 60s"], messages)

    def test_no_wait_by_default_even_with_a_notice_callback(self):
        messages = []
        api, calls, sleeps = self.make([self.process(1, stderr=self.RATE_LIMITED)], notice=messages.append)
        with self.assertRaises(collect.ApiError):
            api.get_json("x")
        self.assertEqual(([], [], 1), (messages, sleeps, len(calls)))


def pr_info(number, opened, closed=None):
    """A PR as returned by GET pulls?head=...: times are seconds after BASE."""
    return {"number": number, "created_at": at(opened), "closed_at": at(closed) if closed is not None else None}


def commit_pr(number, head_ref):
    """A PR as returned by GET commits/{sha}/pulls."""
    return {"number": number, "head": {"ref": head_ref}}


class PrResolverTests(unittest.TestCase):
    def resolver(self, **api_options):
        api = FakeApi(**api_options)
        return collect.PrResolver(api, "MolarVerse"), api

    def test_the_commit_lookup_wins_and_no_branch_lookup_is_made(self):
        resolver, api = self.resolver(pulls={SHA: [commit_pr(7, "feature/x")]}, branch_prs={"feature/x": [pr_info(9, 0)]})
        self.assertEqual(7, resolver.resolve(SHA, "feature/x", at(100)))
        self.assertEqual([f"commits/{SHA}/pulls?per_page=100"], api.calls)

    def test_stacked_pr_whose_head_is_the_merge_of_another_pr_falls_back_to_the_branch(self):
        # GitHub returns the PR that MERGED the commit (#727), not the PR whose
        # head it is (#725): exactly what the first dry run hit.
        resolver, _ = self.resolver(
            pulls={SHA: [commit_pr(727, "ci/rate-limits")]},
            branch_prs={"ci/collector": [pr_info(725, 0)]},
        )
        self.assertEqual(725, resolver.resolve(SHA, "ci/collector", at(500)))

    def test_no_pr_from_the_branch_gives_none(self):
        resolver, _ = self.resolver(pulls={}, branch_prs={})
        self.assertIsNone(resolver.resolve(SHA, "never/a/pr", at(100)))

    def test_picks_the_pr_that_was_open_when_the_run_was_created(self):
        # one branch name reused by three PRs over time
        prs = [pr_info(1, 0, 100), pr_info(2, 200, 300), pr_info(3, 400)]
        resolver, _ = self.resolver(branch_prs={"reused": prs})
        self.assertEqual(1, resolver.resolve("a" * 40, "reused", at(50)))
        self.assertEqual(2, resolver.resolve("b" * 40, "reused", at(250)))
        self.assertEqual(3, resolver.resolve("c" * 40, "reused", at(999)))

    def test_a_run_between_two_prs_belongs_to_neither(self):
        resolver, _ = self.resolver(branch_prs={"reused": [pr_info(1, 0, 100), pr_info(2, 200, 300)]})
        self.assertIsNone(resolver.resolve(SHA, "reused", at(150)))

    def test_ignores_prs_created_after_the_run_and_closed_before_it(self):
        resolver, _ = self.resolver(branch_prs={"b": [pr_info(1, 500), pr_info(2, 0, 100)]})
        self.assertIsNone(resolver.resolve(SHA, "b", at(300)))

    def test_the_latest_created_pr_wins_if_several_were_open_at_once(self):
        resolver, _ = self.resolver(branch_prs={"b": [pr_info(1, 0), pr_info(2, 50)]})
        self.assertEqual(2, resolver.resolve(SHA, "b", at(100)))

    def test_a_pr_closed_exactly_when_the_run_was_created_still_counts(self):
        resolver, _ = self.resolver(branch_prs={"b": [pr_info(1, 0, 100)]})
        self.assertEqual(1, resolver.resolve(SHA, "b", at(100)))

    def test_the_branch_list_is_fetched_once_per_branch_and_results_are_cached(self):
        resolver, api = self.resolver(branch_prs={"b": [pr_info(1, 0)]})
        for sha in ("a" * 40, "b" * 40, "a" * 40):
            self.assertEqual(1, resolver.resolve(sha, "b", at(10)))
        self.assertEqual(1, sum(1 for c in api.calls if c.startswith("pulls?")))
        self.assertEqual(2, sum(1 for c in api.calls if c.startswith("commits/")))  # 2 distinct shas

    def test_branch_names_are_url_encoded(self):
        # "+" would otherwise be read as a space
        branch = "Input-parser-rework-+-keys/issue-574"
        resolver, api = self.resolver(branch_prs={branch: [pr_info(5, 0)]})
        self.assertEqual(5, resolver.resolve(SHA, branch, at(10)))
        call = next(c for c in api.calls if c.startswith("pulls?"))
        self.assertIn("head=MolarVerse:Input-parser-rework-%2B-keys/issue-574", call)

    def test_a_404_from_either_lookup_means_no_pr(self):
        gone = collect.ApiError("gone", status=404)
        resolver, _ = self.resolver(pulls={SHA: gone}, branch_prs={"b": gone})
        self.assertIsNone(resolver.resolve(SHA, "b", at(10)))

    def test_other_errors_propagate_so_the_run_is_retried_later(self):
        boom = collect.ApiError("boom", status=500)
        with self.assertRaises(collect.ApiError):
            self.resolver(pulls={SHA: boom})[0].resolve(SHA, "b", at(10))
        with self.assertRaises(collect.ApiError):
            self.resolver(branch_prs={"b": boom})[0].resolve(SHA, "b", at(10))


class CollectUsesTheBranchFallbackTests(unittest.TestCase):
    def test_collect_records_the_fallback_pr_number(self):
        with tempfile.TemporaryDirectory() as directory:
            run = make_run(event="pull_request", branch="ci/collector")
            api = FakeApi(
                runs=[run],
                jobs={100: [make_job(1)]},
                pulls={SHA: [commit_pr(727, "ci/rate-limits")]},
                branch_prs={"ci/collector": [pr_info(725, -60)]},
            )
            collect_into(api, [run], directory)
            self.assertEqual(725, read_shard(directory)[0]["pr_number"])


class FixPrNumbersTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data = Path(self.directory.name)

    def record(self, job_id, event="pull_request", branch="b", pr_number=None, sha=SHA):
        run = make_run(job_id, event=event, branch=branch)
        run["head_sha"] = sha
        found = collect.build_records(run, [make_job(job_id)], lambda *_: pr_number, lambda _: None, [])[0][0]
        return found

    def write(self, name, records):
        (self.data / name).write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))

    def fix(self, api, **kwargs):
        lines = []
        found = collect.fix_pr_numbers(api, CONFIG, self.data, out=lines.append, **kwargs)
        return found, lines

    def test_fills_in_only_the_null_pr_numbers_of_pull_request_records(self):
        nulls = self.record(1, branch="stacked")
        known = self.record(2, branch="known", pr_number=11)
        pushed = self.record(3, event="push", branch="dev")
        no_pr = self.record(4, branch="nothing")
        self.write("2026-W40.jsonl", [nulls, known, pushed, no_pr])
        api = FakeApi(branch_prs={"stacked": [pr_info(725, -60)]})

        (found, fixed), lines = self.fix(api)

        self.assertEqual((2, 1), (found, fixed))
        rows = read_shard(self.data)
        self.assertEqual([725, 11, None, None], [r["pr_number"] for r in rows])
        self.assertIn("2 pull_request records without pr_number; fixed 1, 1 still unresolved", lines[-1])

    def test_only_the_pr_number_value_changes_every_other_byte_is_kept(self):
        before = self.record(1, branch="stacked")
        untouched = self.record(2, branch="known", pr_number=11)
        self.write("2026-W40.jsonl", [before, untouched])
        original = (self.data / "2026-W40.jsonl").read_text().splitlines()

        self.fix(FakeApi(branch_prs={"stacked": [pr_info(725, -60)]}))

        changed = (self.data / "2026-W40.jsonl").read_text().splitlines()
        self.assertEqual(original[1], changed[1])
        self.assertEqual(original[0].replace('"pr_number":null', '"pr_number":725'), changed[0])

    def test_dry_run_reports_but_writes_nothing(self):
        self.write("2026-W40.jsonl", [self.record(1, branch="stacked")])
        before = (self.data / "2026-W40.jsonl").read_bytes()
        (found, fixed), lines = self.fix(FakeApi(branch_prs={"stacked": [pr_info(725, -60)]}), dry_run=True)
        self.assertEqual((1, 1), (found, fixed))
        self.assertEqual(before, (self.data / "2026-W40.jsonl").read_bytes())
        self.assertIn("would fix 1", lines[-1])

    def test_a_second_run_changes_nothing(self):
        self.write("2026-W40.jsonl", [self.record(1, branch="stacked")])
        api = FakeApi(branch_prs={"stacked": [pr_info(725, -60)]})
        self.fix(api)
        after_first = (self.data / "2026-W40.jsonl").read_bytes()
        (found, fixed), _ = self.fix(api)
        self.assertEqual((0, 0), (found, fixed))
        self.assertEqual(after_first, (self.data / "2026-W40.jsonl").read_bytes())

    def test_files_with_nothing_to_fix_are_not_rewritten_and_no_temp_file_is_left(self):
        self.write("2026-W39.jsonl", [self.record(1, branch="known", pr_number=3)])
        self.write("2026-W40.jsonl", [self.record(2, branch="stacked")])
        untouched = self.data / "2026-W39.jsonl"
        stamp = untouched.stat().st_mtime_ns
        self.fix(FakeApi(branch_prs={"stacked": [pr_info(725, -60)]}))
        self.assertEqual(stamp, untouched.stat().st_mtime_ns)
        self.assertEqual([], list(self.data.glob("*.tmp")))

    def test_an_api_error_leaves_the_data_unchanged(self):
        self.write("2026-W40.jsonl", [self.record(1, branch="stacked")])
        before = (self.data / "2026-W40.jsonl").read_bytes()
        api = FakeApi(pulls={SHA: collect.ApiError("boom", status=500)})
        with self.assertRaises(collect.ApiError):
            self.fix(api)
        self.assertEqual(before, (self.data / "2026-W40.jsonl").read_bytes())


class FixPrNumbersCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = []

    def run_main(self, api, *argv):
        return collect.main(
            ["--data-dir", self.directory.name, *argv], api=api, today=date(2026, 9, 30), out=self.output.append
        )

    def test_fix_pr_numbers_does_not_list_or_collect_runs(self):
        api = FakeApi()
        self.assertEqual(0, self.run_main(api, "--fix-pr-numbers"))
        self.assertEqual([], [c for c in api.calls if c.startswith("actions/")])
        self.assertIn("0 pull_request records without pr_number", self.output[-1])

    def test_rate_limit_during_the_repair_exits_with_2(self):
        record = collect.build_records(
            make_run(event="pull_request", branch="b"), [make_job(1)], lambda *_: None, lambda _: None, []
        )[0][0]
        (Path(self.directory.name) / "2026-W40.jsonl").write_text(json.dumps(record) + "\n")
        limited = collect.ApiError("gh: API rate limit exceeded (HTTP 403)", status=403, rate_limited=True)
        self.assertEqual(2, self.run_main(FakeApi(pulls={SHA: limited}), "--fix-pr-numbers"))
        self.assertTrue(any("could not look up pull requests" in line for line in self.output))


class ShortMessageTests(unittest.TestCase):
    def test_strips_the_gh_prefix_and_the_support_boilerplate(self):
        error = collect.ApiError(
            "gh: API rate limit exceeded for user ID 77. If you reach out to GitHub Support for help, "
            "please include the request ID X (HTTP 403)"
        )
        self.assertEqual("API rate limit exceeded for user ID 77.", collect.short_message(error))

    def test_leaves_other_messages_alone(self):
        self.assertEqual("timed out", collect.short_message(collect.ApiError("timed out")))


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data = Path(self.directory.name)

    def one_run_api(self, **kwargs):
        run = make_run()
        return run, FakeApi(runs=[run], jobs={100: [make_job(1000), make_job(1001, name="lint")]}, **kwargs)

    def test_writes_records_to_the_shard_of_the_runs_week(self):
        run, api = self.one_run_api()
        result = collect_into(api, [run], self.data)
        self.assertEqual(0, result.exit_code())
        self.assertEqual(2, result.records_written)
        records = read_shard(self.data)
        self.assertEqual([1000, 1001], [r["job_id"] for r in records])
        for record in records:
            self.assertEqual([], validation_errors(record))

    def test_second_run_writes_nothing_and_does_not_refetch_jobs(self):
        run, api = self.one_run_api()
        collect_into(api, [run], self.data)
        before = (self.data / "2026-W40.jsonl").read_text()
        api.calls.clear()

        result = collect_into(api, [run], self.data)

        self.assertEqual(0, result.records_written)
        self.assertEqual(1, result.runs_already_collected)
        self.assertEqual([], [c for c in api.calls if "/jobs" in c])
        self.assertEqual(before, (self.data / "2026-W40.jsonl").read_text())

    def test_rerun_appends_only_the_new_attempt_using_filter_all(self):
        run, api = self.one_run_api()
        collect_into(api, [run], self.data)

        rerun = make_run(attempt=2)
        api.runs = [rerun]
        api.jobs = {100: [make_job(1000), make_job(1001, name="lint"), make_job(2000, attempt=2)]}
        api.calls.clear()
        result = collect_into(api, [rerun], self.data)

        self.assertEqual(1, result.records_written)
        records = read_shard(self.data)
        self.assertEqual([1000, 1001, 2000], [r["job_id"] for r in records])
        self.assertEqual(2, records[-1]["run_attempt"])
        job_calls = [c for c in api.calls if "/jobs" in c]
        self.assertTrue(job_calls and all("filter=all" in c for c in job_calls))

    def test_dry_run_writes_nothing(self):
        run, api = self.one_run_api()
        result = collect_into(api, [run], self.data, dry_run=True)
        self.assertEqual(2, result.records_written)
        self.assertEqual([], list(self.data.glob("*.jsonl")))

    def test_max_runs_takes_the_newest_runs_first(self):
        old, new = make_run(1, created=0), make_run(2, created=3600)
        api = FakeApi(runs=[old, new], jobs={1: [make_job(10)], 2: [make_job(20)]})
        result = collect_into(api, [old, new], self.data, max_runs=1)
        self.assertEqual(1, result.runs_collected)
        self.assertEqual([20], [r["job_id"] for r in read_shard(self.data)])

    def test_appending_to_a_shard_without_a_trailing_newline(self):
        first = collect.build_records(make_run(1), [make_job(10)], lambda *_: None, lambda _: None, [])[0][0]
        (self.data / "2026-W40.jsonl").write_text(json.dumps(first))  # no newline
        run = make_run(2)
        api = FakeApi(runs=[run], jobs={2: [make_job(20)]})
        collect_into(api, [run], self.data)
        self.assertEqual([10, 20], [r["job_id"] for r in read_shard(self.data)])

    def test_a_corrupt_shard_line_is_an_error_not_a_silent_duplicate_risk(self):
        (self.data / "2026-W40.jsonl").write_text("{not json}\n")
        run, api = self.one_run_api()
        with self.assertRaises(ValueError):
            collect_into(api, [run], self.data)

    def test_one_failing_run_is_reported_and_the_others_are_still_collected(self):
        good, bad = make_run(1, created=0), make_run(2, created=10)
        api = FakeApi(
            runs=[good, bad],
            jobs={1: [make_job(10)]},
            job_errors={2: collect.ApiError("HTTP 500", status=500)},
        )
        result = collect_into(api, [good, bad], self.data)
        self.assertEqual(1, result.exit_code())
        self.assertEqual(1, len(result.errors))
        self.assertEqual([10], [r["job_id"] for r in read_shard(self.data)])

    def test_rate_limit_stops_early_but_keeps_what_was_collected(self):
        first, second = make_run(1, created=100), make_run(2, created=0)  # newest first
        api = FakeApi(
            runs=[first, second],
            jobs={1: [make_job(10)], 2: [make_job(20)]},
            job_errors={2: collect.ApiError("rate limit", status=403, rate_limited=True)},
        )
        result = collect_into(api, [first, second], self.data)
        self.assertEqual(2, result.exit_code())
        self.assertEqual([10], [r["job_id"] for r in read_shard(self.data)])

    def test_rate_limit_message_is_kept_once_and_shown_in_the_summary(self):
        limited = collect.ApiError(
            "gh: API rate limit exceeded for user ID 1. If you reach out to GitHub Support (HTTP 403)",
            status=403,
            rate_limited=True,
        )
        runs = [make_run(n, created=n * 10) for n in (1, 2, 3)]
        api = FakeApi(runs=runs, jobs={}, job_errors={1: limited, 2: limited, 3: limited})
        result = collect_into(api, runs, self.data)
        self.assertEqual("API rate limit exceeded for user ID 1.", result.rate_limit_message)

        lines = []
        collect.summarize(result, False, out=lines.append)
        stopped = [line for line in lines if "STOPPED EARLY" in line]
        self.assertEqual(1, len(stopped))
        self.assertIn("API rate limit exceeded for user ID 1.", stopped[0])
        self.assertIn("--rate-limit-wait", stopped[0])
        self.assertNotIn("If you reach out", "\n".join(lines))

    def test_summary_without_a_message_still_says_it_stopped_early(self):
        result = collect.Result()
        result.rate_limited = True
        lines = []
        collect.summarize(result, False, out=lines.append)
        self.assertTrue(any("STOPPED EARLY: API rate limit reached;" in line for line in lines))

    def test_pr_lookup_404_means_no_pr_and_other_errors_fail_the_run(self):
        run = make_run(event="pull_request", branch="feature/x")
        api = FakeApi(runs=[run], jobs={100: [make_job(1)]}, pulls={SHA: collect.ApiError("gone", status=404)})
        collect_into(api, [run], self.data)
        self.assertIsNone(read_shard(self.data)[0]["pr_number"])

        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        api = FakeApi(runs=[run], jobs={100: [make_job(1)]}, pulls={SHA: collect.ApiError("boom", status=500)})
        result = collect_into(api, [run], other.name)
        self.assertEqual(1, result.exit_code())
        self.assertEqual([], list(Path(other.name).glob("*.jsonl")))  # retried next time

    def test_pr_number_matches_the_pr_whose_head_is_the_runs_branch(self):
        run = make_run(event="pull_request", branch="feature/x")
        pulls = {SHA: [{"number": 5, "head": {"ref": "other"}}, {"number": 9, "head": {"ref": "feature/x"}}]}
        api = FakeApi(runs=[run], jobs={100: [make_job(1)]}, pulls=pulls)
        collect_into(api, [run], self.data)
        self.assertEqual(9, read_shard(self.data)[0]["pr_number"])

    def test_several_workers_give_the_same_records(self):
        runs = [make_run(n, created=n * 10) for n in range(1, 7)]
        api = FakeApi(runs=runs, jobs={n: [make_job(n * 10)] for n in range(1, 7)})
        collect_into(api, runs, self.data, workers=4)
        self.assertEqual([10, 20, 30, 40, 50, 60], sorted(r["job_id"] for r in read_shard(self.data)))


# --- build-analysis artifacts -------------------------------------------------


def make_summary(job_id=1000, run_id=100, attempt=1, **overrides):
    """What summarise_build.py writes (field for field)."""
    summary = {
        "schema_version": 1,
        "kind": "build-analysis",
        "run_id": run_id,
        "run_attempt": attempt,
        "job_id": job_id,
        "job_key": "lint",
        "artifact": f"build-timings-lint-a{attempt}",
        "ninja": {
            "log_version": 7,
            "complete": False,
            "steps": 550,
            "wall_s": 338.5,
            "cpu_s": 1138.4,
            "parallelism": 3.36,
            "tail_after_compile_s": 0.0,
            "by_kind": {
                "compile": {"steps": 445, "cpu_s": 1132.9},
                "archive": {"steps": 0, "cpu_s": 0.0},
                "link": {"steps": 104, "cpu_s": 5.3},
                "other": {"steps": 1, "cpu_s": 0.2},
            },
            "slowest": [
                {"target": "tests/CMakeFiles/t.dir/t.cpp.o", "kind": "compile", "seconds": 9.3},
                {"target": "apps/PQ", "kind": "link", "seconds": 2.0},
            ],
        },
        "ccache": {
            "hits": 179,
            "misses": 2,
            "hit_rate": 0.989,
            "counters": {"cache_miss": 2, "could_not_use_precompiled_header": 269, "direct_cache_hit": 179},
        },
    }
    summary.update(overrides)
    return summary


def make_zip(summary, name="build-analysis.json", extra=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, json.dumps(summary) if not isinstance(summary, (bytes, str)) else summary)
        for other, content in (extra or {}).items():
            archive.writestr(other, content)
    return buffer.getvalue()


def make_artifact(artifact_id=5, name="build-timings-lint-a1", size=900, expired=False):
    return {"id": artifact_id, "name": name, "size_in_bytes": size, "expired": expired}


class SanitiseSummaryTests(unittest.TestCase):
    def test_a_real_looking_summary_passes_and_unknown_fields_are_dropped(self):
        summary = make_summary(evil="x", ninja=dict(make_summary()["ninja"], extra="y"))
        clean = collect.sanitise_summary(summary)
        self.assertEqual((100, 1, 1000), (clean["run_id"], clean["run_attempt"], clean["job_id"]))
        self.assertNotIn("evil", clean)
        self.assertNotIn("extra", clean["ninja"])
        self.assertEqual(269, clean["ccache"]["counters"]["could_not_use_precompiled_header"])

    def test_missing_parts_may_be_null(self):
        clean = collect.sanitise_summary(make_summary(ninja=None, ccache=None))
        self.assertEqual((None, None), (clean["ninja"], clean["ccache"]))

    def test_an_unusable_ninja_log_keeps_its_error(self):
        ninja = {"log_version": 4, "complete": None, "steps": 0, "error": "no usable .ninja_log"}
        self.assertEqual("no usable .ninja_log", collect.sanitise_summary(make_summary(ninja=ninja))["ninja"]["error"])

    def rejected(self, **overrides):
        with self.assertRaises(ValueError, msg=str(overrides)[:80]):
            collect.sanitise_summary(make_summary(**overrides))

    def test_rejects_wrong_identity_and_types(self):
        self.rejected(kind="job")
        self.rejected(schema_version=2)
        self.rejected(job_id="1000")
        self.rejected(job_id=True)
        self.rejected(job_id=-1)
        self.rejected(run_attempt=0)

    def mutated(self, path, value):
        summary = make_summary()
        node = summary
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with self.assertRaises(ValueError, msg=f"{path}={value!r}"):
            collect.sanitise_summary(summary)

    def test_rejects_hostile_numbers(self):
        self.mutated(["ninja", "wall_s"], float("nan"))
        self.mutated(["ninja", "wall_s"], float("inf"))
        self.mutated(["ninja", "wall_s"], -1)
        self.mutated(["ninja", "wall_s"], 10**9)
        self.mutated(["ninja", "wall_s"], "10")
        self.mutated(["ninja", "steps"], 1.5)
        self.mutated(["ccache", "hit_rate"], 1.5)
        self.mutated(["ccache", "counters", "cache_miss"], 10**15)
        self.mutated(["ccache", "counters", "cache_miss"], "2")

    def test_rejects_hostile_text(self):
        self.mutated(["ninja", "slowest", 0, "target"], "a\nb")
        self.mutated(["ninja", "slowest", 0, "target"], "x" * (collect.MAX_TARGET_CHARS + 1))
        self.mutated(["ninja", "slowest", 0, "target"], "")
        self.mutated(["ninja", "slowest", 0, "kind"], "evil")
        self.mutated(["ccache", "counters", "Evil Key"], 1)
        self.mutated(["ccache", "counters", "x" * 65], 1)

    def test_rejects_oversized_structures(self):
        self.mutated(["ninja", "slowest"], [make_summary()["ninja"]["slowest"][0]] * (collect.MAX_SLOWEST + 1))
        self.mutated(["ccache", "counters"], {f"c{i}": 1 for i in range(collect.MAX_COUNTERS + 1)})
        self.mutated(["ninja", "by_kind", "gpu"], {"steps": 1, "cpu_s": 1})

    def test_the_clean_summary_is_independent_of_the_input(self):
        summary = make_summary()
        clean = collect.sanitise_summary(summary)
        summary["ninja"]["slowest"][0]["target"] = "changed"
        self.assertEqual("tests/CMakeFiles/t.dir/t.cpp.o", clean["ninja"]["slowest"][0]["target"])


def make_includes(**overrides):
    includes = {
        "objects": 403,
        "unique_files": 1080,
        "include_pairs": 189520,
        "project_files": 402,
        "project_pairs": 27820,
        "digest": "a" * 64,
        "top_project_files": [
            {"file": "external/mstd/include/mstd/enum.hpp", "fan_in": 353},
            {"file": "include/engine/engineOutput.hpp", "fan_in": 124},
        ],
    }
    includes.update(overrides)
    return includes


def make_clang(**overrides):
    clang = {
        "schema_version": 1,
        "kind": "clang-trace-summary",
        "limits": {"files": 20, "headers": 30, "templates": 30},
        "files": 403,
        "unreadable": 0,
        "total_s": 1234.198,
        "frontend_s": 800.5,
        "backend_s": 433.7,
        "source_events": 54899,
        "instantiation_events": 341005,
        "slowest_files": [{"file": "src/constraints/mShake.cpp", "total_s": 18.3, "frontend_s": 7.7, "backend_s": 10.6}],
        "headers": [{"header": "include/engine/engineOutput.hpp", "inclusive_s": 52.7, "self_s": 42.8, "events": 124, "files": 124}],
        "templates": [{"name": "std::span<const ExceptionType>", "count": 298, "inclusive_s": 12.0, "self_s": 4.4}],
    }
    clang.update(overrides)
    return clang


class SanitiseIncludesTests(unittest.TestCase):
    def clean(self, **overrides):
        return collect.sanitise_summary(make_summary(includes=make_includes(**overrides)))["includes"]

    def test_a_real_looking_graph_passes_and_extra_fields_are_dropped(self):
        includes = make_includes(evil="x")
        clean = collect.sanitise_summary(make_summary(includes=includes))["includes"]
        self.assertEqual(("a" * 64, 189520), (clean["digest"], clean["include_pairs"]))
        self.assertNotIn("evil", clean)
        self.assertEqual(353, clean["top_project_files"][0]["fan_in"])

    def test_missing_graph_is_none(self):
        self.assertIsNone(collect.sanitise_summary(make_summary())["includes"])

    def test_rejects_bad_digest_numbers_and_lists(self):
        for overrides in (
            {"digest": "A" * 64},
            {"digest": "a" * 63},
            {"digest": 5},
            {"include_pairs": -1},
            {"include_pairs": 1.5},
            {"objects": "403"},
            {"project_pairs": 10**13},
            {"top_project_files": "x"},
            {"top_project_files": [{"file": "a.hpp", "fan_in": 1}] * (collect.MAX_CLANG_LIST + 1)},
            {"top_project_files": [{"file": "a\nb", "fan_in": 1}]},
            {"top_project_files": [{"file": "", "fan_in": 1}]},
            {"top_project_files": [{"file": "a.hpp", "fan_in": "2"}]},
        ):
            with self.assertRaises(ValueError, msg=str(overrides)[:60]):
                self.clean(**overrides)


class SanitiseClangTests(unittest.TestCase):
    def clean(self, **overrides):
        return collect.sanitise_summary(make_summary(), make_clang(**overrides))["clang"]

    def test_a_real_looking_summary_passes_and_extra_fields_are_dropped(self):
        clang = make_clang(evil=1)
        clang["headers"][0]["evil"] = 1
        clean = collect.sanitise_summary(make_summary(), clang)["clang"]
        self.assertEqual((403, 1234.198), (clean["files"], clean["total_s"]))
        self.assertNotIn("evil", clean)
        self.assertNotIn("evil", clean["headers"][0])
        self.assertEqual(124, clean["headers"][0]["files"])

    def test_no_clang_part_is_none(self):
        self.assertIsNone(collect.sanitise_summary(make_summary())["clang"])

    def test_frontend_and_backend_of_a_file_may_be_null(self):
        entry = {"file": "a.cpp", "total_s": 1.0, "frontend_s": None, "backend_s": None}
        self.assertIsNone(self.clean(slowest_files=[entry])["slowest_files"][0]["frontend_s"])

    def test_rejects_the_wrong_identity(self):
        for overrides in ({"kind": "clang-trace-detail"}, {"kind": "job"}, {"schema_version": 2}):
            with self.assertRaises(ValueError, msg=str(overrides)):
                self.clean(**overrides)

    def test_rejects_hostile_numbers_text_and_sizes(self):
        header = make_clang()["headers"][0]
        for overrides in (
            {"total_s": float("nan")},
            {"total_s": -1},
            {"total_s": 10**9},
            {"files": "403"},
            {"source_events": 1.5},
            {"limits": {"files": 0, "headers": 30, "templates": 30}},
            {"limits": {"files": 20, "headers": 30}},
            {"limits": "x"},
            {"slowest_files": [{"file": "a", "total_s": 1.0, "frontend_s": 1.0, "backend_s": 1.0}] * (collect.MAX_CLANG_FILES + 1)},
            {"slowest_files": [{"file": "a\nb", "total_s": 1.0, "frontend_s": 1.0, "backend_s": 1.0}]},
            {"headers": [header] * (collect.MAX_CLANG_LIST + 1)},
            {"headers": [dict(header, header="x" * (collect.MAX_TARGET_CHARS + 1))]},
            {"headers": [dict(header, self_s=float("inf"))]},
            {"headers": [dict(header, files=-1)]},
            {"headers": "x"},
            {"templates": [{"name": "", "count": 1, "inclusive_s": 1.0, "self_s": 1.0}]},
            {"templates": [{"name": "T", "count": "1", "inclusive_s": 1.0, "self_s": 1.0}]},
        ):
            with self.assertRaises(ValueError, msg=str(overrides)[:70]):
                self.clean(**overrides)

    def test_the_clean_summary_is_independent_of_the_input(self):
        clang = make_clang()
        clean = collect.sanitise_summary(make_summary(), clang)["clang"]
        clang["headers"][0]["header"] = "changed"
        self.assertEqual("include/engine/engineOutput.hpp", clean["headers"][0]["header"])


class ReadArtifactTests(unittest.TestCase):
    def test_reads_both_members_and_ignores_the_others(self):
        blob = make_zip(make_summary(), extra={"clang-traces.json": json.dumps(make_clang()), "ninja_log.txt": "x" * 100, "traces/big.json": "{}"})
        summary, clang = collect.read_artifact(blob)
        self.assertEqual(1000, summary["job_id"])
        self.assertEqual("clang-trace-summary", clang["kind"])

    def test_the_clang_member_is_optional_but_the_summary_is_not(self):
        self.assertIsNone(collect.read_artifact(make_zip(make_summary()))[1])
        with self.assertRaises(KeyError):
            collect.read_artifact(make_zip(make_summary(), name="other.json", extra={"clang-traces.json": "{}"}))

    def test_an_oversized_or_broken_clang_member_raises_what_the_collector_catches(self):
        with self.assertRaises(ValueError) as raised:
            collect.read_artifact(make_zip(make_summary(), extra={"clang-traces.json": " " * (collect.MAX_ARTIFACT_BYTES + 10)}))
        self.assertNotIsInstance(raised.exception, json.JSONDecodeError)  # rejected by the size check, not by parsing
        self.assertIn("too large", str(raised.exception))
        with self.assertRaises(json.JSONDecodeError):
            collect.read_artifact(make_zip(make_summary(), extra={"clang-traces.json": "{broken"}))


class ClangRecordsTests(unittest.TestCase):
    def build(self, event, zip_extra=True, summary=None):
        run = make_run(100, "Clang Build", event, "dev" if event == "push" else "feature/x")
        jobs = [make_job(1000, "clang-build")]
        extra = {"clang-traces.json": json.dumps(make_clang())} if zip_extra else None
        blob = make_zip(summary or make_summary(includes=make_includes()), extra=extra)
        api = FakeApi(runs=[run], jobs={100: jobs}, artifacts={100: [make_artifact(5, "build-timings-clang-a1")]}, blobs={5: blob})
        job_records, _, _ = records_for(run, jobs)
        return collect.build_analysis_records(api, run, job_records)

    def test_pushes_get_the_clang_summary_and_the_include_graph(self):
        records, dropped = self.build("push")
        self.assertEqual(0, sum(dropped.values()))
        self.assertEqual([], validation_errors(records[0], ANALYSIS_SCHEMA))
        self.assertEqual(1234.198, records[0]["clang"]["total_s"])
        self.assertEqual("a" * 64, records[0]["includes"]["digest"])

    def test_pull_requests_keep_the_include_graph_but_not_the_clang_summary(self):
        records, _ = self.build("pull_request")
        self.assertIsNone(records[0]["clang"])
        self.assertEqual(189520, records[0]["includes"]["include_pairs"])
        self.assertEqual([], validation_errors(records[0], ANALYSIS_SCHEMA))

    def test_records_without_the_new_parts_still_validate(self):
        records, _ = self.build("push", zip_extra=False, summary=make_summary())
        self.assertEqual((None, None), (records[0]["clang"], records[0]["includes"]))
        self.assertEqual([], validation_errors(records[0], ANALYSIS_SCHEMA))

    def test_an_invalid_clang_part_drops_the_artifact_and_counts_it(self):
        run = make_run(100, "Clang Build", "push", "dev")
        jobs = [make_job(1000, "clang-build")]
        bad = make_zip(make_summary(), extra={"clang-traces.json": json.dumps(make_clang(total_s=-5))})
        api = FakeApi(runs=[run], jobs={100: jobs}, artifacts={100: [make_artifact(5, "build-timings-clang-a1")]}, blobs={5: bad})
        job_records, _, _ = records_for(run, jobs)
        records, dropped = collect.build_analysis_records(api, run, job_records)
        self.assertEqual(([], {"invalid build-analysis artifact": 1}), (records, dict(dropped)))

    def test_the_raw_trace_artifact_is_never_downloaded(self):
        run = make_run(100, "Clang Build", "push", "dev")
        jobs = [make_job(1000, "clang-build")]
        artifacts = [make_artifact(5, "build-timings-clang-a1"), make_artifact(6, "clang-traces-clang-a1", size=30_000_000)]
        api = FakeApi(runs=[run], jobs={100: jobs}, artifacts={100: artifacts}, blobs={5: make_zip(make_summary())})
        job_records, _, _ = records_for(run, jobs)
        collect.build_analysis_records(api, run, job_records)
        self.assertEqual(["actions/artifacts/5/zip"], [c for c in api.calls if c.endswith("/zip")])

    def test_the_clang_workflow_is_configured(self):
        self.assertIn("Clang Build", CONFIG["workflows"])
        self.assertIn("Clang Build", CONFIG["build_analysis_workflows"])


class ReadSummaryTests(unittest.TestCase):
    def test_reads_the_json_from_the_zip(self):
        self.assertEqual(1000, collect.read_summary(make_zip(make_summary()))["job_id"])

    def test_wrong_inputs_raise_what_the_collector_catches(self):
        with self.assertRaises(zipfile.BadZipFile):
            collect.read_summary(b"not a zip")
        with self.assertRaises(KeyError):
            collect.read_summary(make_zip(make_summary(), name="other.json"))
        with self.assertRaises(json.JSONDecodeError):
            collect.read_summary(make_zip("{not json"))
        with self.assertRaises(ValueError):
            collect.read_summary(make_zip(" " * (collect.MAX_ARTIFACT_BYTES + 10)))


def analysis_api(artifacts, blobs, run=None, jobs=None):
    run = run or make_run(100, "BUILD", "push", "dev")
    jobs = jobs if jobs is not None else [make_job(1000, "lint"), make_job(1001, "other")]
    return run, FakeApi(runs=[run], jobs={run["id"]: jobs}, artifacts={run["id"]: artifacts}, blobs=blobs)


class BuildAnalysisRecordsTests(unittest.TestCase):
    def build(self, artifacts, blobs, **kwargs):
        run, api = analysis_api(artifacts, blobs, **kwargs)
        job_records, _, _ = records_for(run, api.jobs[run["id"]])
        records, dropped = collect.build_analysis_records(api, run, job_records)
        return records, dropped, api, job_records

    def test_joins_to_the_job_record_using_api_values(self):
        records, dropped, _, job_records = self.build([make_artifact()], {5: make_zip(make_summary())})
        self.assertEqual(0, sum(dropped.values()))
        record = records[0]
        self.assertEqual([], validation_errors(record, ANALYSIS_SCHEMA))
        job = next(j for j in job_records if j["job_id"] == 1000)
        for key in ("workflow", "run_id", "run_attempt", "event", "branch", "head_sha", "job_id", "job", "created_at", "conclusion"):
            self.assertEqual(job[key], record[key], key)
        self.assertEqual("lint", record["job"])
        self.assertEqual(269, record["ccache"]["counters"]["could_not_use_precompiled_header"])

    def test_the_artifact_cannot_override_join_keys(self):
        summary = make_summary(workflow="Evil", branch="evil", head_sha="e" * 40, event="schedule", job="evil")
        records, _, _, job_records = self.build([make_artifact()], {5: make_zip(summary)})
        self.assertEqual("BUILD", records[0]["workflow"])
        self.assertEqual("dev", records[0]["branch"])
        self.assertEqual(SHA, records[0]["head_sha"])

    def test_artifacts_that_do_not_match_are_dropped_not_recorded(self):
        cases = {
            "other run": make_summary(run_id=999),
            "other attempt": make_summary(attempt=2),
            "unknown job": make_summary(job_id=4242),
            "null job id": make_summary(job_id=None),
        }
        for label, summary in cases.items():
            records, dropped, _, _ = self.build([make_artifact()], {5: make_zip(summary)})
            self.assertEqual([], records, label)
            self.assertEqual(1, sum(dropped.values()), label)

    def test_invalid_content_is_dropped(self):
        for blob in (b"junk", make_zip("{broken"), make_zip(make_summary(), name="x.json"), make_zip(make_summary(kind="job"))):
            records, dropped, _, _ = self.build([make_artifact()], {5: blob})
            self.assertEqual([], records)
            self.assertEqual({"invalid build-analysis artifact": 1}, dict(dropped))

    def test_unrelated_expired_and_oversized_artifacts_are_not_downloaded(self):
        artifacts = [
            make_artifact(5, "docs-site"),
            make_artifact(6, expired=True),
            make_artifact(7, size=collect.MAX_ARTIFACT_BYTES + 1),
        ]
        records, dropped, api, _ = self.build(artifacts, {})  # no blobs: a download would raise KeyError
        self.assertEqual([], records)
        self.assertEqual([], [c for c in api.calls if c.endswith("/zip")])
        self.assertEqual(
            {"build-analysis artifact expired": 1, "build-analysis artifact too large": 1}, dict(dropped)
        )

    def test_a_vanished_artifact_is_dropped_but_other_api_errors_propagate(self):
        records, dropped, _, _ = self.build([make_artifact()], {5: collect.ApiError("gone", status=410)})
        self.assertEqual({"build-analysis artifact expired": 1}, dict(dropped))
        with self.assertRaises(collect.ApiError):
            self.build([make_artifact()], {5: collect.ApiError("boom", status=500)})
        with self.assertRaises(collect.ApiError) as raised:
            self.build([make_artifact()], {5: collect.ApiError("limit", rate_limited=True)})
        self.assertTrue(raised.exception.rate_limited)

    def test_two_artifacts_for_one_job_give_one_record(self):
        blob = make_zip(make_summary())
        records, _, _, _ = self.build([make_artifact(5), make_artifact(6, "build-timings-lint-a1-copy")], {5: blob, 6: blob})
        self.assertEqual(1, len(records))

    def test_a_job_without_an_artifact_gets_no_record(self):
        records, _, _, _ = self.build([], {})
        self.assertEqual([], records)

    def test_every_attempt_of_a_rerun_run_can_have_its_own_record(self):
        jobs = [make_job(1000, "lint"), make_job(2000, "lint", attempt=2)]
        run = make_run(100, "BUILD", "push", "dev", attempt=2)
        artifacts = [make_artifact(5, "build-timings-lint-a1"), make_artifact(6, "build-timings-lint-a2")]
        blobs = {5: make_zip(make_summary(1000, attempt=1)), 6: make_zip(make_summary(2000, attempt=2))}
        _, api = analysis_api(artifacts, blobs, run=run, jobs=jobs)
        job_records, _, _ = records_for(run, jobs)
        records, _ = collect.build_analysis_records(api, run, job_records)
        self.assertEqual([(1000, 1), (2000, 2)], sorted((r["job_id"], r["run_attempt"]) for r in records))


class CollectBuildAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data = Path(self.directory.name)

    def lines(self):
        return [json.loads(line) for line in (self.data / "2026-W40.jsonl").read_text().splitlines()]

    def test_writes_job_and_analysis_records_once(self):
        run, api = analysis_api([make_artifact()], {5: make_zip(make_summary())})
        result = collect_into(api, [run], self.data)
        self.assertEqual(3, result.records_written)
        self.assertEqual(1, result.analyses_written)
        kinds = sorted((r["job_id"], r["kind"]) for r in self.lines())
        self.assertEqual([(1000, "build-analysis"), (1000, "job"), (1001, "job")], kinds)
        # a job record always precedes its analysis
        order = [(r["job_id"], r["kind"]) for r in self.lines()]
        self.assertLess(order.index((1000, "job")), order.index((1000, "build-analysis")))

        before = (self.data / "2026-W40.jsonl").read_text()
        again = collect_into(api, [run], self.data)
        self.assertEqual(0, again.records_written)
        self.assertEqual(before, (self.data / "2026-W40.jsonl").read_text())

    def test_other_workflows_are_not_asked_for_artifacts(self):
        run = make_run(100, "Docs", "push", "dev")
        _, api = analysis_api([make_artifact()], {}, run=run)
        collect_into(api, [run], self.data)
        self.assertEqual([], [c for c in api.calls if "artifacts" in c])

    def test_a_failing_download_retries_the_whole_run_later(self):
        run, api = analysis_api([make_artifact()], {5: collect.ApiError("boom", status=500)})
        result = collect_into(api, [run], self.data)
        self.assertEqual(1, len(result.errors))
        self.assertFalse((self.data / "2026-W40.jsonl").exists())

        api.blobs[5] = make_zip(make_summary())
        result = collect_into(api, [run], self.data)
        self.assertEqual((0, 3), (len(result.errors), result.records_written))

    def test_a_rate_limit_during_the_download_stops_without_writing(self):
        run, api = analysis_api([make_artifact()], {5: collect.ApiError("limit", rate_limited=True)})
        result = collect_into(api, [run], self.data)
        self.assertTrue(result.rate_limited)
        self.assertFalse((self.data / "2026-W40.jsonl").exists())

    def test_dry_run_writes_nothing_but_counts(self):
        run, api = analysis_api([make_artifact()], {5: make_zip(make_summary())})
        result = collect_into(api, [run], self.data, dry_run=True)
        self.assertEqual((3, 1), (result.records_written, result.analyses_written))
        self.assertFalse((self.data / "2026-W40.jsonl").exists())

    def test_summary_mentions_the_analyses_and_dropped_artifacts(self):
        run, api = analysis_api([make_artifact(), make_artifact(6, "build-timings-x-a1")], {5: make_zip(make_summary()), 6: b"junk"})
        result = collect_into(api, [run], self.data)
        lines = []
        collect.summarize(result, False, out=lines.append)
        self.assertIn("(incl. 1 build analyses)", lines[0])
        self.assertTrue(any("invalid build-analysis artifact=1" in line for line in lines))


class ShardIndexKindTests(unittest.TestCase):
    def test_jobs_and_analyses_are_tracked_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            job = {"kind": "job", "job_id": 1, "run_id": 9, "run_attempt": 1}
            analysis = {"kind": "build-analysis", "job_id": 2, "run_id": 9, "run_attempt": 1}
            (Path(directory) / "2026-W40.jsonl").write_text(json.dumps(job) + "\n" + json.dumps(analysis) + "\n")
            index = collect.ShardIndex(directory)
            self.assertTrue(index.has_record("2026-W40.jsonl", job))
            self.assertTrue(index.has_record("2026-W40.jsonl", analysis))
            self.assertFalse(index.has_record("2026-W40.jsonl", dict(analysis, job_id=1)))
            self.assertFalse(index.has_record("2026-W40.jsonl", dict(job, job_id=2)))

    def test_old_records_without_a_kind_count_as_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "2026-W40.jsonl").write_text(json.dumps({"job_id": 1, "run_id": 9, "run_attempt": 2}) + "\n")
            index = collect.ShardIndex(directory)
            self.assertTrue(index.has_attempt("2026-W40.jsonl", 9, 2))


class FixPrNumbersIgnoresAnalysesTests(unittest.TestCase):
    def test_build_analysis_records_are_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            analysis = make_analysis_line()
            path = Path(directory) / "2026-W40.jsonl"
            path.write_text(analysis + "\n")
            found, fixed = collect.fix_pr_numbers(FakeApi(), CONFIG, directory, out=lambda *_: None)
            self.assertEqual((0, 0), (found, fixed))
            self.assertEqual(analysis + "\n", path.read_text())


def make_analysis_line():
    run, api = analysis_api([make_artifact()], {5: make_zip(make_summary())}, run=make_run(100, "BUILD", "pull_request", "feature/x"))
    job_records, _, _ = records_for(run, api.jobs[100], pr=None)
    records, _ = collect.build_analysis_records(api, run, job_records)
    return json.dumps(records[0], separators=(",", ":"))


class GhApiBinaryTests(unittest.TestCase):
    def api(self, proc):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return proc

        return collect.GhApi("MolarVerse/PQ", run=run, sleep=lambda _: None), calls

    def test_get_bytes_returns_raw_bytes_and_asks_for_binary_output(self):
        api, calls = self.api(subprocess.CompletedProcess([], 0, stdout=b"PK\x03\x04\xff", stderr=b""))
        self.assertEqual(b"PK\x03\x04\xff", api.get_bytes("actions/artifacts/5/zip"))
        command, kwargs = calls[0]
        self.assertEqual("repos/MolarVerse/PQ/actions/artifacts/5/zip", command[-1])
        self.assertFalse(kwargs["text"])

    def test_binary_errors_are_classified_like_text_ones(self):
        api, _ = self.api(subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"gh: Not Found (HTTP 404)"))
        with self.assertRaises(collect.ApiError) as raised:
            api.get_bytes("actions/artifacts/5/zip")
        self.assertEqual(404, raised.exception.status)


class MainTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = []

    def run_main(self, api, *argv):
        code = collect.main(
            ["--data-dir", self.directory.name, "--workers", "1", *argv],
            api=api,
            today=date(2026, 9, 30),
            out=self.output.append,
        )
        return code

    def test_only_recorded_workflows_are_collected_over_the_lookback_window(self):
        recorded, ignored = make_run(1, name="BUILD"), make_run(2, name="PQ Bot")
        api = FakeApi(runs=[recorded, ignored], jobs={1: [make_job(10)], 2: [make_job(20)]})
        self.assertEqual(0, self.run_main(api, "--since-days", "2"))
        self.assertEqual([10], [r["job_id"] for r in read_shard(self.directory.name)])
        listing = [c for c in api.calls if c.startswith("actions/runs?")]
        self.assertEqual(2, len(listing))  # one query per day
        self.assertIn("2026-09-29", listing[0])
        self.assertIn("2026-09-30", listing[1])

    def test_run_id_collects_that_run_regardless_of_the_workflow_allowlist(self):
        run = make_run(7, name="PQ Bot")
        api = FakeApi(runs=[run], jobs={7: [make_job(70)]})
        self.assertEqual(0, self.run_main(api, "--run-id", "7"))
        self.assertEqual([70], [r["job_id"] for r in read_shard(self.directory.name)])

    def test_unfinished_run_id_is_skipped(self):
        run = make_run(7)
        run["status"] = "in_progress"
        api = FakeApi(runs=[run], jobs={7: [make_job(70)]})
        self.assertEqual(0, self.run_main(api, "--run-id", "7"))
        self.assertEqual([], list(Path(self.directory.name).glob("*.jsonl")))

    def test_listing_failure_is_reported_with_the_right_exit_code(self):
        class Failing(FakeApi):
            def get_json(self, path):
                raise collect.ApiError("rate limit", status=403, rate_limited=True)

        self.assertEqual(2, self.run_main(Failing(), "--since-days", "1"))
        self.assertTrue(any("could not list runs" in line for line in self.output))

    def test_listing_failure_shows_the_shortened_rate_limit_message(self):
        class Failing(FakeApi):
            def get_json(self, path):
                raise collect.ApiError(
                    "gh: API rate limit exceeded for user ID 1. If you reach out to GitHub Support (HTTP 403)",
                    status=403,
                    rate_limited=True,
                )

        self.run_main(Failing(), "--since-days", "1")
        self.assertEqual(["ERROR could not list runs: API rate limit exceeded for user ID 1."], self.output)

    def test_rate_limit_wait_is_passed_to_the_api_in_seconds(self):
        seen = {}

        def fake_gh_api(repo, **options):
            seen.update(options, repo=repo)
            return FakeApi()

        with mock.patch.object(collect, "GhApi", side_effect=fake_gh_api):
            self.run_main(None, "--since-days", "1", "--rate-limit-wait", "5")
        self.assertEqual(300, seen["rate_limit_wait"])
        self.assertEqual("MolarVerse/PQ", seen["repo"])
        self.assertTrue(callable(seen["notice"]))

    def test_rate_limit_wait_defaults_to_not_waiting(self):
        seen = {}
        with mock.patch.object(collect, "GhApi", side_effect=lambda repo, **o: (seen.update(o), FakeApi())[1]):
            self.run_main(None, "--since-days", "1")
        self.assertEqual(0, seen["rate_limit_wait"])

    def test_negative_rate_limit_wait_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()) as usage, self.assertRaises(SystemExit):
            self.run_main(FakeApi(), "--rate-limit-wait", "-1")
        self.assertIn("must not be negative", usage.getvalue())

    def test_summary_mentions_dropped_jobs_and_counts(self):
        run = make_run()
        api = FakeApi(runs=[run], jobs={100: [make_job(1), make_job(2, conclusion="skipped")]})
        self.run_main(api, "--since-days", "1", "--dry-run")
        text = "\n".join(self.output)
        self.assertIn("would write 1 records", text)
        self.assertIn("skipped=1", text)


if __name__ == "__main__":
    unittest.main()
