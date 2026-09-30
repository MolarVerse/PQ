import contextlib
import importlib.util
import io
import json
import re
import subprocess
import sys
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
CONFIG = json.loads((METRICS / "config.json").read_text())


# --- a small JSON Schema validator -----------------------------------------
# jsonschema is not installed where these tests run, so this implements just
# the keywords schema.json uses. MiniValidatorTests keep it honest.


def _is_type(value, name):
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
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
    if _is_type(value, "integer") and value < schema.get("minimum", value):
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

    def resolve_pr(sha, branch):
        pr_calls.append((sha, branch))
        return pr

    def fetch_log(job_id):
        return (logs or {}).get(job_id)

    records, dropped = collect.build_records(run, jobs, resolve_pr, fetch_log, list(signatures))
    return records, dropped, pr_calls


class FakeApi:
    """Stands in for GhApi: serves runs, jobs, PRs and logs from dicts."""

    def __init__(self, runs=(), jobs=None, pulls=None, logs=None, job_errors=None):
        self.runs = list(runs)
        self.jobs = jobs or {}
        self.pulls = pulls or {}
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
        raise AssertionError(f"unexpected API path {path}")

    @staticmethod
    def _page(items, query):
        size, page = int(query["per_page"][0]), int(query["page"][0])
        return items[(page - 1) * size : page * size]

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
        self.assertEqual([(SHA, "feature/x")], calls)  # once per run

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
