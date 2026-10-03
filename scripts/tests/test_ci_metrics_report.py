import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

METRICS = Path(__file__).resolve().parents[2] / ".github" / "ci-metrics"
SPEC = importlib.util.spec_from_file_location("ci_metrics_report", METRICS / "report.py")
report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report
SPEC.loader.exec_module(report)

NEWEST = datetime(2026, 9, 30, 12, 0, 0, tzinfo=timezone.utc)
_ids = iter(range(1, 10**9))


def stamp(moment):
    return moment.strftime(report.TIMESTAMP_FORMAT)


def record(
    *,
    workflow="BUILD",
    job="build",
    event="pull_request",
    branch="feature/x",
    run_id=None,
    attempt=1,
    conclusion="success",
    created=NEWEST,
    queue_s=3,
    duration_s=600,
    eigen=None,
    infra=None,
    **extra,
):
    started = created + timedelta(seconds=queue_s)
    data = {
        "schema_version": 1,
        "kind": "job",
        "workflow": workflow,
        "run_id": run_id if run_id is not None else next(_ids),
        "run_attempt": attempt,
        "event": event,
        "branch": branch,
        "job": job,
        "conclusion": conclusion,
        "created_at": stamp(created),
        "started_at": stamp(started),
        "completed_at": stamp(started + timedelta(seconds=duration_s)),
        "queue_s": queue_s,
        "duration_s": duration_s,
        "flags": {"eigen_cache_hit": eigen, "is_rerun": attempt > 1, "infra_failure": infra},
    }
    data.update(extra)
    return data


def analysis(
    *,
    workflow="BUILD",
    job="build-static-lto",
    event="push",
    branch="dev",
    attempt=1,
    conclusion="success",
    created=NEWEST,
    ninja="default",
    ccache="default",
    **extra,
):
    if ninja == "default":
        ninja = {
            "log_version": 7,
            "complete": True,
            "steps": 600,
            "wall_s": 900.0,
            "cpu_s": 5000.0,
            "parallelism": 5.5,
            "tail_after_compile_s": 700.0,
            "by_kind": {
                "compile": {"steps": 400, "cpu_s": 400.0},
                "archive": {"steps": 50, "cpu_s": 100.0},
                "link": {"steps": 150, "cpu_s": 4500.0},
                "other": {"steps": 0, "cpu_s": 0.0},
            },
            "slowest": [
                {"target": "tests/testA", "kind": "link", "seconds": 68.0},
                {"target": "tests/testB", "kind": "link", "seconds": 60.0},
            ],
        }
    if ccache == "default":
        ccache = {
            "hits": 140,
            "misses": 2,
            "hit_rate": 0.9859,
            "counters": {report.PCH_COUNTER: 263, "cache_size_kibibyte": 80000, "max_cache_size_kibibyte": 488281},
        }
    data = {
        "schema_version": 1,
        "kind": "build-analysis",
        "workflow": workflow,
        "run_id": next(_ids),
        "run_attempt": attempt,
        "event": event,
        "branch": branch,
        "head_sha": "a" * 40,
        "job_id": next(_ids),
        "job": job,
        "created_at": stamp(created),
        "conclusion": conclusion,
        "ninja": ninja,
        "ccache": ccache,
    }
    data.update(extra)
    return data


def days_ago(days, hours=0):
    return NEWEST - timedelta(days=days, hours=hours)


class Options:
    window_days = 14
    regression_percent = 20.0
    min_samples = 5


class DataDirTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data = Path(self.directory.name)

    def write(self, records, name="2026-W40.jsonl", raw=None):
        lines = [json.dumps(r) for r in records] + (raw or [])
        (self.data / name).write_text("\n".join(lines) + "\n")

    def render(self, records, **options):
        self.write(records)
        jobs, stats = report.load_jobs(self.data)
        opts = Options()
        for key, value in options.items():
            setattr(opts, key, value)
        return report.render(jobs, stats, opts)


class PercentileAndFormatTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(3, report.percentile([1, 2, 3, 4, 5], 50))
        self.assertEqual(2.5, report.percentile([1, 2, 3, 4], 50))
        self.assertAlmostEqual(9.1, report.percentile(list(range(1, 11)), 90))
        self.assertEqual(7, report.percentile([7], 90))

    def test_format_duration(self):
        self.assertEqual("-", report.format_duration(None))
        self.assertEqual("42s", report.format_duration(42))
        self.assertEqual("1m 05s", report.format_duration(65))
        self.assertEqual("16m 22s", report.format_duration(982.4))

    def test_iso_week_uses_the_iso_year(self):
        self.assertEqual("2026-W01", report.iso_week(datetime(2025, 12, 29, tzinfo=timezone.utc)))
        self.assertEqual("2026-W40", report.iso_week(datetime(2026, 9, 30, tzinfo=timezone.utc)))

    def test_table_cells_escape_pipes(self):
        self.assertEqual("| a\\|b |", report.table(["h"], [["a|b"]])[-1])


class CompareTests(unittest.TestCase):
    def check(self, current, previous):
        return report.compare(current, previous, regression_percent=20.0, min_samples=5)

    def test_flags_a_large_relative_and_absolute_increase(self):
        change, regression = self.check((10, 720), (10, 600))
        self.assertAlmostEqual(0.2, change)
        self.assertFalse(regression)  # exactly 20% is not "more than"
        change, regression = self.check((10, 730), (10, 600))
        self.assertTrue(regression)

    def test_a_tiny_job_is_not_flagged_for_a_large_percentage(self):
        _, regression = self.check((10, 20), (10, 10))
        self.assertFalse(regression)

    def test_too_few_samples_or_no_baseline_gives_no_verdict(self):
        self.assertEqual((None, False), self.check((4, 900), (10, 600)))
        self.assertEqual((None, False), self.check((10, 900), (4, 600)))
        self.assertEqual((None, False), self.check((10, 900), (0, None)))

    def test_improvement_is_never_a_regression(self):
        change, regression = self.check((10, 300), (10, 600))
        self.assertAlmostEqual(-0.5, change)
        self.assertFalse(regression)


class LoadTests(DataDirTestCase):
    def test_reads_all_shards_and_counts_bad_lines_without_failing(self):
        self.write([record()], raw=["not json", '{"schema_version": 1, "kind": "job"}', ""])
        self.write([record(schema_version=2), record(kind="mystery")], name="2026-W41.jsonl")
        jobs, stats = report.load_jobs(self.data)
        self.assertEqual(1, len(jobs))
        self.assertEqual(2, stats.files)
        self.assertEqual(2, stats.invalid)
        self.assertEqual(2, stats.ignored)
        self.assertEqual(5, stats.records)

    def test_build_analysis_records_are_counted_but_not_unread(self):
        self.write([record(), analysis()])
        jobs, stats = report.load_jobs(self.data)
        self.assertEqual((1, 1, 0, 0), (len(jobs), stats.analyses, stats.ignored, stats.invalid))
        self.assertEqual(1, len(stats.analysis_records))
        self.assertEqual(("BUILD", "build-static-lto"), (stats.analysis_records[0].workflow, stats.analysis_records[0].job))

    def test_analysis_records_of_another_version_or_a_broken_shape_are_not_used(self):
        self.write([record(), analysis(schema_version=2)], raw=[json.dumps({"schema_version": 1, "kind": "build-analysis"})])
        jobs, stats = report.load_jobs(self.data)
        self.assertEqual((0, 1, 1), (stats.analyses, stats.ignored, stats.invalid))
        self.assertIn("Unread records: 1 of an unknown schema version or kind, 1 malformed", report.render(jobs, stats, Options()))

    def test_ignores_files_that_are_not_shards(self):
        self.write([record()])
        (self.data / "CI_TIMINGS.md").write_text("# old report\n")
        jobs, stats = report.load_jobs(self.data)
        self.assertEqual((1, 1), (len(jobs), stats.files))


class ExclusionTests(unittest.TestCase):
    def reason(self, **fields):
        return report.exclusion_reason(report.parse_job(record(**fields)))

    def test_kept(self):
        self.assertIsNone(self.reason())
        self.assertIsNone(self.reason(event="push", branch="dev"))

    def test_each_reason(self):
        self.assertEqual("cancelled", self.reason(conclusion="cancelled"))
        self.assertEqual("rerun (attempt > 1)", self.reason(attempt=2))
        self.assertEqual("infrastructure failure", self.reason(conclusion="failure", infra=True))
        self.assertEqual("failed or other non-success conclusion", self.reason(conclusion="failure", infra=False))
        self.assertEqual("event other than push or pull_request", self.reason(event="schedule"))
        self.assertEqual("push to a branch other than dev", self.reason(event="push", branch="main"))

    def test_first_matching_reason_wins(self):
        self.assertEqual("cancelled", self.reason(conclusion="cancelled", attempt=2))
        self.assertEqual("infrastructure failure", self.reason(conclusion="failure", infra=True, attempt=2))


class JobTableTests(DataDirTestCase):
    def test_splits_by_event_and_compares_with_the_previous_window(self):
        records = []
        for i in range(6):
            records.append(record(created=days_ago(1, i), duration_s=1000))
            records.append(record(created=days_ago(20, i), duration_s=600))
            records.append(record(created=days_ago(1, i), event="push", branch="dev", duration_s=500))
        text = self.render(records)
        self.assertIn("| build | pull_request | 6 | 16m 40s | 16m 40s | 10m 00s | +67% **regression** |", text)
        self.assertIn("| build | push (dev) | 6 | 8m 20s | 8m 20s | - | n/a |", text)
        self.assertIn("- BUILD / build, pull_request: median 10m 00s to 16m 40s (+67%, 6 runs)", text)

    def test_data_older_than_the_previous_window_is_not_a_baseline(self):
        records = [record(created=days_ago(1, i), duration_s=1000) for i in range(6)]
        records += [record(created=days_ago(40, i), duration_s=100) for i in range(6)]
        text = self.render(records)
        self.assertIn("| build | pull_request | 6 | 16m 40s | 16m 40s | - | n/a |", text)

    def test_queue_time_is_reported_apart_from_duration(self):
        records = [record(created=days_ago(1, i), queue_s=90, duration_s=600) for i in range(5)]
        text = self.render(records)
        self.assertIn("| build | pull_request | 5 | 10m 00s | 10m 00s | - | n/a | 1m 30s | 1m 30s |", text)

    def test_excluded_jobs_do_not_influence_the_statistics(self):
        records = [record(created=days_ago(1, i), duration_s=600) for i in range(5)]
        records += [
            record(created=days_ago(1), duration_s=9000, conclusion="cancelled"),
            record(created=days_ago(1), duration_s=9000, attempt=2),
            record(created=days_ago(1), duration_s=9000, conclusion="failure"),
            record(created=days_ago(1), duration_s=9000, event="push", branch="main"),
        ]
        text = self.render(records)
        self.assertIn("| build | pull_request | 5 | 10m 00s | 10m 00s |", text)
        self.assertIn("| cancelled | 1 |", text)
        self.assertIn("| rerun (attempt > 1) | 1 |", text)
        self.assertIn("Kept: 5 of 9 jobs", text)

    def test_jobs_without_a_run_in_the_current_window_are_omitted(self):
        records = [record(created=days_ago(1), job="new")] + [
            record(created=days_ago(20, i), job="gone") for i in range(3)
        ]
        text = self.render(records)
        self.assertIn("| new | pull_request |", text)
        self.assertNotIn("| gone |", text)

    def test_window_length_is_configurable(self):
        records = [record(created=NEWEST, job="latest"), record(created=days_ago(10), job="older")]
        self.assertIn("| older | pull_request |", self.render(records, window_days=14))
        self.assertNotIn("| older | pull_request |", self.render(records, window_days=7))


class CriticalPathTests(DataDirTestCase):
    def run_of(self, run_id, created, jobs, **common):
        """jobs: list of (name, offset_s_from_created, queue_s, duration_s)."""
        return [
            record(
                run_id=run_id,
                job=name,
                created=created + timedelta(seconds=offset),
                queue_s=queue,
                duration_s=duration,
                **common,
            )
            for name, offset, queue, duration in jobs
        ]

    def test_wall_clock_spans_the_dependency_chain_and_names_the_last_job(self):
        records = []
        for i in range(5):
            records += self.run_of(
                1000 + i,
                days_ago(1, i),
                [("changes", 0, 2, 8), ("build", 10, 5, 800), ("lint", 10, 5, 100), ("ci-build-gate", 815, 2, 3)],
            )
        text = self.render(records)
        self.assertIn(
            "| BUILD | pull_request | 5 | 13m 40s | 13m 40s | - | n/a | ci-build-gate (100%) |", text
        )

    def test_runs_with_any_failed_job_or_a_second_attempt_are_dropped(self):
        ok = [("changes", 0, 1, 5), ("build", 6, 1, 500)]
        records = []
        for i in range(5):
            records += self.run_of(2000 + i, days_ago(1, i), ok)
        records += self.run_of(3000, days_ago(1), ok + [("lint", 6, 1, 5)])
        records[-1]["conclusion"] = "failure"
        records += self.run_of(3001, days_ago(1), [("build", 0, 1, 99999)], attempt=2)
        text = self.render(records)
        self.assertIn("| BUILD | pull_request | 5 | 8m 27s |", text)
        self.assertIn("| run with a non-successful job | 1 |", text)

    def test_runs_where_the_paths_filter_skipped_every_real_job_are_dropped(self):
        records = []
        for i in range(5):
            records += self.run_of(4000 + i, days_ago(1, i), [("changes", 0, 1, 5), ("build", 6, 1, 600), ("ci-build-gate", 610, 1, 2)])
        for i in range(3):
            records += self.run_of(5000 + i, days_ago(2, i), [("changes", 0, 1, 5), ("ci-build-gate", 8, 1, 2)])
        text = self.render(records)
        self.assertIn("| BUILD | pull_request | 5 | 10m 13s |", text)
        self.assertIn("| run where the paths filter skipped every real job | 3 |", text)

    def test_a_workflow_with_a_single_ungated_job_is_kept(self):
        records = [record(workflow="Docs", job="validate", created=days_ago(1, i), duration_s=300) for i in range(5)]
        self.assertIn("| Docs | pull_request | 5 | 5m 03s |", self.render(records))

    def test_push_runs_to_other_branches_are_not_counted(self):
        records = [record(event="push", branch="main", created=days_ago(1, i)) for i in range(5)]
        text = self.render(records)
        self.assertNotIn("| BUILD | push (dev) |", text)


class EigenTests(DataDirTestCase):
    def test_hit_rate_per_week_and_event(self):
        week = NEWEST
        records = [
            record(created=week, eigen=True, event="pull_request"),
            record(created=week, eigen=True, event="pull_request"),
            record(created=week, eigen=False, event="pull_request"),
            record(created=week, eigen=False, event="push", branch="dev"),
            record(created=week, eigen=None),
        ]
        text = self.render(records)
        self.assertIn("| 2026-W40 | 0% (1) | 67% (3) | 50% (4) |", text)
        self.assertIn('line [50]', text)

    def test_no_eigen_steps_yet(self):
        self.assertIn("No jobs with the Eigen cache steps yet.", self.render([record()]))


class TrendTests(DataDirTestCase):
    def test_chart_has_one_point_per_week_with_enough_runs(self):
        records = []
        for week in range(3):
            for i in range(3):
                records.append(record(created=days_ago(7 * week, i), duration_s=600, run_id=100 * week + i))
        records.append(record(created=days_ago(40), run_id=9999))  # one lonely run: week omitted
        text = self.render(records)
        self.assertIn("xychart-beta", text)
        self.assertIn('x-axis ["W38", "W39", "W40"]', text)
        self.assertIn("line [10.1, 10.1, 10.1]", text)

    def test_no_chart_without_enough_runs(self):
        self.assertIn("Not enough BUILD runs yet", self.render([record()]))


def builds(count, **fields):
    """`count` analyses spread over the last days (one every 6 hours)."""
    return [analysis(created=days_ago(0, 6 * i + 1), **fields) for i in range(count)]


class BuildAnalysisSectionTests(DataDirTestCase):
    def render_with(self, analyses, extra_jobs=(), **options):
        return self.render([record(created=NEWEST), *extra_jobs, *analyses], **options)

    def test_without_records_it_says_so(self):
        text = self.render([record()])
        self.assertIn("## Build analysis", text)
        self.assertIn("No build analysis records yet", text)
        self.assertNotIn("### ccache", text)

    def test_ccache_row_has_hit_rate_pch_blocked_calls_and_cache_full(self):
        full = {"hits": 179, "misses": 2, "hit_rate": 0.989, "counters": {report.PCH_COUNTER: 269, "cache_size_kibibyte": 488692, "max_cache_size_kibibyte": 488281}}
        text = self.render_with(builds(5, job="lint", ccache=full))
        self.assertIn("| BUILD / lint | push (dev) | 5 | 99% | - | n/a | 269 | 60% | 5/5 |", text)

    def test_cache_full_means_at_least_95_percent_of_the_limit(self):
        def with_size(kib):
            return {"hits": 1, "misses": 1, "hit_rate": 0.5, "counters": {"cache_size_kibibyte": kib, "max_cache_size_kibibyte": 1000}}

        rows = report.ccache_rows(
            [report.parse_analysis(analysis(job=f"j{kib}", ccache=with_size(kib))) for kib in (900, 949, 950, 1000)],
            report.Windows(NEWEST, 14),
            Options(),
        )
        self.assertEqual({"j900": "0/1", "j949": "0/1", "j950": "1/1", "j1000": "1/1"}, {r[0].split(" / ")[1]: r[-1] for r in rows})

    def test_hit_rate_is_compared_with_the_previous_window_in_points(self):
        now = [analysis(created=days_ago(1, i), ccache={"hits": 90, "misses": 10, "hit_rate": 0.9, "counters": {}}) for i in range(5)]
        before = [analysis(created=days_ago(20, i), ccache={"hits": 80, "misses": 20, "hit_rate": 0.8, "counters": {}}) for i in range(5)]
        text = self.render_with(now + before)
        self.assertIn("| BUILD / build-static-lto | push (dev) | 5 | 90% | 80% | +10.0 pp | 0 | 0% | - |", text)

    def test_builds_without_ccache_or_cacheable_calls_are_handled(self):
        empty = {"hits": 0, "misses": 0, "hit_rate": None, "counters": {}}
        text = self.render_with(builds(2, ccache=None) + builds(2, job="idle", ccache=empty))
        self.assertIn("| BUILD / idle | push (dev) | 2 | - | - | n/a | 0 | - | - |", text)
        self.assertNotIn("BUILD / build-static-lto | push (dev) | 2 |", text.split("### Ninja builds")[0])

    def test_excluded_analyses_are_not_counted(self):
        extra = [
            analysis(attempt=2),
            analysis(conclusion="cancelled"),
            analysis(branch="main"),
            analysis(event="schedule"),
        ]
        text = self.render_with(builds(2) + extra)
        self.assertIn("From 2 build analysis records", text)

    def test_ninja_row_uses_complete_builds_only_and_reports_the_rest(self):
        partial = dict(analysis()["ninja"], complete=False, wall_s=1.0)
        unknown = dict(analysis()["ninja"], complete=None, wall_s=2.0)
        text = self.render_with(builds(3) + builds(2, job="lint", ninja=partial) + builds(1, job="x", ninja=unknown))
        self.assertIn("| BUILD / build-static-lto | push (dev) | 3 | 15m 00s | 83m 20s | 5.5 | 11m 40s | 90% |", text)
        self.assertNotIn("| BUILD / lint | push (dev) | 2 | 1", text.split("### Slowest")[0].split("### Ninja builds")[1])
        self.assertIn("not full builds: 3 with `complete` false or unknown", text)

    def test_a_build_with_an_error_is_ignored(self):
        broken = {"log_version": 4, "complete": None, "steps": 0, "error": "no usable .ninja_log"}
        text = self.render_with(builds(2, ninja=broken))
        self.assertIn("No complete ninja builds in the current window.", text)
        self.assertIn("No ninja builds of pushes to `dev`", text)

    def test_slowest_steps_aggregate_over_dev_pushes_only(self):
        def with_times(a, b):
            ninja = dict(analysis()["ninja"])
            ninja["slowest"] = [
                {"target": "tests/testA", "kind": "link", "seconds": a},
                {"target": "tests/testB", "kind": "link", "seconds": b},
            ]
            return ninja

        pushes = [analysis(created=days_ago(1, i), ninja=with_times(60 + i, 50)) for i in range(3)]
        pushes.append(analysis(created=days_ago(1), ninja=dict(with_times(10, 10), slowest=[{"target": "tests/testB", "kind": "link", "seconds": 10.0}])))
        pr = analysis(event="pull_request", branch="feature/x", ninja=with_times(999, 999))
        text = self.render_with(pushes + [pr])
        section = text.split("### Slowest build steps on `dev`")[1]
        self.assertIn("#### BUILD / build-static-lto (4 of 4 builds complete)", section)
        self.assertIn("| `tests/testA` | link | 3/4 | 1m 01s | 1m 02s |", section)
        self.assertIn("| `tests/testB` | link | 4/4 | 50.0s | 50.0s |", section)
        self.assertNotIn("999", section)
        self.assertLess(section.index("testA"), section.index("testB"))

    def test_slowest_steps_are_capped(self):
        ninja = dict(analysis()["ninja"])
        ninja["slowest"] = [{"target": f"t{i:02d}", "kind": "compile", "seconds": 20.0 - i} for i in range(20)]
        text = self.render_with(builds(1, ninja=ninja))
        section = text.split("### Slowest build steps on `dev`")[1]
        self.assertEqual(report.SLOWEST_SHOWN, section.count("| `t"))

    def test_partial_builds_still_show_their_slowest_steps_with_the_complete_count(self):
        partial = dict(analysis()["ninja"], complete=False)
        text = self.render_with(builds(3, job="lint", ninja=partial))
        self.assertIn("#### BUILD / lint (0 of 3 builds complete)", text)

    def test_weekly_chart_needs_enough_builds_per_week(self):
        many = [analysis(created=days_ago(0, i)) for i in range(3)] + [analysis(created=days_ago(7, i)) for i in range(3)] + [analysis(created=days_ago(14))]
        text = self.render_with(many)
        self.assertIn('title "Share of compiler calls blocked by the PCH, weekly median (%)"', text)
        self.assertIn('x-axis ["W39", "W40"]', text)
        self.assertIn("line [64.9, 64.9]", text)
        self.assertNotIn("Share of compiler calls blocked", self.render_with(builds(2)))

    def test_old_analyses_fall_out_of_the_current_window(self):
        text = self.render_with([analysis(created=days_ago(20))])
        self.assertIn("From 1 build analysis records", text)
        self.assertIn("No ccache statistics in the current window.", text)


def clang_summary(total=1000.0, headers=(), files=(), frontend=None, backend=None, source_events=50000, instantiation_events=300000):
    return {
        "schema_version": 1,
        "kind": "clang-trace-summary",
        "limits": {"files": 20, "headers": 30, "templates": 30},
        "files": 400,
        "unreadable": 0,
        "total_s": total,
        "frontend_s": frontend if frontend is not None else total * 0.6,
        "backend_s": backend if backend is not None else total * 0.4,
        "source_events": source_events,
        "instantiation_events": instantiation_events,
        "slowest_files": [{"file": n, "total_s": v, "frontend_s": v / 2, "backend_s": v / 2} for n, v in files],
        "headers": [{"header": n, "inclusive_s": v, "self_s": v, "events": 10, "files": 10} for n, v in headers],
        "templates": [],
    }


def include_summary(pairs=1000, project=300, digest="d" * 64, top=()):
    return {
        "objects": 403,
        "unique_files": 100,
        "include_pairs": pairs,
        "project_files": 50,
        "project_pairs": project,
        "digest": digest,
        "top_project_files": [{"file": f, "fan_in": n} for f, n in top],
    }


def clang_builds(count, start_hours=1, step_hours=6, job="clang-build", **fields):
    return [
        analysis(workflow="Clang Build", job=job, created=days_ago(0, start_hours + step_hours * i), **fields)
        for i in range(count)
    ]


class ClangSectionTests(DataDirTestCase):
    def render_with(self, analyses, extra_jobs=(), **options):
        return self.render([record(created=NEWEST), *extra_jobs, *analyses], **options)

    def section(self, text):
        return text.split("### Clang build times")[1].split("### Include graph")[0]

    def test_without_clang_summaries_it_says_so(self):
        text = self.render_with([analysis()])
        self.assertIn("No clang trace summaries yet", self.section(text))

    def test_medians_of_the_current_window_and_the_change_against_the_previous_one(self):
        now = [analysis(workflow="Clang Build", created=days_ago(1, i), clang=clang_summary(total=t)) for i, t in enumerate((1000.0, 1200.0, 1100.0, 1100.0, 1100.0))]
        before = [analysis(workflow="Clang Build", created=days_ago(20, i), clang=clang_summary(total=900.0)) for i in range(5)]
        text = self.section(self.render_with(now + before))
        self.assertIn("5 builds in the current window, 5 in the previous one (medians).", text)
        self.assertIn("| Compiler time | 18m 20s | 15m 00s | +22.2% |", text)
        self.assertIn("| Header inclusions (events of at least 0.5 ms) | 50,000 | 50,000 | n/a |", text)

    def test_too_few_builds_give_no_change(self):
        now = [analysis(workflow="Clang Build", created=days_ago(1, i), clang=clang_summary(total=1100.0)) for i in range(2)]
        before = [analysis(workflow="Clang Build", created=days_ago(20), clang=clang_summary(total=900.0))]
        self.assertIn("| Compiler time | 18m 20s | 15m 00s | n/a |", self.section(self.render_with(now + before)))

    def test_only_pushes_to_dev_count(self):
        pr = analysis(workflow="Clang Build", event="pull_request", branch="feature/x", clang=clang_summary(total=9999.0))
        other = analysis(workflow="Clang Build", branch="main", clang=clang_summary(total=9999.0))
        text = self.section(self.render_with(clang_builds(3, clang=clang_summary(total=1000.0)) + [pr, other]))
        self.assertIn("3 builds in the current window", text)
        self.assertIn("| Compiler time | 16m 40s |", text)

    def test_headers_are_ranked_by_their_share_of_the_total_not_by_seconds(self):
        slow = analysis(workflow="Clang Build", created=days_ago(0, 1), clang=clang_summary(total=2000.0, headers=[("include/x.hpp", 100.0), ("include/y.hpp", 90.0)]))
        fast = analysis(workflow="Clang Build", created=days_ago(0, 7), clang=clang_summary(total=1000.0, headers=[("include/x.hpp", 50.0), ("include/y.hpp", 49.0)]))
        text = self.section(self.render_with([slow, fast]))
        rows = [line for line in text.split("#### Slowest files")[0].splitlines() if line.startswith("| `")]
        self.assertTrue(rows[0].startswith("| `include/x.hpp` | 2/2 | 1m 15s | 5.00% |"), rows[0])
        self.assertTrue(rows[1].startswith("| `include/y.hpp` | 2/2 | "), rows[1])
        self.assertIn("4.70%", rows[1])

    def test_a_header_with_more_seconds_but_a_smaller_share_ranks_lower(self):
        slow = [analysis(workflow="Clang Build", created=days_ago(0, 1 + 6 * i), clang=clang_summary(total=3000.0, headers=[("include/a.hpp", 120.0)])) for i in range(3)]
        fast = [analysis(workflow="Clang Build", created=days_ago(1, 1 + 6 * i), clang=clang_summary(total=1000.0, headers=[("include/b.hpp", 50.0)])) for i in range(3)]
        text = self.section(self.render_with(slow + fast))
        rows = [line for line in text.split("#### Slowest files")[0].splitlines() if line.startswith("| `")]
        self.assertTrue(rows[0].startswith("| `include/b.hpp` | 3/6 | 50.0s | 5.00% |"), rows[0])
        self.assertTrue(rows[1].startswith("| `include/a.hpp` | 3/6 | 2m 00s | 4.00% |"), rows[1])

    def test_a_header_missing_from_some_builds_shows_how_often_it_appeared(self):
        builds = [analysis(workflow="Clang Build", created=days_ago(0, 1 + 6 * i), clang=clang_summary(total=1000.0, headers=[("include/rare.hpp", 30.0)] if i == 0 else [])) for i in range(3)]
        self.assertIn("| `include/rare.hpp` | 1/3 |", self.section(self.render_with(builds)))

    def test_share_change_needs_enough_builds_in_both_windows(self):
        now = [analysis(workflow="Clang Build", created=days_ago(1, i), clang=clang_summary(total=1000.0, headers=[("include/h.hpp", 50.0)])) for i in range(5)]
        before = [analysis(workflow="Clang Build", created=days_ago(20, i), clang=clang_summary(total=1000.0, headers=[("include/h.hpp", 40.0)])) for i in range(5)]
        text = self.section(self.render_with(now + before))
        self.assertIn("| `include/h.hpp` | 5/5 | 50.0s | 5.00% | 4.00% | +1.00 pp |", text)
        few = self.section(self.render_with(now + before[:2]))
        self.assertIn("| `include/h.hpp` | 5/5 | 50.0s | 5.00% | 4.00% | n/a |", few)

    def test_slowest_files_are_listed_with_their_share(self):
        builds = clang_builds(3, clang=clang_summary(total=1000.0, files=[("src/a.cpp", 20.0), ("src/b.cpp", 10.0)]))
        text = self.section(self.render_with(builds))
        files = text.split("#### Slowest files")[1]
        self.assertIn("| `src/a.cpp` | 3/3 | 20.0s | 2.00% |", files)
        self.assertLess(files.index("src/a.cpp"), files.index("src/b.cpp"))

    def test_the_weekly_chart_needs_enough_builds(self):
        text = self.section(self.render_with(clang_builds(3, clang=clang_summary(total=1200.0))))
        self.assertIn('title "Clang compiler time, dev pushes, weekly median (minutes)"', text)
        self.assertIn("line [20]", text)
        self.assertNotIn("xychart-beta", self.section(self.render_with(clang_builds(2, clang=clang_summary()))))

    def test_hostile_names_are_made_safe(self):
        evil = "x|y`z\n</details>"
        text = self.section(self.render_with(clang_builds(3, clang=clang_summary(headers=[(evil, 30.0)], files=[(evil, 9.0)]))))
        rows = [line for line in text.splitlines() if "x/y" in line]
        self.assertEqual(2, len(rows))
        for row in rows:
            self.assertEqual(5 if "Share" in row or row.count("|") == 5 else row.count("|"), row.count("|"))
            self.assertNotIn("`z", row)

    def test_old_records_without_the_new_parts_do_not_break_anything(self):
        old = analysis()
        old.pop("includes", None)
        old.pop("clang", None)
        text = self.render_with([old])
        self.assertIn("No clang trace summaries yet", text)
        self.assertIn("No include graph records yet", text)


class IncludeSectionTests(DataDirTestCase):
    def render_with(self, analyses):
        return self.render([record(created=NEWEST), *analyses])

    def section(self, text):
        return text.split("### Include graph (exact)")[1].split("## Eigen cache")[0]

    def test_latest_values_and_the_exact_change_against_the_previous_window(self):
        builds = [
            analysis(workflow="Clang Build", job="clang-build", created=days_ago(1), includes=include_summary(1030, 330, "a" * 64)),
            analysis(workflow="Clang Build", job="clang-build", created=days_ago(2), includes=include_summary(1000, 300, "b" * 64)),
            analysis(workflow="Clang Build", job="clang-build", created=days_ago(20), includes=include_summary(990, 295, "c" * 64)),
            analysis(workflow="Clang Build", job="clang-build", created=days_ago(21), includes=include_summary(900, 250, "d" * 64)),
        ]
        text = self.section(self.render_with(builds))
        self.assertIn("| Clang Build / clang-build | 2 | 1,030 | +40 | 330 | +35 | 2 |", text)

    def test_one_distinct_graph_and_no_previous_window(self):
        builds = [analysis(workflow="Clang Build", job="clang-build", created=days_ago(1, i), includes=include_summary()) for i in range(3)]
        self.assertIn("| Clang Build / clang-build | 3 | 1,000 | n/a | 300 | n/a | 1 |", self.section(self.render_with(builds)))

    def test_jobs_are_listed_separately_and_only_dev_pushes_count(self):
        builds = [
            analysis(job="build-static-lto", created=days_ago(1), includes=include_summary(500, 100)),
            analysis(job="lint", created=days_ago(1), includes=include_summary(700, 200)),
            analysis(job="lint", event="pull_request", branch="feature/x", created=days_ago(0, 1), includes=include_summary(9999, 9999)),
            analysis(job="lint", branch="main", created=days_ago(0, 2), includes=include_summary(9999, 9999)),
        ]
        text = self.section(self.render_with(builds))
        self.assertIn("| BUILD / build-static-lto | 1 | 500 |", text)
        self.assertIn("| BUILD / lint | 1 | 700 |", text)
        self.assertNotIn("9,999", text)

    def test_the_highest_fan_in_headers_of_the_latest_clang_build(self):
        old = analysis(workflow="Clang Build", job="clang-build", created=days_ago(3), includes=include_summary(top=[("include/old.hpp", 1)]))
        new = analysis(workflow="Clang Build", job="clang-build", created=days_ago(1), includes=include_summary(top=[("include/a.hpp", 300), ("include/b.hpp", 120)]))
        text = self.section(self.render_with([old, new]))
        self.assertIn("#### Project headers with the highest fan-in (latest clang build)", text)
        self.assertIn("| `include/a.hpp` | 300 |", text)
        self.assertLess(text.index("include/a.hpp"), text.index("include/b.hpp"))
        self.assertNotIn("old.hpp", text)

    def test_without_records_it_says_so(self):
        self.assertIn("No include graph records yet", self.section(self.render_with([analysis()])))

    def test_older_than_the_current_window_gives_no_current_graph(self):
        build = analysis(workflow="Clang Build", job="clang-build", created=days_ago(20), includes=include_summary())
        self.assertIn("No include graph in the current window.", self.section(self.render_with([build])))


class NonGatingTests(DataDirTestCase):
    def test_the_clang_workflow_is_not_in_the_critical_path_table_but_in_the_job_table(self):
        records = []
        for i in range(5):
            run = 7000 + i
            records.append(record(workflow="Clang Build", job="clang-build", run_id=run, created=days_ago(1, i), duration_s=500))
            records.append(record(workflow="BUILD", job="build", run_id=run + 100, created=days_ago(1, i), duration_s=500))
        text = self.render(records)
        critical = text.split("## Wall-clock per workflow (critical path)")[1].split("## Trend")[0]
        self.assertNotIn("Clang Build", critical.split("| Workflow |")[1])
        self.assertIn("Non-gating workflows (Clang Build)", critical)
        jobs = text.split("## Jobs")[1]
        self.assertIn("### Clang Build", jobs)


class AnalysisParsingTests(unittest.TestCase):
    def test_includes_and_clang_are_parsed_and_optional(self):
        parsed = report.parse_analysis(analysis(includes=include_summary(), clang=clang_summary()))
        self.assertEqual(1000, parsed.includes["include_pairs"])
        self.assertEqual(1000.0, parsed.clang["total_s"])
        data = analysis()
        data.pop("includes", None); data.pop("clang", None)
        old = report.parse_analysis(data)
        self.assertEqual((None, None), (old.includes, old.clang))


class RenderTests(DataDirTestCase):
    def test_output_is_deterministic(self):
        records = [record(created=days_ago(1, i), eigen=i % 2 == 0) for i in range(8)]
        self.assertEqual(self.render(records), self.render(list(reversed(records))))

    def test_nothing_excluded_says_so_instead_of_an_empty_table(self):
        text = self.render([record(created=days_ago(1, i)) for i in range(5)])
        self.assertIn("Kept: 5 of 5 jobs", text)
        self.assertNotIn("| Reason | Jobs |", text)
        self.assertNotIn("| Reason | Runs |", text)

    def test_no_data_renders_a_stub(self):
        self.write([])
        jobs, stats = report.load_jobs(self.data)
        self.assertIn("No timing data found yet", report.render(jobs, stats, Options()))

    def test_unread_records_are_mentioned(self):
        self.write([record()], raw=["garbage"])
        jobs, stats = report.load_jobs(self.data)
        self.assertIn("1 malformed", report.render(jobs, stats, Options()))


class MainTests(DataDirTestCase):
    def run_main(self, *args):
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = report.main(["--data-dir", str(self.data), *args])
            except SystemExit as exit_:
                code = exit_.code
        return code, out.getvalue(), err.getvalue()

    def test_writes_next_to_the_data_and_is_a_no_op_the_second_time(self):
        self.write([record(created=days_ago(1, i)) for i in range(5)])
        code, out, _ = self.run_main()
        self.assertEqual(0, code)
        self.assertIn("wrote", out)
        page = self.data / "CI_TIMINGS.md"
        first = page.read_bytes()
        mtime = page.stat().st_mtime_ns
        code, out, _ = self.run_main()
        self.assertEqual(0, code)
        self.assertIn("unchanged", out)
        self.assertEqual((first, mtime), (page.read_bytes(), page.stat().st_mtime_ns))
        self.assertFalse((self.data / "CI_TIMINGS.md.tmp").exists())

    def test_custom_output_path(self):
        self.write([record()])
        target = self.data / "elsewhere.md"
        self.assertEqual(0, self.run_main("--out", str(target))[0])
        self.assertTrue(target.exists())
        self.assertFalse((self.data / "CI_TIMINGS.md").exists())

    def test_missing_data_directory_is_an_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = report.main(["--data-dir", str(self.data / "nope")])
        self.assertEqual(1, code)
        self.assertIn("data directory not found", err.getvalue())

    def test_rejects_nonsensical_options(self):
        self.write([record()])
        for bad in (["--window-days", "0"], ["--min-samples", "0"], ["--regression-percent", "-1"]):
            code, _, _ = self.run_main(*bad)
            self.assertEqual(2, code, bad)


if __name__ == "__main__":
    unittest.main()
