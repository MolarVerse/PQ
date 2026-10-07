import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

TOOL_DIR = Path(__file__).resolve().parents[1] / "pq_code_stats"
sys.path.insert(0, str(TOOL_DIR))

import cs_ci as ci  # noqa: E402
import cs_classify as classify  # noqa: E402
import cs_history as history  # noqa: E402
import cs_output as output  # noqa: E402
import cs_report as report  # noqa: E402
import pqstats  # noqa: E402

GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_AUTHOR_NAME": "Tester",
           "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "Tester", "GIT_COMMITTER_EMAIL": "t@example.invalid"}


def lines(text, count):
    return (text + "\n") * count


class Repo:
    """A throwaway git repository with a dev branch, two merged PRs and a direct commit (dates in Jan 2026)."""

    def __init__(self, path):
        self.path = Path(path)

    def git(self, *args, when=None):
        env = dict(os.environ, **GIT_ENV)
        if when:
            env.update(GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when)
        subprocess.run(["git", "-C", str(self.path), *args], check=True, capture_output=True, env=env)

    def write(self, name, content):
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode())

    def commit(self, message, when):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message, when=when)

    def build(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "dev")
        self.write("src/a.cpp", lines("x", 10))
        self.write("include/a.hpp", lines("h", 5))
        self.write("tests/t.cpp", lines("t", 4))
        self.write(".github/workflows/ci.yml", lines("c", 6))
        self.write("external/lib/x.hpp", lines("e", 50))
        self.write("README.md", lines("r", 3))
        self.commit("Initial commit", "2026-01-05T10:00:00+00:00")

        self.git("checkout", "-q", "-b", "feature1")
        self.write("src/a.cpp", "y\n" + lines("x", 9) + lines("z", 3))
        self.write("include/b.tpp", lines("b", 7))
        self.write("tests/t2.cpp", lines("u", 5))
        self.write("integration_tests/data.dat", lines("d", 100))
        self.commit("work on feature1", "2026-01-06T10:00:00+00:00")
        self.git("checkout", "-q", "dev")
        self.git("merge", "-q", "--no-ff", "-m", "Merge pull request #1 from owner/feature1", "-m", "Add b template", "feature1",
                 when="2026-01-07T12:00:00+00:00")

        self.git("checkout", "-q", "-b", "feature2")
        (self.path / "tests/t.cpp").unlink()
        self.write(".github/workflows/ci.yml", lines("c", 8))
        self.write("benchmarks/perf/p.cpp", lines("p", 3))
        self.write("apps/main.cpp", lines("m", 8))
        self.write("src/blob.bin", b"\x00\x01\x02\x03" * 10)
        self.commit("work on feature2", "2026-01-19T10:00:00+00:00")
        self.git("checkout", "-q", "dev")
        self.git("merge", "-q", "--no-ff", "-m", "Merge pull request #2 from owner/feature2", "-m", "Remove t, add perf", "feature2",
                 when="2026-01-20T09:00:00+00:00")

        self.write("docs/readme.md", lines("d", 2))
        self.commit("Update docs", "2026-01-21T08:00:00+00:00")
        return self


class FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.repo = Repo(Path(cls._tmp.name) / "repo").build()
        cls.records = history.read_history(cls.repo.path, "dev")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def record(self, pr=None, kind=None):
        return next(r for r in self.records if (pr is None or r.change.pr == pr) and (kind is None or r.change.kind == kind))


class ClassifyTests(unittest.TestCase):
    def test_areas_groups_and_kinds(self):
        expected = {
            "src/a.cpp": ("src", "production", "source"), "src/x/CMakeLists.txt": ("src", "production", "cmake"),
            "include/a.hpp": ("include", "production", "header"), "include/a.tpp": ("include", "production", "tpp"),
            "apps/main.cpp": ("apps", "production", "source"), "tests/src/t.cpp": ("tests", "tests", "source"),
            "integration_tests/x/ref.xyz": ("integration_tests", "tests", "other"),
            "benchmarks/perf/p.cpp": ("perf", "perf", "source"), "benchmarks/src/b.cpp": ("benchmarks", "perf", "source"),
            ".github/workflows/a.yml": ("ci_workflows", "ci", "config"), ".github/actions/x/action.yml": ("ci_other", "ci", "config"),
            ".github/ci-metrics/data/2026-W01.jsonl": ("ci_data", "data", "config"),
            "scripts/x.py": ("scripts", "other", "python"), "docs/index.rst": ("docs", "docs", "doc"),
            "changes/user/bugfix.x.md": ("changelog", "docs", "doc"), "CHANGELOG.md": ("changelog", "docs", "doc"),
            "CMakeLists.txt": ("cmake", "build", "cmake"), ".cmake/eigen.cmake": ("cmake", "build", "cmake"),
            "README.md": ("other", "other", "doc"), "weird.xyz": ("other", "other", "other"),
        }
        for path, result in expected.items():
            self.assertEqual(result, classify.classify(path), path)

    def test_submodules_are_not_counted(self):
        self.assertIsNone(classify.classify("external/mstd/include/x.hpp"))

    def test_the_more_specific_prefix_wins(self):
        self.assertEqual("perf", classify.classify("benchmarks/perf/x.cpp")[0])
        self.assertEqual("benchmarks", classify.classify("benchmarks/x.cpp")[0])
        self.assertEqual("ci_data", classify.classify(".github/ci-metrics/data/x.jsonl")[0])
        self.assertEqual("ci_other", classify.classify(".github/ci-metrics/collect.py")[0])

    def test_header_like_combines_header_and_tpp(self):
        self.assertEqual(["header_like", "header_like", "source", "other"],
                         [classify.class_of(k) for k in ("header", "tpp", "source", "python")])


class ParseTests(unittest.TestCase):
    def block(self, sha, parents, subject, body="", files=""):
        return f"\x01{sha}\x00{parents}\x002026-01-07T14:00:00+02:00\x00Ann\x00{subject}\x00{body}\x02\n\n{files}"

    def test_change_kinds_titles_and_branches(self):
        text = (
            self.block("a" * 40, "p1 p2", "Merge pull request #7 from owner/feature/x-1", "Add the thing\n\nmore")
            + self.block("b" * 40, "p1", "Fix the bug (#12)")
            + self.block("c" * 40, "p1 p2", "Merge branch 'main' into dev")
            + self.block("d" * 40, "p1", "Update README.md")
        )
        changes = [change for change, _ in history.parse_log(text)]
        self.assertEqual(["pr", "pr", "merge", "commit"], [c.kind for c in changes])
        self.assertEqual([7, 12, None, None], [c.pr for c in changes])
        self.assertEqual("Add the thing", changes[0].title)
        self.assertEqual("feature/x-1", changes[0].branch)
        self.assertEqual("Fix the bug", changes[1].title)
        self.assertEqual(datetime(2026, 1, 7, 12, 0, tzinfo=timezone.utc), changes[0].date)

    def test_files_pair_status_with_numbers_and_binary_files_count_no_lines(self):
        files = (":100644 100644 aaa bbb M\tsrc/a.cpp\n:000000 100644 000 ccc A\tsrc/b.bin\n:100644 000000 ddd 000 D\told.hpp\n"
                 "\n3\t1\tsrc/a.cpp\n-\t-\tsrc/b.bin\n0\t9\told.hpp\n")
        (_, changed), = history.parse_log(self.block("a" * 40, "p1", "x", files=files))
        self.assertEqual({"src/a.cpp": ("M", 3, 1), "src/b.bin": ("A", 0, 0), "old.hpp": ("D", 0, 9)}, changed)

    def test_a_broken_header_is_reported(self):
        with self.assertRaises(history.HistoryError):
            history.parse_log("\x01only\x00three\x00fields\x02")


class SplitAndRollTests(unittest.TestCase):
    def test_the_split_counts_files_and_lines_per_area_and_kind(self):
        changed = {"src/a.cpp": ("M", 4, 1), "src/n.cpp": ("A", 6, 0), "src/o.cpp": ("D", 0, 9), "include/h.hpp": ("A", 2, 0),
                   "external/x.hpp": ("A", 99, 0)}
        splits = history.split_of_change(changed)
        self.assertEqual(history.Split(1, 1, 1, 10, 10), splits[("src", "source")])
        self.assertEqual(history.Split(1, 0, 0, 2, 0), splits[("include", "header")])
        self.assertEqual(2, len(splits))   # the submodule is not counted

    def test_the_state_is_rolled_forward_oldest_first(self):
        change = lambda n: history.Change(str(n), [], datetime(2026, 1, n, tzinfo=timezone.utc), "a", "s", "commit", None, "s", None)
        entries = [  # newest first, like git log
            (change(3), {"src/a.cpp": ("D", 0, 12)}),
            (change(2), {"src/a.cpp": ("M", 5, 3), "src/b.cpp": ("A", 4, 0)}),
            (change(1), {"src/a.cpp": ("A", 10, 0)}),
        ]
        records = history.roll_forward(entries)
        self.assertEqual(["1", "2", "3"], [r.change.sha for r in records])
        self.assertEqual({("src", "source"): (1, 10)}, records[0].state)
        self.assertEqual({("src", "source"): (2, 16)}, records[1].state)
        self.assertEqual({("src", "source"): (1, 4)}, records[2].state)

    def test_empty_states_are_dropped(self):
        change = lambda n: history.Change(str(n), [], datetime(2026, 1, n, tzinfo=timezone.utc), "a", "s", "commit", None, "s", None)
        records = history.roll_forward([(change(2), {"src/a.cpp": ("D", 0, 5)}), (change(1), {"src/a.cpp": ("A", 5, 0)})])
        self.assertEqual({}, records[1].state)


class RepoHistoryTests(FixtureCase):
    def test_the_changes_in_order_with_their_kinds(self):
        self.assertEqual(["commit", "pr", "pr", "commit"], [r.change.kind for r in self.records])
        self.assertEqual([None, 1, 2, None], [r.change.pr for r in self.records])
        self.assertEqual("Add b template", self.record(pr=1).change.title)
        self.assertEqual("feature2", self.record(pr=2).change.branch)

    def test_a_merged_pr_is_its_net_change_against_the_first_parent(self):
        splits = self.record(pr=1).splits
        self.assertEqual(history.Split(0, 0, 1, 4, 1), splits[("src", "source")])
        self.assertEqual(history.Split(1, 0, 0, 7, 0), splits[("include", "tpp")])
        self.assertEqual(history.Split(1, 0, 0, 5, 0), splits[("tests", "source")])
        self.assertEqual(history.Split(1, 0, 0, 100, 0), splits[("integration_tests", "other")])
        self.assertNotIn(("tests", "other"), splits)

    def test_deleted_files_binary_files_and_the_submodule(self):
        splits = self.record(pr=2).splits
        self.assertEqual(history.Split(0, 1, 0, 0, 4), splits[("tests", "source")])
        self.assertEqual(history.Split(1, 0, 0, 0, 0), splits[("src", "other")])   # binary: a file, no lines
        self.assertEqual(history.Split(0, 0, 1, 2, 0), splits[("ci_workflows", "config")])
        self.assertTrue(all(area != "external" for area, _ in self.record(kind="commit").state))

    def test_the_final_state_and_its_verification(self):
        final = self.records[-1].state
        self.assertEqual((1, 13), final[("src", "source")])
        self.assertEqual((1, 7), final[("include", "tpp")])
        self.assertEqual((1, 5), final[("tests", "source")])
        self.assertEqual((1, 100), final[("integration_tests", "other")])
        self.assertEqual((1, 2), final[("docs", "doc")])
        self.assertEqual({}, {k: v for k, v in final.items() if v[0] < 0 or v[1] < 0})
        counted = history.count_tree(self.repo.path, "dev")
        self.assertEqual(final, counted)
        self.assertEqual([], history.verify(self.records, counted))

    def test_verification_reports_a_difference(self):
        counted = dict(history.count_tree(self.repo.path, "dev"))
        counted[("src", "source")] = (1, 14)
        self.assertEqual([(("src", "source"), (1, 13), (1, 14))], history.verify(self.records, counted))

    def test_a_file_without_a_trailing_newline_counts_its_last_line(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Repo(directory)
            repo.path.mkdir(exist_ok=True)
            repo.git("init", "-q", "-b", "dev")
            repo.write("src/a.cpp", "one\ntwo")
            repo.commit("c", "2026-01-05T10:00:00+00:00")
            self.assertEqual((1, 2), history.count_tree(directory, "dev")[("src", "source")])
            self.assertEqual((1, 2), history.read_history(directory, "dev")[-1].state[("src", "source")])

    def test_a_submodule_outside_external_is_a_pointer_and_counts_nowhere(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Repo(directory)
            repo.path.mkdir(exist_ok=True)
            repo.git("init", "-q", "-b", "dev")
            repo.write("src/a.cpp", lines("x", 4))
            repo.git("add", "-A")
            repo.git("update-index", "--add", "--cacheinfo", f"160000,{'a' * 40},vendor/sub")
            repo.git("commit", "-q", "-m", "add a pointer", when="2026-01-05T10:00:00+00:00")
            records = history.read_history(directory, "dev")
            self.assertEqual({("src", "source"): (1, 4)}, records[-1].state)
            self.assertNotIn(("other", "other"), records[-1].splits)
            self.assertEqual([], history.verify(records, history.count_tree(directory, "dev")))

    def test_ref_resolution_falls_back_and_fails_loudly(self):
        self.assertEqual("dev", history.resolve_ref(self.repo.path, "dev"))
        self.assertEqual("dev", history.resolve_ref(self.repo.path))   # origin/dev does not exist here, dev does
        with self.assertRaises(history.HistoryError):
            history.resolve_ref(self.repo.path, "nope")


class OutputTests(FixtureCase):
    def test_the_change_row_splits_groups_and_cpp_scopes(self):
        row = output.change_row(self.record(pr=1), None)
        self.assertEqual((116, 1, 115), (row["lines_added"], row["lines_deleted"], row["lines_net"]))
        self.assertEqual((3, 0, 1), (row["files_added"], row["files_deleted"], row["files_modified"]))
        self.assertEqual((11, 1), (row["production_added"], row["production_deleted"]))
        self.assertEqual(105, row["tests_added"])                     # unit tests (5) + integration reference data (100)
        self.assertEqual((7, 0, 4, 1), (row["prod_header_like_added"], row["prod_header_like_deleted"],
                                        row["prod_source_added"], row["prod_source_deleted"]))
        self.assertEqual(5, row["tests_source_added"])                 # C++ only: the data file is not counted
        self.assertEqual(("pr", 1, "2026-W02"), (row["kind"], row["pr"], row["week"]))
        self.assertEqual("", row["ci_runs"])
    def test_ci_columns_for_a_pr(self):
        row = output.change_row(self.record(pr=2), {2: {"runs": 9, "attempts": 12, "failed": 2}})
        self.assertEqual((9, 12, 2), (row["ci_runs"], row["ci_attempts"], row["ci_failed_runs"]))
        row = output.change_row(self.record(pr=1), {2: {}})
        self.assertEqual((0, 0, 0), (row["ci_runs"], row["ci_attempts"], row["ci_failed_runs"]))

    def test_the_metrics_row_after_the_last_change(self):
        row = output.metrics_row(self.records[-1])
        # production = src + include + apps: header a.hpp (5) + b.tpp (7); source a.cpp (13) + apps/main.cpp (8)
        self.assertEqual((2, 12, 2, 21), (row["prod_header_like_files"], row["prod_header_like_lines"],
                                          row["prod_source_files"], row["prod_source_lines"]))
        self.assertEqual((1, 5), (row["tests_source_files"], row["tests_source_lines"]))
        self.assertEqual((1, 3), (row["perf_source_files"], row["perf_source_lines"]))
        self.assertEqual((0, 0), (row["tests_header_like_files"], row["tests_header_like_lines"]))
    def test_metric_ratios(self):
        row = output.metrics_row(self.records[-1])
        self.assertEqual(round(21 / 33, 4), row["prod_source_share"])
        self.assertEqual(round(5 / 33, 4), row["tests_to_prod"])
        self.assertEqual(round(3 / 33, 4), row["perf_to_prod"])
        self.assertEqual(round(8 / 33, 4), row["tests_perf_to_prod"])

    def test_a_ratio_without_a_denominator_is_empty(self):
        self.assertEqual("", output.ratio(1, 0))
        self.assertEqual(0.0, output.metrics_row(self.records[0])["perf_to_prod"])   # the root commit has code but no perf
    def test_ci_and_integration_columns(self):
        row = output.metrics_row(self.records[-1])
        self.assertEqual((1, 8, 0, 0, 1, 8), (row["ci_workflow_files"], row["ci_workflow_lines"], row["ci_other_files"],
                                              row["ci_other_lines"], row["ci_files"], row["ci_lines"]))
        self.assertEqual((1, 100), (row["integration_files"], row["integration_lines"]))
        self.assertEqual((2, 105), (row["tests_files"], row["tests_lines"]))   # group tests: unit tests + integration data
    def test_weeks_without_changes_are_zero_rows_not_gaps(self):
        rows = output.weekly_rows(self.records, None)
        self.assertEqual(["2026-W02", "2026-W03", "2026-W04"], [r["week"] for r in rows])
        self.assertEqual(["2026-01-05", "2026-01-12", "2026-01-19"], [r["week_start"] for r in rows])
        self.assertEqual((2, 1), (rows[0]["changes"], rows[0]["prs"]))
        self.assertEqual((0, 0, 0), (rows[1]["changes"], rows[1]["production_added"], rows[1]["prod_source_net"]))
        self.assertEqual((2, 1), (rows[2]["changes"], rows[2]["prs"]))
        self.assertEqual(13, rows[0]["prod_source_net"])           # the root commit adds 10 source lines, PR 1 nets +3
        self.assertEqual(8, rows[2]["prod_source_net"])            # apps/main.cpp
        self.assertEqual(-4, rows[2]["tests_source_net"])
        self.assertEqual(("", "", ""), (rows[0]["ci_runs"], rows[0]["ci_failed_runs"], rows[0]["ci_attempts"]))
    def test_weekly_ci_runs_come_from_the_daily_rows(self):
        daily = [{"date": "2026-01-07", "workflow": "BUILD", "event": "push", "conclusion": "success", "runs": "5", "attempts": "5"},
                 {"date": "2026-01-08", "workflow": "BUILD", "event": "push", "conclusion": "failure", "runs": "2", "attempts": "3"}]
        rows = output.weekly_rows(self.records, daily, (date(2026, 1, 5), date(2026, 1, 25)))
        self.assertEqual((7, 2, 8), (rows[0]["ci_runs"], rows[0]["ci_failed_runs"], rows[0]["ci_attempts"]))
        self.assertEqual((0, 0, 0), (rows[1]["ci_runs"], rows[1]["ci_failed_runs"], rows[1]["ci_attempts"]))   # in the window: none

    def test_weeks_outside_the_ci_window_stay_unknown(self):
        daily = [{"date": "2026-01-14", "workflow": "BUILD", "event": "push", "conclusion": "success", "runs": "5", "attempts": "5"}]
        rows = output.weekly_rows(self.records, daily, (date(2026, 1, 12), date(2026, 1, 25)))
        self.assertEqual("", rows[0]["ci_runs"])                  # before the window
        self.assertEqual(5, rows[1]["ci_runs"])
        self.assertEqual(0, rows[2]["ci_runs"])
        by_default = output.weekly_rows(self.records, daily)        # the window defaults to the days that have runs
        self.assertEqual(("", 5, ""), (by_default[0]["ci_runs"], by_default[1]["ci_runs"], by_default[2]["ci_runs"]))
    def test_every_file_is_written_with_stable_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = output.write_all(directory, self.records)
            self.assertEqual({"changes", "splits", "state", "metrics", "weekly"}, set(paths))
            changes = output.read_csv(paths["changes"])
            self.assertEqual(4, len(changes))
            self.assertEqual(output.change_columns(), list(changes[0]))
            self.assertEqual(output.SPLIT_COLUMNS, list(output.read_csv(paths["splits"])[0]))
            self.assertEqual(output.STATE_COLUMNS, list(output.read_csv(paths["state"])[0]))
            self.assertEqual(len(self.records), len(output.read_csv(paths["metrics"])))
            self.assertEqual(sum(len(r.splits) for r in self.records), len(output.read_csv(paths["splits"])))
            self.assertEqual(sum(len(r.state) for r in self.records), len(output.read_csv(paths["state"])))
            self.assertEqual("2026-01-07T12:00:00Z", changes[1]["date"])

    def test_the_csv_column_names_are_part_of_the_interface(self):
        for name in ("prod_header_like_added", "prod_source_deleted", "tests_source_added", "perf_source_added",
                     "ci_runs", "lines_net", "production_added", "tests_deleted"):
            self.assertIn(name, output.change_columns())
        for name in ("prod_source_share", "tests_perf_to_prod", "ci_workflow_lines", "ci_files", "integration_lines"):
            self.assertIn(name, output.metrics_columns())
        for name in ("week_start", "prod_header_like_net", "ci_failed_runs"):
            self.assertIn(name, output.weekly_columns())


RUNS = (
    "1\tBUILD\tpull_request\tfailure\tcompleted\t2026-01-06T10:00:00Z\t1\t\tfeature1\n"
    "2\tBUILD\tpull_request\tsuccess\tcompleted\t2026-01-06T11:00:00Z\t2\t\tfeature1\n"
    "3\tBUILD\tpush\tsuccess\tcompleted\t2026-01-07T12:00:01Z\t1\t\tdev\n"
    "4\tLINT\tpull_request\tsuccess\tcompleted\t2026-01-06T10:00:00Z\t1\t99\tother-branch\n"
    "5\tLINT\tpull_request\t\tin_progress\t2026-01-19T10:00:00Z\t1\t\tfeature2\n"
)


class CiTests(unittest.TestCase):
    def test_runs_are_parsed(self):
        runs = ci.parse_runs(RUNS + "garbage\n")
        self.assertEqual(5, len(runs))
        self.assertEqual(("failure", 1, None, "feature1"), (runs[0]["conclusion"], runs[0]["attempt"], runs[0]["pr"], runs[0]["branch"]))
        self.assertEqual("none", runs[4]["conclusion"])
        self.assertEqual(99, runs[3]["pr"])

    def test_daily_rows_count_runs_and_attempts(self):
        daily, _ = ci.aggregate(ci.parse_runs(RUNS))
        build = [r for r in daily if r["workflow"] == "BUILD" and r["event"] == "pull_request"]
        self.assertEqual({"date": "2026-01-06", "workflow": "BUILD", "event": "pull_request", "conclusion": "failure", "runs": 1, "attempts": 1}, build[0])
        self.assertEqual(5, len(daily))
        self.assertEqual(2, build[1]["attempts"])   # the successful run is the second attempt
    def test_runs_are_attributed_to_prs_by_branch_unless_the_api_says_otherwise(self):
        branches = {"feature1": [("2026-01-07T12:00:00Z", 1)], "feature2": [("2026-01-20T09:00:00Z", 2)]}
        _, per_pr = ci.aggregate(ci.parse_runs(RUNS), branches)
        self.assertEqual({"runs": 2, "attempts": 3, "failed": 1}, per_pr[1])
        self.assertEqual({"runs": 1, "attempts": 1, "failed": 0}, per_pr[2])
        self.assertEqual({"runs": 1, "attempts": 1, "failed": 0}, per_pr[99])   # the number from the API wins
        self.assertEqual({1, 2, 99}, set(per_pr))                              # push runs belong to no PR

    def test_a_push_to_a_branch_named_like_a_pr_branch_is_not_a_pr_run(self):
        run = {"id": "9", "workflow": "BUILD", "event": "push", "conclusion": "success", "status": "completed",
               "created": "2026-01-06T10:00:00Z", "attempt": 1, "pr": None, "branch": "feature1"}
        daily, per_pr = ci.aggregate([run], {"feature1": [("2026-01-07T12:00:00Z", 1)]})
        self.assertEqual({}, per_pr)
        self.assertEqual(1, len(daily))

    def test_a_reused_branch_name_goes_to_the_next_merge_after_the_run(self):
        branches = {"feature1": [("2026-01-07T12:00:00Z", 1), ("2026-02-01T12:00:00Z", 8)]}
        run = {"id": "9", "workflow": "BUILD", "event": "pull_request", "conclusion": "success", "status": "completed",
               "created": "2026-01-20T10:00:00Z", "attempt": 1, "pr": None, "branch": "feature1"}
        _, per_pr = ci.aggregate([run], branches)
        self.assertEqual({8}, set(per_pr))

    def test_each_day_is_queried_once_and_duplicates_collapse(self):
        queries = []

        def api(path):
            queries.append(path)
            return RUNS.splitlines()[0] + "\n"

        runs = ci.fetch_runs("o/r", date(2026, 1, 6), date(2026, 1, 8), api=api, log=lambda *_: None)
        self.assertEqual(3, len(queries))
        self.assertIn("created=2026-01-07..2026-01-07", queries[1])
        self.assertEqual(1, len(runs))

    def test_the_window_ends_today_and_has_the_requested_length(self):
        first, last = ci.window(3, today=date(2026, 1, 10))
        self.assertEqual((date(2026, 1, 8), date(2026, 1, 10)), (first, last))


class ReportTests(FixtureCase):
    def rows(self, ci_daily=None):
        with tempfile.TemporaryDirectory() as directory:
            paths = output.write_all(directory, self.records, ci_daily)
            return {k: output.read_csv(paths[k]) for k in ("metrics", "weekly", "changes")}

    def page(self, ci_daily=None, mutate=None):
        rows = self.rows(ci_daily)
        if mutate:
            mutate(rows)
        return report.render(rows["metrics"], rows["weekly"], rows["changes"], ci_daily, {"ref": "dev", "head": "abc"})

    def test_the_page_has_every_chart_and_the_latest_changes(self):
        page = self.page()
        self.assertEqual(11, page.count('<div class="panel">'))
        for title in ("Production C++ lines", "Source share", "Test and perf C++ lines per production line", "CI files", "Net C++ lines per week"):
            self.assertIn(title, page)
        self.assertIn("#2", page)
        self.assertEqual(1, page.count("<table>"))
        self.assertNotIn("CI runs per week", page)
        self.assertEqual([], re.findall(r'(?:src|href)="http', page))
    def test_the_ci_chart_appears_with_ci_data(self):
        daily = [{"date": "2026-01-07", "workflow": "BUILD", "event": "push", "conclusion": "success", "runs": "5", "attempts": "5"}]
        self.assertIn("CI runs per week", self.page(daily))

    def test_titles_from_git_are_escaped(self):
        def mutate(rows):
            rows["changes"][-1]["title"] = "<script>alert(1)</script>"

        page = self.page(mutate=mutate)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)
        self.assertNotIn("<script>alert(1)", page)

    def test_charts_survive_missing_values_single_points_and_no_data(self):
        self.assertIn("no data", report.time_chart("t", [("a", [])]))
        self.assertIn("no data", report.time_chart("t", [("a", [(1.0, None)])]))
        single = report.time_chart("t", [("a", [(1.0, 5.0)])])
        self.assertEqual(0, single.count("<polyline") - 1)   # one series, one polyline
        self.assertIn("&lt;b&gt;", report.time_chart("t", [("<b>", [(1.0, 1.0), (2.0, 2.0)])]))

    def test_a_step_chart_holds_the_value_until_the_next_change(self):
        chart = report.time_chart("t", [("a", [(0.0, 1.0), (10.0, 3.0)])], step=True)
        points = re.search(r'points="([^"]+)"', chart).group(1).split()
        self.assertEqual(3, len(points))                                    # start, the held value at the next time, the new value
        self.assertEqual(points[1].split(",")[0], points[2].split(",")[0])   # a vertical step at the second time
        self.assertEqual(points[0].split(",")[1], points[1].split(",")[1])   # the value is held until then
        sloped = re.search(r'points="([^"]+)"', report.time_chart("t", [("a", [(0.0, 1.0), (10.0, 3.0)])])).group(1).split()
        self.assertEqual(2, len(sloped))
    def test_nice_ticks_cover_the_range(self):
        ticks = report.nice_ticks(-2500, 10000)
        self.assertLessEqual(ticks[0], -2500)
        self.assertGreaterEqual(ticks[-1], 10000)
        self.assertEqual(("2.5k", "12k", "1.5M", "7", "0.5"),
                         tuple(report.format_value(v) for v in (2500, 12345, 1.5e6, 7, 0.5)))


class CliTests(FixtureCase):
    def setUp(self):
        self._out = tempfile.TemporaryDirectory()
        self.out = Path(self._out.name)

    def tearDown(self):
        self._out.cleanup()

    def run_cli(self, *args):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = pqstats.main(["--out", str(self.out), *args])
        return code, buffer.getvalue()

    def collect(self, *args):
        return self.run_cli("collect", "--repo-root", str(self.repo.path), "--ref", "dev", *args)

    def test_collect_writes_everything_and_verifies(self):
        code, text = self.collect()
        self.assertEqual(0, code)
        self.assertIn("verified", text)
        for name in ("changes.csv", "splits.csv", "state.csv", "metrics.csv", "weekly.csv", "report.html", "meta.json"):
            self.assertTrue((self.out / name).exists(), name)
        meta = json.loads((self.out / "meta.json").read_text())
        self.assertEqual((4, 4, True, "dev"), (meta["changes_total"], meta["changes_written"], meta["verified"], meta["ref"]))

    def test_since_keeps_the_totals_of_everything_before(self):
        self.collect("--since", "2026-01-20")
        metrics = output.read_csv(self.out / "metrics.csv")
        self.assertEqual(2, len(metrics))
        self.assertEqual("21", metrics[0]["prod_source_lines"])   # includes what happened before the cut-off
        self.assertEqual(2, len(output.read_csv(self.out / "changes.csv")))
    def test_excluded_prs_leave_the_flows_but_not_the_totals(self):
        self.collect("--exclude-pr", "1")
        self.assertEqual([2], [int(r["pr"]) for r in output.read_csv(self.out / "changes.csv") if r["pr"]])
        self.assertEqual("21", output.read_csv(self.out / "metrics.csv")[-1]["prod_source_lines"])
    def test_a_failed_verification_writes_nothing(self):
        with mock.patch.object(history, "verify", return_value=[(("src", "source"), (1, 1), (1, 2))]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            self.collect()
        self.assertIn("not writing anything", str(caught.exception))
        self.assertEqual([], list(self.out.iterdir()))

    def test_no_verify_skips_the_check(self):
        with mock.patch.object(history, "verify", side_effect=AssertionError("must not verify")):
            self.assertEqual(0, self.collect("--no-verify")[0])
        self.assertIsNone(json.loads((self.out / "meta.json").read_text())["verified"])

    def test_ci_failures_do_not_stop_the_collection(self):
        with mock.patch.object(ci, "fetch_runs", side_effect=ci.CiError("no gh")), contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(0, self.collect("--ci")[0])
        self.assertIn("no CI runs", errors.getvalue())
        self.assertFalse((self.out / "ci_daily.csv").exists())

    def test_ci_runs_are_joined_to_prs(self):
        runs = ci.parse_runs(RUNS)
        with mock.patch.object(ci, "fetch_runs", return_value=runs):
            self.collect("--ci", "--ci-days", "10")
        changes = {r["pr"]: r for r in output.read_csv(self.out / "changes.csv") if r["pr"]}
        self.assertEqual(("2", "3", "1"), (changes["1"]["ci_runs"], changes["1"]["ci_attempts"], changes["1"]["ci_failed_runs"]))
        self.assertTrue((self.out / "ci_daily.csv").exists())
        self.assertIn("CI runs per week", (self.out / "report.html").read_text())

    def test_show_and_report_need_data_first(self):
        for command in ("show", "report"):
            with self.assertRaises(SystemExit):
                self.run_cli(command)
        self.collect("--no-report")
        self.assertFalse((self.out / "report.html").exists())
        self.assertIn("report.html", self.run_cli("report")[1])
        table = self.run_cli("show", "--last", "2")[1]
        self.assertIn("#2", table)
        self.assertIn("Update docs", table)
        self.assertNotIn("#1 ", table)

    def test_the_output_directory_precedence(self):
        self.assertEqual(Path("/cli"), pqstats.data_dir("/cli", {pqstats.ENV_VAR: "/env"}))
        self.assertEqual(Path("/env"), pqstats.data_dir(None, {pqstats.ENV_VAR: "/env"}))
        self.assertEqual(Path("/xdg/pq-code-stats"), pqstats.data_dir(None, {"XDG_DATA_HOME": "/xdg"}))


class ToNumberTests(unittest.TestCase):
    def test_valid_invalid_and_missing_values(self):
        self.assertEqual(42.0, report.to_number("42"))
        self.assertIsNone(report.to_number("not-a-number"))
        self.assertIsNone(report.to_number(None))


if __name__ == "__main__":
    unittest.main()
