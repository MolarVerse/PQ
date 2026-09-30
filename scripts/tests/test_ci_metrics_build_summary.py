import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

METRICS = Path(__file__).resolve().parents[2] / ".github" / "ci-metrics"
SPEC = importlib.util.spec_from_file_location("ci_metrics_build_summary", METRICS / "summarise_build.py")
summary_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary_module
SPEC.loader.exec_module(summary_module)

HEADER = "# ninja log v5\n"


def entry(start, end, output, command_hash="abc", mtime="1"):
    return f"{start}\t{end}\t{mtime}\t{output}\t{command_hash}\n"


# 1 s of setup, two compiles in parallel, then an archive and a 6 s link.
SAMPLE = HEADER + "".join(
    [
        entry(0, 1000, "src/CMakeFiles/pq_pch.dir/cmake_pch.hxx.gch", "h0"),
        entry(1000, 4000, "src/CMakeFiles/a.dir/a.cpp.o", "h1"),
        entry(1000, 9000, "src/CMakeFiles/b.dir/b.cpp.o", "h2"),
        entry(9000, 9500, "src/liba.a", "h3"),
        entry(9500, 15500, "apps/PQ", "h4"),
    ]
)

CCACHE_STATS = (
    "stats_updated_timestamp\t1790000000\n"
    "stats_zeroed_timestamp\t1789999000\n"
    "direct_cache_hit\t129\n"
    "preprocessed_cache_hit\t1\n"
    "cache_miss\t13\n"
    "could_not_use_precompiled_header\t269\n"
    "compile_failed\t0\n"
    "cache_size_kibibyte\t318000\n"
)


class ClassifyTests(unittest.TestCase):
    def test_kinds(self):
        c = summary_module.classify
        self.assertEqual("compile", c("src/CMakeFiles/x.dir/x.cpp.o"))
        self.assertEqual("compile", c("src/CMakeFiles/pq_pch.dir/cmake_pch.hxx.gch"))
        self.assertEqual("archive", c("src/libpq.a"))
        self.assertEqual("link", c("src/libpq.so"))
        self.assertEqual("link", c("src/libpq.so.1.2"))
        self.assertEqual("link", c("apps/PQ"))
        self.assertEqual("link", c("tests/src/testFoo"))
        self.assertEqual("other", c("docs/index.html"))
        self.assertEqual("other", c("generated/config.hpp"))


class ParseNinjaLogTests(unittest.TestCase):
    def test_reads_version_and_steps_in_start_order(self):
        steps, version, ignored = summary_module.parse_ninja_log(SAMPLE)
        self.assertEqual(5, version)
        self.assertEqual(0, ignored)
        self.assertEqual(
            ["cmake_pch.hxx.gch", "a.cpp.o", "b.cpp.o", "liba.a", "PQ"],
            [Path(step["target"]).name for step in steps],
        )

    def test_a_rebuilt_output_keeps_only_its_last_entry(self):
        text = HEADER + entry(0, 5000, "x.o", "old") + entry(100, 200, "y.o", "y") + entry(300, 900, "x.o", "new")
        steps, _, _ = summary_module.parse_ninja_log(text)
        self.assertEqual({"y.o": 100, "x.o": 600}, {s["target"]: s["end"] - s["start"] for s in steps})

    def test_outputs_of_one_command_count_as_one_step(self):
        text = HEADER + entry(0, 500, "gen.hpp", "same") + entry(0, 500, "gen.cpp", "same")
        steps, _, _ = summary_module.parse_ninja_log(text)
        self.assertEqual(1, len(steps))

    def test_malformed_lines_are_counted_and_skipped(self):
        text = HEADER + "garbage\n" + "1\t2\n" + entry(500, 100, "backwards.o") + entry(0, 10, "ok.o")
        steps, _, ignored = summary_module.parse_ninja_log(text)
        self.assertEqual(["ok.o"], [s["target"] for s in steps])
        self.assertEqual(3, ignored)

    def test_missing_header_gives_no_version(self):
        _, version, _ = summary_module.parse_ninja_log(entry(0, 10, "ok.o"))
        self.assertIsNone(version)


class SummariseNinjaTests(unittest.TestCase):
    def test_totals_and_split(self):
        s = summary_module.summarise_ninja(SAMPLE, pending_steps=0)
        self.assertEqual(15.5, s["wall_s"])
        self.assertEqual(1 + 3 + 8 + 0.5 + 6, s["cpu_s"])
        self.assertEqual(round(18.5 / 15.5, 2), s["parallelism"])
        self.assertTrue(s["complete"])
        self.assertEqual(5, s["steps"])
        self.assertEqual({"steps": 3, "cpu_s": 12.0}, s["by_kind"]["compile"])
        self.assertEqual({"steps": 1, "cpu_s": 0.5}, s["by_kind"]["archive"])
        self.assertEqual({"steps": 1, "cpu_s": 6.0}, s["by_kind"]["link"])
        self.assertEqual({"steps": 0, "cpu_s": 0.0}, s["by_kind"]["other"])

    def test_time_after_the_last_compile_is_the_link_tail(self):
        self.assertEqual(6.5, summary_module.summarise_ninja(SAMPLE, 0)["tail_after_compile_s"])

    def test_slowest_are_sorted_and_capped(self):
        many = HEADER + "".join(entry(0, 1000 + i, f"t{i:03d}.o", f"h{i}") for i in range(30))
        slowest = summary_module.summarise_ninja(many, 0)["slowest"]
        self.assertEqual(summary_module.SLOWEST, len(slowest))
        self.assertEqual("t029.o", slowest[0]["target"])
        self.assertEqual("compile", slowest[0]["kind"])
        top = summary_module.summarise_ninja(SAMPLE, 0)["slowest"][:2]
        self.assertEqual(["src/CMakeFiles/b.dir/b.cpp.o", "apps/PQ"], [x["target"] for x in top])
        self.assertEqual([8.0, 6.0], [x["seconds"] for x in top])

    def test_completeness_follows_the_pending_step_count(self):
        self.assertTrue(summary_module.summarise_ninja(SAMPLE, 0)["complete"])
        partial = summary_module.summarise_ninja(SAMPLE, 12)
        self.assertFalse(partial["complete"])
        self.assertEqual(12, partial["pending_steps"])
        self.assertIsNone(summary_module.summarise_ninja(SAMPLE, None)["complete"])

    def test_log_versions_5_to_7_are_read(self):
        for version in (5, 6, 7):
            text = SAMPLE.replace("v5", f"v{version}", 1)
            self.assertEqual(5, summary_module.summarise_ninja(text, 0)["steps"], version)

    def test_unusable_logs_say_why_instead_of_raising(self):
        for text in ("", "# ninja log v4\n" + entry(0, 1, "a.o"), "# ninja log v8\n" + entry(0, 1, "a.o"), HEADER):
            s = summary_module.summarise_ninja(text, 0)
            self.assertIsNone(s["complete"])
            self.assertIn("error", s)

    def test_a_no_op_build_has_no_parallelism(self):
        s = summary_module.summarise_ninja(HEADER + entry(5, 5, "a.o"), 0)
        self.assertIsNone(s["parallelism"])
        self.assertEqual(0.0, s["wall_s"])


class CcacheTests(unittest.TestCase):
    def test_parses_counters_and_the_hit_rate_of_cacheable_calls(self):
        c = summary_module.parse_ccache_stats(CCACHE_STATS)
        self.assertEqual(130, c["hits"])
        self.assertEqual(13, c["misses"])
        self.assertEqual(round(130 / 143, 4), c["hit_rate"])
        self.assertEqual(269, c["counters"]["could_not_use_precompiled_header"])

    def test_timestamps_and_zero_counters_are_dropped(self):
        c = summary_module.parse_ccache_stats(CCACHE_STATS)
        self.assertNotIn("stats_updated_timestamp", c["counters"])
        self.assertNotIn("compile_failed", c["counters"])

    def test_no_cacheable_calls_gives_no_rate(self):
        c = summary_module.parse_ccache_stats("direct_cache_hit\t0\ncache_miss\t0\nunsupported_compiler_option\t4\n")
        self.assertIsNone(c["hit_rate"])
        self.assertEqual({"unsupported_compiler_option": 4}, c["counters"])

    def test_unrecognised_output_gives_none(self):
        self.assertIsNone(summary_module.parse_ccache_stats(""))
        self.assertIsNone(summary_module.parse_ccache_stats("ccache: command not found\n"))


class DryRunTests(unittest.TestCase):
    def dry_run(self, stdout="", stderr="", raises=None, returncode=0):
        with mock.patch.object(summary_module.subprocess, "run") as run:
            if raises:
                run.side_effect = raises
            else:
                run.return_value = subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)
            result = summary_module.ninja_dry_run("build")
            self.command = run.call_args[0][0] if run.call_args else None
            return result

    def test_no_work_to_do_means_complete(self):
        self.assertEqual(0, self.dry_run("ninja: no work to do.\n")[0])
        self.assertEqual(["ninja", "-C", "build", "-n", "-d", "explain"], self.command)

    def test_counts_the_steps_ninja_would_still_run_and_keeps_the_explanation(self):
        pending, explanation = self.dry_run(
            "[1/3] Building CXX a.o\n[2/3] Building CXX b.o\n[3/3] Linking x\n",
            stderr="ninja explain: output x older than most recent input a.o\n",
        )
        self.assertEqual(3, pending)
        self.assertIn("ninja explain: output x older than most recent input a.o", explanation)
        self.assertIn("[3/3] Linking x", explanation)

    def test_the_explanation_is_capped(self):
        _, explanation = self.dry_run("[1/1] x\n", stderr="explain\n" * 1000)
        self.assertEqual(summary_module.EXPLAIN_LINES, len(explanation.splitlines()))

    def test_unknown_output_a_failing_ninja_or_a_missing_ninja_is_unknown(self):
        self.assertIsNone(self.dry_run("something else\n")[0])
        self.assertEqual((None, None), self.dry_run("", returncode=1))
        self.assertEqual((None, None), self.dry_run(raises=FileNotFoundError()))


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.build = self.root / "build"
        self.out = self.root / "out"
        self.env = {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2", "GITHUB_JOB": "lint"}

    def run_main(self, *extra, tools=None):
        """`tools` maps the program name to stdout (None = fails)."""
        tools = tools or {}

        def fake_run(command, **_):
            if command[0] in tools and tools[command[0]] is not None:
                return subprocess.CompletedProcess(command, 0, stdout=tools[command[0]], stderr="explained\n")
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

        out = io.StringIO()
        with mock.patch.object(summary_module.subprocess, "run", side_effect=fake_run), contextlib.redirect_stdout(out):
            code = summary_module.main(
                ["--out", str(self.out), "--name", "build-timings-lint-a2", "--build-dir", str(self.build), *extra],
                env=self.env,
            )
        return code, json.loads((self.out / "build-analysis.json").read_text()), out.getvalue()

    def test_writes_summary_and_raw_files(self):
        self.build.mkdir()
        (self.build / ".ninja_log").write_text(SAMPLE)
        code, data, printed = self.run_main(
            "--ccache", "--job-id", "555", tools={"ninja": "ninja: no work to do.\n", "ccache": CCACHE_STATS}
        )
        self.assertEqual(0, code)
        self.assertEqual(
            {
                "schema_version": 1,
                "kind": "build-analysis",
                "run_id": 123,
                "run_attempt": 2,
                "job_id": 555,
                "job_key": "lint",
                "artifact": "build-timings-lint-a2",
            },
            {k: data[k] for k in ("schema_version", "kind", "run_id", "run_attempt", "job_id", "job_key", "artifact")},
        )
        self.assertTrue(data["ninja"]["complete"])
        self.assertEqual(13, data["ccache"]["misses"])
        self.assertEqual(SAMPLE, (self.out / "ninja_log.txt").read_text())
        self.assertEqual(CCACHE_STATS, (self.out / "ccache-stats.txt").read_text())
        self.assertIn("ninja steps=5 complete=True", printed)

    def test_a_partial_build_is_flagged(self):
        self.build.mkdir()
        (self.build / ".ninja_log").write_text(SAMPLE)
        _, data, _ = self.run_main(tools={"ninja": "[1/2] Building CXX broken.o\n[2/2] Linking x\n"})
        self.assertFalse(data["ninja"]["complete"])
        self.assertEqual(2, data["ninja"]["pending_steps"])
        self.assertIn("explained", (self.out / "ninja-pending.txt").read_text())

    def test_a_complete_build_writes_no_explanation(self):
        self.build.mkdir()
        (self.build / ".ninja_log").write_text(SAMPLE)
        self.run_main(tools={"ninja": "ninja: no work to do.\n"})
        self.assertFalse((self.out / "ninja-pending.txt").exists())

    def test_without_any_input_it_still_succeeds_and_records_nulls(self):
        code, data, _ = self.run_main()
        self.assertEqual(0, code)
        self.assertIsNone(data["ninja"])
        self.assertIsNone(data["ccache"])
        self.assertIsNone(data["job_id"])
        self.assertFalse((self.out / "ninja_log.txt").exists())

    def test_ccache_is_only_queried_when_asked_for(self):
        _, data, _ = self.run_main(tools={"ccache": CCACHE_STATS})
        self.assertIsNone(data["ccache"])

    def test_a_failing_ccache_leaves_the_rest_intact(self):
        self.build.mkdir()
        (self.build / ".ninja_log").write_text(SAMPLE)
        _, data, _ = self.run_main("--ccache", tools={"ninja": "ninja: no work to do.\n", "ccache": None})
        self.assertIsNone(data["ccache"])
        self.assertEqual(5, data["ninja"]["steps"])

    def test_a_binary_log_does_not_crash(self):
        self.build.mkdir()
        (self.build / ".ninja_log").write_bytes(b"\xff\xfe\x00garbage")
        code, data, _ = self.run_main()
        self.assertEqual(0, code)
        self.assertIn("error", data["ninja"])


if __name__ == "__main__":
    unittest.main()
