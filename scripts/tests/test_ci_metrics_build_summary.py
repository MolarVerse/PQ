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
        s = summary_module.summarise_ninja(SAMPLE, build_ok=True)
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
        self.assertEqual(6.5, summary_module.summarise_ninja(SAMPLE, True)["tail_after_compile_s"])

    def test_slowest_are_sorted_and_capped(self):
        many = HEADER + "".join(entry(0, 1000 + i, f"t{i:03d}.o", f"h{i}") for i in range(30))
        slowest = summary_module.summarise_ninja(many, True)["slowest"]
        self.assertEqual(summary_module.SLOWEST, len(slowest))
        self.assertEqual("t029.o", slowest[0]["target"])
        self.assertEqual("compile", slowest[0]["kind"])
        top = summary_module.summarise_ninja(SAMPLE, True)["slowest"][:2]
        self.assertEqual(["src/CMakeFiles/b.dir/b.cpp.o", "apps/PQ"], [x["target"] for x in top])
        self.assertEqual([8.0, 6.0], [x["seconds"] for x in top])

    def test_completeness_is_what_the_job_reported(self):
        self.assertTrue(summary_module.summarise_ninja(SAMPLE, True)["complete"])
        self.assertFalse(summary_module.summarise_ninja(SAMPLE, False)["complete"])
        self.assertIsNone(summary_module.summarise_ninja(SAMPLE, None)["complete"])

    def test_log_versions_5_to_7_are_read(self):
        for version in (5, 6, 7):
            text = SAMPLE.replace("v5", f"v{version}", 1)
            self.assertEqual(5, summary_module.summarise_ninja(text, True)["steps"], version)

    def test_unusable_logs_say_why_instead_of_raising(self):
        for text in ("", "# ninja log v4\n" + entry(0, 1, "a.o"), "# ninja log v8\n" + entry(0, 1, "a.o"), HEADER):
            s = summary_module.summarise_ninja(text, True)
            self.assertIsNone(s["complete"])
            self.assertIn("error", s)

    def test_a_no_op_build_has_no_parallelism(self):
        s = summary_module.summarise_ninja(HEADER + entry(5, 5, "a.o"), True)
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


class BuildSucceededTests(unittest.TestCase):
    def test_maps_step_outcomes(self):
        f = summary_module.build_succeeded
        self.assertTrue(f("success"))
        self.assertFalse(f("failure"))
        self.assertFalse(f("cancelled"))
        self.assertIsNone(f("skipped"))
        self.assertIsNone(f(None))


DEPS = (
    "a.o: #deps 5, deps mtime 1790000000000000000 (VALID)\n"
    "    /repo/src/a.cpp\n"
    "    /repo/include/x.hpp\n"
    "    /repo/include/y.hpp\n"
    "    /usr/include/stdio.h\n"
    "    /repo/include/../include/x.hpp\n"
    "\n"
    "b.o: #deps 3, deps mtime 1790000000000000000 (VALID)\n"
    "    /repo/src/b.cpp\n"
    "    /repo/include/x.hpp\n"
    "    /repo/build/gen/config.hpp\n"
    "\n"
    "libfoo.a: #deps 1, deps mtime 1790000000000000000 (VALID)\n"
    "    /repo/include/not-counted.hpp\n"
    "\n"
    "garbage line without a target\n"
)


class IncludeGraphTests(unittest.TestCase):
    def graph(self, text=DEPS, build="/repo/build", root="/repo"):
        return summary_module.include_graph(summary_module.parse_ninja_deps(text), build, root)

    def test_parse_reads_targets_and_their_dependencies(self):
        targets = summary_module.parse_ninja_deps(DEPS)
        self.assertEqual({"a.o", "b.o", "libfoo.a"}, set(targets))
        self.assertEqual(5, len(targets["a.o"]))
        self.assertEqual(["/repo/include/not-counted.hpp"], targets["libfoo.a"])

    def test_counts_translation_units_pairs_and_project_files(self):
        summary, fan_in = self.graph()
        self.assertEqual(2, summary["objects"])
        self.assertEqual(4, summary["unique_files"])  # x, y, stdio, config; the sources are not dependencies
        self.assertEqual(5, summary["include_pairs"])
        self.assertEqual({"include/x.hpp": 2, "include/y.hpp": 1}, fan_in)
        self.assertEqual((2, 3), (summary["project_files"], summary["project_pairs"]))

    def test_the_translation_units_own_source_and_generated_or_system_files_are_not_project_headers(self):
        _, fan_in = self.graph()
        self.assertNotIn("src/a.cpp", fan_in)
        self.assertNotIn("build/gen/config.hpp", fan_in)
        self.assertNotIn("/usr/include/stdio.h", fan_in)

    def test_only_object_targets_count(self):
        self.assertNotIn("include/not-counted.hpp", self.graph()[1])

    def test_dot_dot_segments_are_normalised_and_deduplicated(self):
        # x.hpp appears twice for a.o (once through ..): one pair
        summary, fan_in = self.graph()
        self.assertEqual(2, fan_in["include/x.hpp"])

    def test_relative_dependencies_are_relative_to_the_build_directory(self):
        text = "a.o: #deps 2, deps mtime 1 (VALID)\n    ../src/a.c\n    ../include/r.hpp\n"
        self.assertEqual({"include/r.hpp": 1}, self.graph(text)[1])

    def test_top_files_are_sorted_by_fan_in_then_name_and_capped(self):
        lines = []
        for i in range(40):
            lines.append(f"o{i}.o: #deps 1, deps mtime 1 (VALID)\n    /repo/include/h{34 - i % 35:02d}.hpp\n\n")
        summary, fan_in = self.graph("".join(lines))
        top = summary["top_project_files"]
        self.assertEqual(summary_module.TOP_HEADERS, len(top))
        self.assertEqual({"file": "include/h30.hpp", "fan_in": 2}, top[0])  # the late names have the high fan-in
        self.assertEqual(35, len(fan_in))
        self.assertEqual(top, sorted(top, key=lambda e: (-e["fan_in"], e["file"])))

    def test_the_digest_does_not_depend_on_the_order_of_a_large_graph(self):
        blocks = []
        for i in range(80):
            deps = "".join(f"    /repo/include/h{(i * 7 + j * 13) % 97}.hpp\n" for j in range(6))
            blocks.append(f"t{i}.o: #deps 6, deps mtime 1 (VALID)\n{deps}")
        forward = "\n".join(blocks)
        backward = "\n".join(reversed(blocks))
        shuffled = "\n".join(blocks[40:] + blocks[:40])
        digests = {self.graph(text)[0]["digest"] for text in (forward, backward, shuffled)}
        self.assertEqual(1, len(digests))

    def test_the_digest_ignores_order_and_workspace_root_but_sees_changed_includes(self):
        reordered = "\n\n".join(reversed(DEPS.split("\n\n")))
        moved = DEPS.replace("/repo/", "/other/checkout/")
        base = self.graph()[0]["digest"]
        self.assertEqual(base, self.graph(reordered)[0]["digest"])
        self.assertEqual(base, self.graph(moved, build="/other/checkout/build", root="/other/checkout")[0]["digest"])
        changed = DEPS.replace("    /repo/include/y.hpp\n", "    /repo/include/z.hpp\n")
        self.assertNotEqual(base, self.graph(changed)[0]["digest"])

    def test_no_objects_or_failing_ninja_give_none(self):
        for output, code in (("", 0), ("libfoo.a: #deps 1, deps mtime 1 (VALID)\n    /x.h\n", 0), (DEPS, 1)):
            with mock.patch.object(summary_module.subprocess, "run") as run:
                run.return_value = subprocess.CompletedProcess([], code, stdout=output, stderr="")
                self.assertEqual((None, None), summary_module.read_includes("/repo/build", "/repo"))
        with mock.patch.object(summary_module.subprocess, "run", side_effect=FileNotFoundError()):
            self.assertEqual((None, None), summary_module.read_includes("/repo/build", "/repo"))


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
            "--ccache", "--job-id", "555", "--build-status", "success", tools={"ccache": CCACHE_STATS}
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

    def test_a_build_that_failed_is_flagged_as_partial(self):
        self.build.mkdir()
        (self.build / ".ninja_log").write_text(SAMPLE)
        _, data, _ = self.run_main("--build-status", "failure")
        self.assertFalse(data["ninja"]["complete"])

    def test_without_a_build_status_completeness_is_unknown(self):
        self.build.mkdir()
        (self.build / ".ninja_log").write_text(SAMPLE)
        _, data, _ = self.run_main()
        self.assertIsNone(data["ninja"]["complete"])
        self.assertEqual(5, data["ninja"]["steps"])

    def test_without_ccache_only_ninja_deps_is_run(self):
        self.build.mkdir()
        (self.build / ".ninja_log").write_text(SAMPLE)
        with mock.patch.object(summary_module.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 1, stdout="", stderr="")
            summary_module.main(["--out", str(self.out), "--name", "n", "--build-dir", str(self.build)], env=self.env)
        self.assertEqual([["ninja", "-C", str(self.build), "-t", "deps"]], [call.args[0] for call in run.call_args_list])

    def test_the_include_graph_goes_into_the_summary_and_a_detail_file(self):
        self.build.mkdir()
        (self.build / ".ninja_log").write_text(SAMPLE)
        deps = DEPS.replace("/repo/", f"{self.root.resolve()}/")
        _, data, printed = self.run_main("--source-root", str(self.root), tools={"ninja": deps})
        self.assertEqual(2, data["includes"]["objects"])
        self.assertEqual(3, data["includes"]["project_pairs"])
        detail = json.loads((self.out / "ninja-includes.json").read_text())
        self.assertEqual(("ninja-includes-detail", 1), (detail["kind"], detail["schema_version"]))
        self.assertEqual({"include/x.hpp": 2, "include/y.hpp": 1}, detail["fan_in"])
        self.assertEqual(data["includes"]["digest"], detail["digest"])
        self.assertIn("include pairs=5", printed)

    def test_no_ninja_log_means_no_include_graph(self):
        _, data, _ = self.run_main(tools={"ninja": DEPS})
        self.assertIsNone(data["includes"])
        self.assertFalse((self.out / "ninja-includes.json").exists())

    def test_a_failing_ccache_leaves_the_rest_intact(self):
        self.build.mkdir()
        (self.build / ".ninja_log").write_text(SAMPLE)
        _, data, _ = self.run_main("--ccache", tools={"ccache": None})
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
