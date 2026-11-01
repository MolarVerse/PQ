"""Tests of scripts/perf_gate.sh with a fake valgrind (no callgrind needed).

A fake benchmark is a shell script; its instruction count is read from a
sibling `<name>.ir` file, and it fails when a `<name>.fail` file exists.
"""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "perf_gate.sh"

FAKE_VALGRIND = """#!/bin/bash
bin="${@: -1}"
if [ -e "$bin.fail" ]; then exit 3; fi
if [ -e "$bin.ir" ]; then echo "==1== I   refs:      $(cat "$bin.ir")" >&2; fi
exit 0
"""


class PerfGateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.pr, self.base, self.bin = self.root / "pr", self.root / "base", self.root / "bin"
        for d in (self.pr, self.base, self.bin):
            d.mkdir()
        valgrind = self.bin / "valgrind"
        valgrind.write_text(FAKE_VALGRIND)
        valgrind.chmod(valgrind.stat().st_mode | stat.S_IEXEC)

    def bench(self, directory, name, ir=None, fail=False, allocs=None):
        path = directory / f"perf_{name}"
        path.write_text('#!/bin/bash\n[ -e "$0.allocs" ] && echo "PQ_PERF_ALLOCS $(cat "$0.allocs") 0" >&2\nexit 0\n')
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        if ir is not None:
            Path(f"{path}.ir").write_text(str(ir))
        if fail:
            Path(f"{path}.fail").write_text("")
        if allocs is not None:
            Path(f"{path}.allocs").write_text(str(allocs))

    def with_alloc_library(self):
        (self.pr / "libperf_alloc_counter.so").write_text("")

    def run_gate(self, base_dir=None, env_extra=None):
        report = self.root / "report.md"
        env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", PERF_REPORT_FILE=str(report))
        env.pop("GITHUB_STEP_SUMMARY", None)
        env.update(env_extra or {})
        result = subprocess.run(["bash", str(SCRIPT), str(self.pr), str(base_dir or self.base), "2"],
                                capture_output=True, text=True, env=env)
        return result, report.read_text()

    def test_unchanged_passes(self):
        self.bench(self.pr, "a", 1_000_000)
        self.bench(self.base, "a", 1_000_000)
        result, report = self.run_gate()
        self.assertEqual(0, result.returncode)
        self.assertIn("no regressions", report)

    def test_regression_fails(self):
        self.bench(self.pr, "a", 1_050_000)
        self.bench(self.base, "a", 1_000_000)
        result, report = self.run_gate()
        self.assertEqual(1, result.returncode)
        self.assertIn("regression", report)

    def test_improvement_passes_and_is_reported(self):
        self.bench(self.pr, "a", 900_000)
        self.bench(self.base, "a", 1_000_000)
        result, report = self.run_gate()
        self.assertEqual(0, result.returncode)
        self.assertIn("improvement", report)

    def test_failing_pr_benchmark_is_not_an_improvement(self):
        self.bench(self.pr, "a", 1_000_000, fail=True)
        self.bench(self.base, "a", 1_000_000)
        result, report = self.run_gate()
        self.assertEqual(1, result.returncode)
        self.assertIn("no instruction count", report)
        self.assertNotIn("improvement", report)

    def test_pr_benchmark_without_a_count_fails(self):
        self.bench(self.pr, "a")
        self.bench(self.base, "a", 1_000_000)
        result, report = self.run_gate()
        self.assertEqual(1, result.returncode)
        self.assertIn("no instruction count", report)

    def test_failing_base_benchmark_is_visible_but_does_not_fail(self):
        self.bench(self.pr, "a", 1_000_000)
        self.bench(self.base, "a", 1_000_000, fail=True)
        result, report = self.run_gate()
        self.assertEqual(0, result.returncode)
        self.assertIn("base: no count", report)

    def test_new_benchmark_passes(self):
        self.bench(self.pr, "a", 1_000_000)
        result, report = self.run_gate()
        self.assertEqual(0, result.returncode)
        self.assertIn("new", report)

    def test_removed_benchmark_is_listed(self):
        self.bench(self.pr, "a", 1_000_000)
        self.bench(self.base, "a", 1_000_000)
        self.bench(self.base, "gone", 2_000_000)
        result, report = self.run_gate()
        self.assertEqual(0, result.returncode)
        self.assertIn("`gone`", report)
        self.assertIn("removed", report)

    def test_removed_benchmark_is_listed_when_only_the_cache_knows_it(self):
        self.bench(self.pr, "a", 1_000_000)
        cache = self.root / "base_ir.tsv"
        cache.write_text("a\t1000000\ngone\t2000000\n")
        result, report = self.run_gate(base_dir=self.root / "no-such-dir", env_extra={"PERF_BASE_IR_FILE": str(cache)})
        self.assertEqual(0, result.returncode)
        self.assertIn("`gone`", report)
        self.assertIn("removed", report)


    def test_size_series_reports_the_scaling_exponent(self):
        for directory, large in ((self.pr, 4_000_000), (self.base, 2_000_000)):
            self.bench(directory, "seriesN100", 1_000_000)
            self.bench(directory, "seriesN200", large)
        result, report = self.run_gate()
        self.assertIn("Scaling", report)
        self.assertIn("| `series` | 100 to 200 | 1.00 | 2.00 |", report)
        self.assertEqual(1, result.returncode)   # the 200-molecule benchmark doubled

    def test_scaling_uses_the_smallest_and_the_largest_size(self):
        for directory in (self.pr, self.base):
            for size, ir in ((100, 1_000_000), (200, 2_000_000), (400, 4_000_000)):
                self.bench(directory, f"seriesN{size}", ir)
        result, report = self.run_gate()
        self.assertEqual(0, result.returncode)
        self.assertIn("| `series` | 100 to 400 | 1.00 | 1.00 |", report)

    def test_no_scaling_table_without_a_series(self):
        self.bench(self.pr, "a", 1_000_000)
        self.bench(self.base, "a", 1_000_000)
        _, report = self.run_gate()
        self.assertNotIn("Scaling", report)

    def test_series_without_base_shows_a_dash(self):
        self.bench(self.pr, "seriesN100", 1_000_000)
        self.bench(self.pr, "seriesN200", 2_000_000)
        result, report = self.run_gate()
        self.assertEqual(0, result.returncode)
        self.assertIn("| `series` | 100 to 200 | – | 1.00 |", report)


    def test_a_large_drop_is_flagged_but_does_not_fail(self):
        self.bench(self.pr, "a", 600_000)
        self.bench(self.base, "a", 1_000_000)
        result, report = self.run_gate()
        self.assertEqual(0, result.returncode)
        self.assertIn("large drop", report)
        self.assertIn("still does its work", report)

    def test_a_moderate_improvement_is_not_flagged(self):
        self.bench(self.pr, "a", 800_000)
        self.bench(self.base, "a", 1_000_000)
        _, report = self.run_gate()
        self.assertIn("improvement", report)
        self.assertNotIn("large drop", report)

    def test_the_large_drop_limit_can_be_changed(self):
        self.bench(self.pr, "a", 800_000)
        self.bench(self.base, "a", 1_000_000)
        _, report = self.run_gate(env_extra={"PERF_LARGE_DROP_PERCENT": "10"})
        self.assertIn("large drop", report)

    def test_a_regression_next_to_a_large_drop_still_fails(self):
        self.bench(self.pr, "a", 500_000)
        self.bench(self.base, "a", 1_000_000)
        self.bench(self.pr, "b", 1_100_000)
        self.bench(self.base, "b", 1_000_000)
        result, report = self.run_gate()
        self.assertEqual(1, result.returncode)
        self.assertIn("regression", report)


    def alloc_case(self, base_allocs, pr_allocs, ir=1_000_000):
        self.with_alloc_library()
        self.bench(self.pr, "a", ir, allocs=pr_allocs)
        self.bench(self.base, "a", ir, allocs=base_allocs)
        return self.run_gate()

    def test_unchanged_allocations_are_shown_and_pass(self):
        result, report = self.alloc_case(10, 10)
        self.assertEqual(0, result.returncode)
        self.assertIn("Heap allocations in the measured loop", report)
        self.assertIn("| `a` | 10 | 10 | +0.00% | ✅ |", report)

    def test_more_allocations_fail(self):
        result, report = self.alloc_case(10, 20)
        self.assertEqual(1, result.returncode)
        self.assertIn("more heap allocations", report)
        self.assertIn("| `a` | 10 | 20 | +100.00% | ❌ |", report)

    def test_the_first_allocation_of_an_allocation_free_loop_fails(self):
        result, report = self.alloc_case(0, 1)
        self.assertEqual(1, result.returncode)
        self.assertIn("| `a` | 0 | 1 | +1 | ❌ |", report)

    def test_fewer_allocations_pass(self):
        result, report = self.alloc_case(20, 10)
        self.assertEqual(0, result.returncode)
        self.assertIn("| `a` | 20 | 10 | -50.00% | 🎉 |", report)

    def test_a_small_rise_within_the_threshold_passes(self):
        result, _ = self.alloc_case(1000, 1010)
        self.assertEqual(0, result.returncode)

    def test_no_allocation_library_means_no_allocation_table(self):
        self.bench(self.pr, "a", 1_000_000, allocs=10)
        self.bench(self.base, "a", 1_000_000, allocs=10)
        result, report = self.run_gate()
        self.assertEqual(0, result.returncode)
        self.assertNotIn("Heap allocations", report)

    def test_a_base_binary_without_the_marker_is_not_compared(self):
        result, report = self.alloc_case(None, 12)
        self.assertEqual(0, result.returncode)
        self.assertIn("| `a` | – | 12 | – | |", report)

    def test_a_new_benchmark_shows_its_allocations(self):
        self.with_alloc_library()
        self.bench(self.pr, "a", 1_000_000, allocs=7)
        result, report = self.run_gate()
        self.assertEqual(0, result.returncode)
        self.assertIn("| `a` | – | 7 | – | |", report)

    def test_cached_base_allocations_are_used_and_written_back(self):
        self.with_alloc_library()
        self.bench(self.pr, "a", 1_000_000, allocs=30)
        cache = self.root / "base_ir.tsv"
        cache.write_text("a\t1000000\t10\n")
        result, report = self.run_gate(
            base_dir=self.root / "no-such-dir", env_extra={"PERF_BASE_IR_FILE": str(cache)}
        )
        self.assertEqual(1, result.returncode)
        self.assertIn("| `a` | 10 | 30 | +200.00% | ❌ |", report)
        self.assertEqual("a\t1000000\t10\n", cache.read_text())

    def test_a_cache_from_before_the_allocation_column_still_works(self):
        self.with_alloc_library()
        self.bench(self.pr, "a", 1_000_000, allocs=30)
        cache = self.root / "base_ir.tsv"
        cache.write_text("a\t1000000\n")
        result, report = self.run_gate(
            base_dir=self.root / "no-such-dir", env_extra={"PERF_BASE_IR_FILE": str(cache)}
        )
        self.assertEqual(0, result.returncode)
        self.assertIn("| `a` | – | 30 | – | |", report)

    def test_the_threshold_of_the_allocation_check_can_be_changed(self):
        self.with_alloc_library()
        self.bench(self.pr, "a", 1_000_000, allocs=1010)
        self.bench(self.base, "a", 1_000_000, allocs=1000)
        result, _ = self.run_gate(env_extra={"PERF_ALLOC_THRESHOLD_PERCENT": "0.5"})
        self.assertEqual(1, result.returncode)


if __name__ == "__main__":
    unittest.main()
