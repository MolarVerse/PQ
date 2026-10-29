"""Tests of scripts/perf_wallclock.py with fake benchmarks (shell scripts)."""

import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import perf_wallclock as pw  # noqa: E402


class PerfWallclockTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.a, self.b = self.root / "a", self.root / "b"
        self.a.mkdir()
        self.b.mkdir()

    def bench(self, directory, name, seconds=0.0, exit_code=0, log=None):
        path = directory / f"perf_{name}"
        lines = ["#!/bin/bash"]
        if log:
            lines.append(f'echo "{directory.name}" >> "{log}"')
        if seconds:
            lines.append(f"sleep {seconds}")
        lines.append(f"exit {exit_code}")
        path.write_text("\n".join(lines) + "\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def test_a_slower_build_gives_a_ratio_above_one(self):
        self.bench(self.a, "x", 0.02)
        self.bench(self.b, "x", 0.06)
        result = pw.measure(self.a, self.b, "x", reps=5, pin=None, warmup=0)
        self.assertGreater(result["wall"]["median"], 2.0)
        self.assertLess(result["wall"]["median"], 4.0)

    def test_identical_builds_give_a_ratio_near_one(self):
        self.bench(self.a, "x", 0.03)
        result = pw.measure(self.a, self.a, "x", reps=7, pin=None, warmup=0)
        self.assertGreater(result["wall"]["median"], 0.8)
        self.assertLess(result["wall"]["median"], 1.25)
        self.assertLessEqual(result["wall"]["min"], result["wall"]["median"])
        self.assertGreaterEqual(result["wall"]["max"], result["wall"]["median"])

    def test_runs_are_interleaved_and_the_order_alternates(self):
        log = self.root / "order.log"
        self.bench(self.a, "x", log=log)
        self.bench(self.b, "x", log=log)
        pw.measure(self.a, self.b, "x", reps=4, pin=None, warmup=0)
        self.assertEqual(["a", "b", "b", "a", "a", "b", "b", "a"], log.read_text().split())

    def test_a_failing_benchmark_raises(self):
        self.bench(self.a, "x", exit_code=3)
        self.bench(self.b, "x")
        with self.assertRaises(RuntimeError):
            pw.measure(self.a, self.b, "x", reps=1, pin=None, warmup=0)

    def test_a_benchmark_missing_on_one_side_is_skipped(self):
        self.bench(self.a, "x", 0.01)
        self.bench(self.b, "x", 0.01)
        self.bench(self.a, "only_a", 0.01)
        out = self.root / "r.json"
        code = pw.main([str(self.a), str(self.b), "--bench", "x", "--bench", "only_a", "--reps", "2", "--json", str(out)])
        self.assertEqual(0, code)
        self.assertEqual(["x"], [r["benchmark"] for r in json.loads(out.read_text())])

    def test_nothing_to_compare_is_an_error(self):
        self.assertEqual(1, pw.main([str(self.a), str(self.b), "--bench", "nothing", "--reps", "1"]))

    def test_percentiles_and_summary(self):
        self.assertEqual(2.0, pw.percentile([1.0, 2.0, 3.0], 0.5))
        summary = pw.summarise([1.0, 1.1, 0.9, 1.0, 1.0])
        self.assertAlmostEqual(1.0, summary["median"])
        self.assertAlmostEqual(0.9, summary["min"])
        self.assertAlmostEqual(1.1, summary["max"])

    def test_the_report_says_when_both_sides_are_the_same_build(self):
        self.bench(self.a, "x", 0.01)
        result = pw.measure(self.a, self.a, "x", reps=2, pin=None, warmup=0)
        self.assertIn("same build", pw.render([result], same_build=True))
        self.assertIn("| `x` |", pw.render([result], same_build=False))


if __name__ == "__main__":
    unittest.main()
