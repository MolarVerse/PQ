"""Tests of scripts/perf_timing.py.

The statistics are tested on synthetic times, the script itself with fake
benchmarks: shell scripts that print a fixed "PQ_PERF_TIME <cpu ns> <wall ns>"
line like the preloaded timer library does.
"""

import importlib.util
import os
import random
import stat
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "perf_timing.py"

spec = importlib.util.spec_from_file_location("perf_timing", SCRIPT)
perf_timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf_timing)


def noisy_rounds(rng, rounds, copies, mean_ns, copy_spread=0.0, noise=0.01):
    """Per round the times of every copy: a fixed factor per copy (layout)
    and a random one per run (jitter)."""
    factors = [1.0 + rng.gauss(0.0, copy_spread) for _ in range(copies)]
    return [
        [mean_ns * f * (1.0 + rng.gauss(0.0, noise)) for f in factors]
        for _ in range(rounds)
    ]


class StatisticsTest(unittest.TestCase):
    def judge(self, pr_mean, base_mean, copy_spread=0.0, rounds=15, seed=1):
        rng = random.Random(seed)
        pr = noisy_rounds(rng, rounds, 4, pr_mean, copy_spread)
        base = noisy_rounds(rng, rounds, 4, base_mean, copy_spread)
        return perf_timing.judge("a", pr, base, 0.10, random.Random(0))

    def test_the_bootstrap_interval_contains_the_median(self):
        values = [1.0 + 0.01 * i for i in range(15)]
        low, high = perf_timing.bootstrap_ci(values, 0.95, random.Random(0))
        self.assertLessEqual(low, 1.07)
        self.assertGreaterEqual(high, 1.07)

    def test_the_bootstrap_is_deterministic_for_a_seed(self):
        values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02]
        first = perf_timing.bootstrap_ci(values, 0.95, random.Random(3))
        second = perf_timing.bootstrap_ci(values, 0.95, random.Random(3))
        self.assertEqual(first, second)

    def test_equal_binaries_are_ok(self):
        self.assertEqual("ok", self.judge(1e7, 1e7).status)

    def test_a_clear_slowdown_is_a_regression(self):
        verdict = self.judge(1.3e7, 1e7)
        self.assertEqual("regression", verdict.status)
        self.assertAlmostEqual(1.3, verdict.ratio, delta=0.05)

    def test_a_clear_speedup_is_an_improvement(self):
        self.assertEqual("improvement", self.judge(0.7e7, 1e7).status)

    def test_a_slowdown_below_the_threshold_is_not_a_regression(self):
        self.assertEqual("ok", self.judge(1.05e7, 1e7).status)

    def test_a_slowdown_at_the_threshold_is_not_enough(self):
        # the whole interval has to be above the threshold
        self.assertEqual("ok", self.judge(1.10e7, 1e7).status)

    def test_a_large_layout_noise_makes_the_benchmark_noisy(self):
        verdict = self.judge(1e7, 1e7, copy_spread=0.15)
        self.assertEqual("noisy", verdict.status)

    def test_the_layout_noise_widens_the_interval(self):
        calm = self.judge(1e7, 1e7, copy_spread=0.0)
        shaky = self.judge(1e7, 1e7, copy_spread=0.03)
        self.assertGreater(shaky.layout, calm.layout)
        self.assertGreater(shaky.high - shaky.low, calm.high - calm.low)

    def test_the_layout_noise_keeps_a_difference_within_it_from_failing(self):
        # +8 % on average, copies of the same binary differ by 4 % each: the
        # measured ratio of a draw is often above +10 %, the interval is not
        regressions = sum(
            self.judge(1.08e7, 1e7, copy_spread=0.04, seed=seed).status
            == "regression"
            for seed in range(100)
        )
        self.assertLessEqual(regressions, 5)

    def test_no_false_regression_for_equal_binaries(self):
        regressions = sum(
            self.judge(1e7, 1e7, copy_spread=0.03, seed=seed).status == "regression"
            for seed in range(100)
        )
        self.assertEqual(0, regressions)


class FakeRunner:
    """Times by binary name with noise; no processes."""

    def __init__(self, times, seed=0):
        self.times = times
        self.rng = random.Random(seed)
        self.calls = {}
        self._count = 0
        self.closed = False

    def copies(self, binary, count):
        paths = []
        for _ in range(count):
            self._count += 1
            paths.append(f"{binary}#{self._count}")
        return paths

    def run(self, path):
        binary = path.split("#")[0]
        self.calls[binary] = self.calls.get(binary, 0) + 1
        mean = self.times[binary]
        if mean is None:
            return None
        return int(mean * (1.0 + self.rng.gauss(0.0, 0.01)))

    def close(self):
        self.closed = True


def options(**overrides):
    values = dict(
        threshold=0.10, rounds=15, confirm_rounds=41, copies=4, min_ms=5.0,
        cpu=None, seed=0, report_only=False, append=None,
    )
    values.update(overrides)
    return Namespace(**values)


class CompareTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.pr, self.base = self.root / "pr", self.root / "base"
        self.pr.mkdir()
        self.base.mkdir()

    def make(self, directory, name):
        path = directory / f"perf_{name}"
        path.write_text("#!/bin/sh\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return str(path)

    def compare(self, benchmarks, **overrides):
        times = {}
        for name, (pr_ns, base_ns) in benchmarks.items():
            if pr_ns != "absent":
                times[self.make(self.pr, name)] = pr_ns
            if base_ns != "absent":
                times[self.make(self.base, name)] = base_ns
        runner = FakeRunner(times)
        messages = []
        result = perf_timing.compare(
            self.pr, self.base, "lib", options(**overrides), runner=runner,
            out=messages.append,
        )
        return result, runner, messages

    def test_each_benchmark_gets_a_verdict(self):
        result, runner, _ = self.compare({"a": (1e7, 1e7), "b": (2.0e7, 1e7)})
        verdicts = {v.name: v for v in result.verdicts}
        self.assertEqual("ok", verdicts["a"].status)
        self.assertEqual("regression", verdicts["b"].status)
        self.assertTrue(runner.closed)

    def test_a_loop_shorter_than_the_minimum_is_listed_not_judged(self):
        result, _, _ = self.compare({"a": (2e6, 2e6)})
        self.assertEqual([], result.verdicts)
        self.assertEqual(["a"], [name for name, _ in result.too_short])

    def test_a_new_benchmark_is_not_compared(self):
        result, _, _ = self.compare({"a": (1e7, "absent")})
        self.assertEqual([], result.verdicts)
        self.assertEqual([], result.unmeasured)

    def test_a_benchmark_without_a_time_is_listed(self):
        result, _, _ = self.compare({"a": (None, 1e7)})
        self.assertEqual(["a"], result.unmeasured)

    def test_a_regression_is_measured_again_and_decided_by_the_second_run(self):
        result, runner, messages = self.compare({"a": (1.4e7, 1e7)})
        self.assertEqual(41, result.verdicts[0].rounds)
        self.assertTrue(result.verdicts[0].confirmed)
        self.assertEqual(1, len(messages))
        # warm-up of the base, 15 + 41 rounds of 4 copies each; the PR binary
        # also runs three times to measure the length of the loop
        self.assertEqual(1 + 4 * 56, runner.calls[str(self.base / "perf_a")])
        self.assertEqual(3 + 4 * 56, runner.calls[str(self.pr / "perf_a")])

    def test_an_unchanged_benchmark_is_measured_only_once(self):
        result, _, messages = self.compare({"a": (1e7, 1e7)})
        self.assertEqual(15, result.verdicts[0].rounds)
        self.assertFalse(result.verdicts[0].confirmed)
        self.assertEqual([], messages)

    def test_the_report_lists_the_verdicts_and_the_short_loops(self):
        result, _, _ = self.compare({"a": (1.4e7, 1e7), "short": (1e6, 1e6)})
        report = perf_timing.render(result, options())
        self.assertIn("| `a` |", report)
        self.assertIn("❌", report)
        self.assertIn("`short`", report)
        self.assertIn("Loop shorter than 5 ms", report)

    def test_the_length_of_the_loop_is_the_median_of_three_runs(self):
        # one slow run of a 1 ms loop must not get it judged, one fast run of
        # a 6 ms loop must not get it dropped
        pr_dir = str(self.pr)

        class Spiky(FakeRunner):
            def __init__(self, times, spike):
                super().__init__(times)
                self.spike = spike
                self.first = {}

            def run(self, path):
                binary = path.split("#")[0]
                count = self.first.get(binary, 0)
                self.first[binary] = count + 1
                if binary.startswith(pr_dir) and count == 1:
                    return int(self.spike)
                return super().run(path)

        for mean, spike, judged in ((1e6, 9e6, False), (6e6, 1e6, True)):
            with self.subTest(mean=mean):
                pr, base = self.make(self.pr, "a"), self.make(self.base, "a")
                runner = Spiky({pr: mean, base: mean}, spike)
                result = perf_timing.compare(
                    self.pr, self.base, "lib", options(), runner=runner,
                    out=lambda _: None,
                )
                self.assertEqual(judged, bool(result.verdicts))


class ScriptTest(unittest.TestCase):
    """The whole script with fake benchmarks (no timer library needed)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.pr, self.base = self.root / "pr", self.root / "base"
        self.pr.mkdir()
        self.base.mkdir()
        (self.pr / "libperf_timer.so").write_text("")

    def bench(self, directory, name, ns):
        path = directory / f"perf_{name}"
        path.write_text(f'#!/bin/sh\necho "PQ_PERF_TIME {ns} {ns}" >&2\n')
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def run_script(self, *extra):
        import subprocess

        env = dict(os.environ)
        env.pop("GITHUB_STEP_SUMMARY", None)
        return subprocess.run(
            ["python3", str(SCRIPT), str(self.pr), str(self.base),
             "--rounds", "5", "--confirm-rounds", "7", *extra],
            capture_output=True, text=True, env=env,
        )

    def test_equal_times_pass(self):
        self.bench(self.pr, "a", 20_000_000)
        self.bench(self.base, "a", 20_000_000)
        result = self.run_script()
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("| `a` | 20.00 | 20.00 | +0.0% |", result.stdout)

    def test_a_regression_fails_with_an_error_annotation(self):
        self.bench(self.pr, "a", 30_000_000)
        self.bench(self.base, "a", 20_000_000)
        result = self.run_script()
        self.assertEqual(1, result.returncode)
        self.assertIn("::error::a is slower: +50.0%", result.stdout)

    def test_report_only_never_fails(self):
        self.bench(self.pr, "a", 30_000_000)
        self.bench(self.base, "a", 20_000_000)
        self.assertEqual(0, self.run_script("--report-only").returncode)

    def test_the_threshold_can_be_changed(self):
        self.bench(self.pr, "a", 30_000_000)
        self.bench(self.base, "a", 20_000_000)
        self.assertEqual(0, self.run_script("--threshold", "80").returncode)

    def test_the_report_is_appended_to_a_file(self):
        self.bench(self.pr, "a", 20_000_000)
        self.bench(self.base, "a", 20_000_000)
        report = self.root / "perf_report.md"
        report.write_text("existing\n")
        self.run_script("--append", str(report))
        text = report.read_text()
        self.assertTrue(text.startswith("existing\n"))
        self.assertIn("Timing of the measured loop", text)

    def test_without_the_timer_library_nothing_is_timed(self):
        (self.pr / "libperf_timer.so").unlink()
        self.bench(self.pr, "a", 30_000_000)
        self.bench(self.base, "a", 20_000_000)
        result = self.run_script()
        self.assertEqual(0, result.returncode)
        self.assertIn("timing skipped", result.stdout)

    def test_a_failing_benchmark_is_listed_as_unmeasured(self):
        path = self.pr / "perf_a"
        path.write_text("#!/bin/sh\nexit 3\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        self.bench(self.base, "a", 20_000_000)
        result = self.run_script()
        self.assertEqual(0, result.returncode)
        self.assertIn("No time reported", result.stdout)

    def test_fewer_than_two_copies_are_rejected(self):
        self.bench(self.pr, "a", 20_000_000)
        self.bench(self.base, "a", 20_000_000)
        self.assertNotEqual(0, self.run_script("--copies", "1").returncode)


if __name__ == "__main__":
    unittest.main()
