#!/usr/bin/env python3
"""Paired, statistically tested timing comparison of the perf benchmarks.

scripts/perf_gate.sh counts instructions, which is exact but blind to memory
bound code, system calls and file I/O. This script compares the CPU time of the
measured loop of every benchmark of a pull request with the one of its base,
measured on the same machine in the same run.

How the noise is handled:

* The time comes from libperf_timer.so (benchmarks/perf/timer): the CPU time
  between the benchmark's perfAllocBegin() marker and the end of the process,
  so startup and setup are not part of it.
* Byte-identical copies of a binary run at different speeds (up to several
  percent for allocation heavy code: where the file lands in the page cache
  changes cache and page behaviour), and base and PR are different files.
  Every benchmark is therefore measured with COPIES (default 4) copies of the
  base and of the PR binary. The ratio of a round is mean(PR copies) /
  mean(base copies); the spread between the copies is the layout noise, and
  its standard error widens the interval (t distribution), as in a two-sample
  test of the copy means.
* Every round runs all copies in a random order (seeded), on one pinned CPU,
  without address space layout randomisation, with an empty environment and
  from paths of the same length. The result is the median ratio with a
  bootstrap interval over the rounds, widened by the layout noise.
* A regression needs the whole widened 95 % interval above the threshold, and
  is confirmed by a second, longer measurement that decides alone. If the
  layout noise alone is larger than the threshold the benchmark is "noisy" and
  not judged. Loops that are too short to be timed are listed, not judged.

Usage: perf_timing.py <pr_bench_dir> <base_bench_dir> [options]
Exit code 1: a confirmed regression (unless --report-only) or a failed run.
"""

import argparse
import ctypes
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ADDR_NO_RANDOMIZE = 0x0040000
TIME_LINE = re.compile(r"^PQ_PERF_TIME (\d+) (\d+)$", re.MULTILINE)


@dataclass
class Verdict:
    name: str
    status: str   # ok, regression, improvement, noisy
    base_ms: float
    pr_ms: float
    ratio: float
    low: float
    high: float
    rounds: int
    layout: float = 0.0   # half width added to the interval by the copies
    confirmed: bool = False


@dataclass
class Result:
    verdicts: list = field(default_factory=list)
    too_short: list = field(default_factory=list)
    unmeasured: list = field(default_factory=list)


def bootstrap_ci(values, confidence, rng, resamples=2000):
    """Confidence interval of the median of values (percentile bootstrap)."""
    n = len(values)
    medians = sorted(
        statistics.median(rng.choices(values, k=n)) for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    low = medians[max(0, math.floor(tail * resamples))]
    high = medians[min(resamples - 1, math.ceil((1.0 - tail) * resamples) - 1)]
    return low, high


# two-sided 95 % quantiles of the t distribution by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def relative_variance(medians):
    mean = statistics.fmean(medians)
    return statistics.variance(medians) / (mean * mean)


def judge(name, pr_rounds, base_rounds, threshold, rng, confirmed=False):
    """Verdict from the times in ns; pr_rounds and base_rounds hold, per round,
    the times of every copy of the PR and of the base binary."""
    copies = len(pr_rounds[0])
    ratios = [
        statistics.fmean(p) / statistics.fmean(b)
        for p, b in zip(pr_rounds, base_rounds)
    ]
    low, high = bootstrap_ci(ratios, 0.95, rng)
    median = statistics.median(ratios)

    # layout noise: spread of the per-copy medians (constant over the rounds)
    pr_copy = [statistics.median(r[j] for r in pr_rounds) for j in range(copies)]
    base_copy = [statistics.median(r[j] for r in base_rounds) for j in range(copies)]
    pooled = (relative_variance(pr_copy) + relative_variance(base_copy)) / 2.0
    degrees = min(10, 2 * (copies - 1))
    margin = T95[degrees] * math.sqrt(pooled * 2.0 / copies)

    if margin > threshold:
        status = "noisy"
    elif low - margin > 1.0 + threshold:
        status = "regression"
    elif high + margin < 1.0 - threshold:
        status = "improvement"
    else:
        status = "ok"

    return Verdict(
        name=name,
        status=status,
        base_ms=statistics.fmean(base_copy) / 1e6,
        pr_ms=statistics.fmean(pr_copy) / 1e6,
        ratio=median,
        low=low - margin,
        high=high + margin,
        rounds=len(ratios),
        layout=margin,
        confirmed=confirmed,
    )


class Runner:
    """Runs benchmark binaries under the timer library."""

    def __init__(self, library, cpu):
        self.library = str(library)
        self.cpu = cpu
        self._links = Path(tempfile.mkdtemp(prefix="pq_perf_timing_"))
        self._count = 0

    def close(self):
        shutil.rmtree(self._links, ignore_errors=True)

    def copies(self, binary, count):
        """count real copies of a binary (own page cache pages), each in a
        directory of fixed name length so the program path is the same for
        every binary."""
        paths = []
        for _ in range(count):
            self._count += 1
            directory = self._links / f"{self._count:04d}"
            directory.mkdir()
            target = directory / "b"
            shutil.copyfile(binary, target)
            target.chmod(0o755)
            paths.append(str(target))
        return paths

    def _prepare_child(self):
        if self.cpu is not None:
            try:
                os.sched_setaffinity(0, {self.cpu})
            except OSError:
                pass
        try:
            ctypes.CDLL(None).personality(ADDR_NO_RANDOMIZE)
        except (OSError, AttributeError):
            pass

    def run(self, path):
        """CPU nanoseconds of the measured loop, or None if there is none."""
        done = subprocess.run(
            ["env", "-i", f"LD_PRELOAD={self.library}", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=self._prepare_child,
        )
        if done.returncode != 0:
            return None
        found = TIME_LINE.findall(done.stderr)
        return int(found[-1][0]) if found else None


def measure(runner, base_paths, pr_paths, rounds, rng):
    """Interleaved rounds of all copies. Returns (pr_rounds, base_rounds), each
    a list per round of the times of the copies, or None on a failure."""
    pr_rounds, base_rounds = [], []
    jobs = [("pr", i, p) for i, p in enumerate(pr_paths)] + [
        ("base", i, p) for i, p in enumerate(base_paths)
    ]

    for _ in range(rounds):
        rng.shuffle(jobs)
        times = {"pr": [0] * len(pr_paths), "base": [0] * len(base_paths)}
        for kind, index, path in jobs:
            ns = runner.run(path)
            if ns is None or ns <= 0:
                return None
            times[kind][index] = ns
        pr_rounds.append(times["pr"])
        base_rounds.append(times["base"])

    return pr_rounds, base_rounds


def benchmarks(directory):
    found = {}
    for path in sorted(Path(directory).glob("perf_*")):
        if path.is_file() and os.access(path, os.X_OK):
            found[path.name[len("perf_"):]] = path
    return found


def compare(pr_dir, base_dir, library, options, runner=None, out=print):
    pr_benchmarks = benchmarks(pr_dir)
    base_benchmarks = benchmarks(base_dir)
    result = Result()
    runner = runner or Runner(library, options.cpu)
    rng = random.Random(options.seed)

    try:
        for name, pr_binary in pr_benchmarks.items():
            if name not in base_benchmarks:
                continue   # new benchmark: nothing to compare with

            pr_paths = runner.copies(pr_binary, options.copies)
            base_paths = runner.copies(base_benchmarks[name], options.copies)

            # warm-up (page cache, dynamic linker), also gives the loop length
            runner.run(base_paths[0])
            first = runner.run(pr_paths[0])
            if first is None:
                result.unmeasured.append(name)
                continue
            if first < options.min_ms * 1e6:
                result.too_short.append((name, first / 1e6))
                continue

            data = measure(runner, base_paths, pr_paths, options.rounds, rng)
            if data is None:
                result.unmeasured.append(name)
                continue

            verdict = judge(name, *data, options.threshold, rng)

            if verdict.status in ("regression", "noisy"):
                out(
                    f"{name}: {verdict.status} in {verdict.rounds} rounds "
                    f"({verdict.ratio - 1:+.1%}), measuring again"
                )
                data = measure(
                    runner, base_paths, pr_paths, options.confirm_rounds, rng
                )
                if data is None:
                    result.unmeasured.append(name)
                    continue
                verdict = judge(
                    name, *data, options.threshold, rng, confirmed=True
                )

            result.verdicts.append(verdict)
    finally:
        runner.close()

    return result


EMOJI = {
    "ok": "✅",
    "regression": "❌",
    "improvement": "🎉",
    "noisy": "⚠️",
}


def render(result, options):
    lines = [
        "<details><summary>Timing of the measured loop (paired, same runner)"
        "</summary>",
        "",
        "CPU time of the loop in ms (median of the interleaved rounds), "
        "change of the PR against the base as the median of the paired "
        "ratios and its 95 % interval (bootstrap over the rounds, widened by "
        f"the spread of {options.copies} copies of each binary). A "
        f"regression needs the whole interval above +{options.threshold:.0%} "
        f"and is confirmed by a second measurement of "
        f"{options.confirm_rounds} rounds. ⚠️ = the layout noise of the "
        "binaries alone exceeds the threshold, not judged.",
        "",
        "| benchmark | base | PR | Δ | 95 % interval | rounds | |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in result.verdicts:
        lines.append(
            f"| `{v.name}` | {v.base_ms:.2f} | {v.pr_ms:.2f} | "
            f"{v.ratio - 1:+.1%} | {v.low - 1:+.1%} to {v.high - 1:+.1%} | "
            f"{v.rounds} | {EMOJI[v.status]} |"
        )
    if result.too_short:
        short = ", ".join(f"`{n}`" for n, _ in result.too_short)
        lines += [
            "",
            f"Loop shorter than {options.min_ms:g} ms, not timed: {short}",
        ]
    if result.unmeasured:
        names = ", ".join(f"`{n}`" for n in result.unmeasured)
        lines += ["", f"No time reported (no marker or failed): {names}"]
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pr_dir")
    parser.add_argument("base_dir")
    parser.add_argument(
        "--library",
        help="libperf_timer.so (default: next to the PR benchmarks)",
    )
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="regression threshold in percent (default 10)")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--confirm-rounds", type=int, default=41)
    parser.add_argument("--copies", type=int, default=4,
                        help="copies of each binary (default 4, at least 2)")
    parser.add_argument("--min-ms", type=float, default=5.0,
                        help="shortest loop that is judged (default 5)")
    parser.add_argument("--cpu", type=int, default=None,
                        help="CPU to pin to (default: the last allowed one)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-only", action="store_true",
                        help="never fail because of a regression")
    parser.add_argument("--append", metavar="FILE",
                        help="append the report to this file")
    options = parser.parse_args(argv)
    options.threshold /= 100.0
    if options.copies < 2:
        parser.error("--copies must be at least 2")
    return options


def main(argv=None):
    options = parse_arguments(argv if argv is not None else sys.argv[1:])
    library = Path(options.library or Path(options.pr_dir) / "libperf_timer.so")

    if not library.is_file():
        print(f"no timer library ({library}), timing skipped")
        return 0

    if options.cpu is None:
        allowed = sorted(os.sched_getaffinity(0))
        options.cpu = allowed[-1] if allowed else None

    result = compare(options.pr_dir, options.base_dir, library, options)
    report = render(result, options)
    print(report)

    if options.append:
        with open(options.append, "a", encoding="utf-8") as handle:
            handle.write("\n" + report)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(report)

    regressions = [v for v in result.verdicts if v.status == "regression"]
    for v in regressions:
        print(
            f"::error::{v.name} is slower: {v.ratio - 1:+.1%} "
            f"(95 % interval {v.low - 1:+.1%} to {v.high - 1:+.1%}, "
            f"threshold +{options.threshold:.0%})"
        )

    if regressions and not options.report_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
