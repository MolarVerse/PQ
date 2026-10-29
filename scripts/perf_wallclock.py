#!/usr/bin/env python3
"""Paired, interleaved wall-clock and CPU timing of the perf benchmarks.

Callgrind (scripts/perf_gate.sh) counts instructions and cannot see cache
behaviour, memory layout or vectorisation. This script times the same
benchmarks of two builds (A = base, B = PR) on the SAME machine, interleaved
(A B, B A, A B, ...) so that both see the same machine state, and reports the
median of the per-pair ratios B/A with their spread.

Run it with the same directory twice (A against A) to measure how large the
ratio varies for identical code on a given machine: that spread is the
smallest change this method can see.

    perf_wallclock.py A_DIR B_DIR [--bench NAME ...] [--reps N] [--pin CPU]
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_BENCHMARKS = ("endToEnd", "interWaterN1728", "interWater")


def run_once(binary, pin):
    """(wall seconds, cpu seconds) of one run; stdout/stderr are discarded."""
    command = [str(binary)]
    if pin is not None and shutil.which("taskset"):
        command = ["taskset", "-c", str(pin), *command]
    start = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, status, usage = os.wait4(process.pid, 0)
    wall = time.perf_counter() - start
    process.returncode = os.waitstatus_to_exitcode(status)
    if process.returncode != 0:
        raise RuntimeError(f"{binary} exited with {process.returncode}")
    return wall, usage.ru_utime + usage.ru_stime


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarise(ratios):
    """Median ratio and its spread over the pairs (as percentages around 1)."""
    return {
        "median": statistics.median(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "q1": percentile(ratios, 0.25),
        "q3": percentile(ratios, 0.75),
    }


def measure(dir_a, dir_b, name, reps, pin, warmup=1):
    """Paired measurement of one benchmark; returns a dict of the results."""
    binary_a = Path(dir_a) / f"perf_{name}"
    binary_b = Path(dir_b) / f"perf_{name}"
    for _ in range(warmup):
        run_once(binary_a, pin)
        run_once(binary_b, pin)
    wall_ratios, cpu_ratios, wall_a, wall_b = [], [], [], []
    for index in range(reps):
        first_is_a = index % 2 == 0
        order = (("a", binary_a), ("b", binary_b)) if first_is_a else (("b", binary_b), ("a", binary_a))
        result = {}
        for side, binary in order:
            result[side] = run_once(binary, pin)
        wall_ratios.append(result["b"][0] / result["a"][0])
        cpu_ratios.append(result["b"][1] / result["a"][1] if result["a"][1] > 0 else float("nan"))
        wall_a.append(result["a"][0])
        wall_b.append(result["b"][0])
    return {
        "benchmark": name,
        "reps": reps,
        "wall_a_median": statistics.median(wall_a),
        "wall_b_median": statistics.median(wall_b),
        "wall": summarise(wall_ratios),
        "cpu": summarise([r for r in cpu_ratios if r == r]),
    }


def percent(ratio):
    return f"{(ratio - 1.0) * 100:+.1f}%"


def render(results, same_build):
    lines = ["| benchmark | median time A | median time B | wall B/A (median) | wall range | cpu B/A (median) | cpu range |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        wall, cpu = r["wall"], r["cpu"]
        lines.append(
            f"| `{r['benchmark']}` | {r['wall_a_median'] * 1000:.1f} ms | {r['wall_b_median'] * 1000:.1f} ms "
            f"| {percent(wall['median'])} | {percent(wall['min'])} to {percent(wall['max'])} "
            f"| {percent(cpu['median'])} | {percent(cpu['min'])} to {percent(cpu['max'])} |"
        )
    note = ("A and B are the same build: the ratios show the spread of identical code on this machine."
            if same_build else "Median of the per-pair ratios B/A over interleaved runs; range = smallest to largest pair.")
    return "\n".join(lines) + f"\n\n{note}\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir_a", help="directory with the perf_* binaries of A (base)")
    parser.add_argument("dir_b", help="directory with the perf_* binaries of B (PR); may equal A")
    parser.add_argument("--bench", action="append", help="benchmark name without the perf_ prefix (repeatable)")
    parser.add_argument("--reps", type=int, default=15, help="interleaved pairs per benchmark")
    parser.add_argument("--pin", type=int, default=None, help="CPU to pin to with taskset (if available)")
    parser.add_argument("--json", help="also write the results as JSON to this file")
    args = parser.parse_args(argv)

    names = args.bench or list(DEFAULT_BENCHMARKS)
    results = []
    for name in names:
        for directory in (args.dir_a, args.dir_b):
            if not (Path(directory) / f"perf_{name}").exists():
                print(f"skipping {name}: no perf_{name} in {directory}", file=sys.stderr)
                break
        else:
            results.append(measure(args.dir_a, args.dir_b, name, args.reps, args.pin))
    if not results:
        print("no benchmark found in both directories", file=sys.stderr)
        return 1
    print(render(results, Path(args.dir_a).resolve() == Path(args.dir_b).resolve()))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
