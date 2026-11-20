"""End-to-end performance scenarios: real PQ runs of examples and integration inputs.

The fixed-work benchmarks of benchmarks/perf measure one part of the code at a
time. A scenario runs the whole program, as a user would, on an input of the
examples or of the integration tests, and compares your build with a base:

* instruction counts of a short run under callgrind (deterministic to about
  1e-7, see below); a change that makes the whole run more expensive shows up
  here, whichever part of the code it is in;
* the CPU time of a longer run (about a second), paired and interleaved with the
  base, with the statistics of scripts/perf_timing.py. This sees what counts
  cannot: file output, memory behaviour, startup.

benchmarks/scenarios.json lists the scenarios. Both builds run exactly the same
input files, taken from your working tree, with only the number of steps
changed (and a fixed random seed). If the base does not understand something
the input uses, the scenario is reported as failing for the base.

The counts of two runs of the same binary differ by about 1e-7 (paths and the
elapsed time end up in the output), so a count verdict needs a threshold well
above that; the default of 1 % is.

Used by scripts/perf_local.py (--suite scenarios).
"""

import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import perf_timing

SCENARIOS_FILE = Path("benchmarks") / "scenarios.json"
RANDOM_SEED = 1
I_REFS = re.compile(r"I\s+refs:\s+([\d,]+)")


@dataclass(frozen=True)
class Scenario:
    name: str
    source: str  # "examples/h2o_mm" or "integration_tests/cgo.zip:cgo"
    input: str
    steps: int  # steps of the timed run (about one second)
    count_steps: int  # steps of the run under callgrind
    description: str = ""


def load_scenarios(text):
    """The scenarios of the JSON text; a malformed file is an error."""
    entries = json.loads(text)
    scenarios = []

    for entry in entries:
        missing = {"name", "source", "input", "steps", "count_steps"} - set(entry)
        if missing:
            raise ValueError(f"scenario {entry.get('name')}: missing {sorted(missing)}")
        scenarios.append(Scenario(**entry))

    names = [s.name for s in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("scenario names are not unique")

    return scenarios


def apply_overrides(text, overrides):
    """Set keys of a PQ input file ("key = value;"), ignoring comments.

    A key that is present is changed in place, a missing one is appended.
    """
    lines = text.splitlines()
    missing = dict(overrides)

    for index, line in enumerate(lines):
        code, hash_sign, comment = line.partition("#")

        for key, value in overrides.items():
            pattern = re.compile(rf"(?<![\w-]){re.escape(key)}(\s*=\s*)[^;]*;")
            code, count = pattern.subn(
                lambda match, k=key, v=value: f"{k}{match.group(1)}{v};", code
            )
            if count:
                missing.pop(key, None)

        lines[index] = code + hash_sign + comment

    lines.extend(f"{key} = {value};" for key, value in missing.items())

    return "\n".join(lines) + "\n"


def prepare(scenario, root, destination, steps):
    """A directory with the files of the scenario and its input set to steps."""
    directory = Path(destination)
    if directory.exists():
        shutil.rmtree(directory)

    archive, _, inner = scenario.source.partition(":")
    if inner:
        with zipfile.ZipFile(Path(root) / archive) as zipped:
            members = [m for m in zipped.namelist() if m.startswith(inner + "/")]
            with tempfile.TemporaryDirectory(prefix="pq_scenario_") as unpacked:
                zipped.extractall(unpacked, members=members)
                shutil.copytree(Path(unpacked) / inner, directory)
    else:
        shutil.copytree(Path(root) / archive, directory)

    for reference in directory.glob("ref_data*"):
        shutil.rmtree(reference, ignore_errors=True)

    path = directory / scenario.input
    if not path.is_file():
        raise ValueError(f"{scenario.name}: {scenario.input} is not in {scenario.source}")

    path.write_text(
        apply_overrides(
            path.read_text(), {"nstep": steps, "random_seed": RANDOM_SEED}
        )
    )

    return directory


def run_cpu_ns(binary, template, input_name, prepare_child):
    """CPU nanoseconds (user + system) of one run in a fresh copy of template.

    None if the program fails. The copy of the files is not part of the time.
    """
    with tempfile.TemporaryDirectory(prefix="pq_scenario_run_") as directory:
        run_directory = Path(directory) / "run"
        shutil.copytree(template, run_directory)

        process = subprocess.Popen(
            ["env", "-i", str(binary), input_name],
            cwd=run_directory,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=prepare_child,
        )
        _, status, usage = os.wait4(process.pid, 0)

        process.returncode = os.waitstatus_to_exitcode(status)
        if process.returncode != 0:
            return None

        return int((usage.ru_utime + usage.ru_stime) * 1e9)


def instruction_count(valgrind, binary, template, input_name):
    """Instructions of one run under callgrind, or None if it fails."""
    with tempfile.TemporaryDirectory(prefix="pq_scenario_run_") as directory:
        run_directory = Path(directory) / "run"
        shutil.copytree(template, run_directory)

        result = subprocess.run(
            ["env", "-i", valgrind, "--tool=callgrind",
             "--callgrind-out-file=/dev/null", str(binary), input_name],
            cwd=run_directory, capture_output=True, text=True, check=False,
        )  # fmt: skip

    found = I_REFS.findall(result.stderr)
    if result.returncode != 0 or not found:
        return None

    return int(found[-1].replace(",", ""))


@dataclass
class CountVerdict:
    name: str
    base: int | None
    pr: int | None
    percent: float | None
    status: str  # ok, regression, improvement, failed, new


def count_verdict(name, base, pr, threshold_percent):
    if pr is None:
        return CountVerdict(name, base, pr, None, "failed")
    if base is None:
        return CountVerdict(name, base, pr, None, "base failed")

    percent = (pr - base) / base * 100.0
    if percent > threshold_percent:
        status = "regression"
    elif percent < -threshold_percent:
        status = "improvement"
    else:
        status = "ok"

    return CountVerdict(name, base, pr, percent, status)


def count_scenarios(scenarios, templates, pr_binary, base_binary, valgrind,
                    threshold_percent, base_cache, out=print):
    """Instruction counts of every scenario for both binaries.

    base_cache maps a scenario name to the cached base count and is extended
    with the counts measured now.
    """
    runner = perf_timing.Runner(None, None)
    verdicts = []

    try:
        pr_path = runner.copies(pr_binary, 1)[0]
        base_path = runner.copies(base_binary, 1)[0]

        for scenario in scenarios:
            template = templates[scenario.name]
            pr = instruction_count(valgrind, pr_path, template, scenario.input)

            base = base_cache.get(scenario.name)
            if base is None:
                base = instruction_count(valgrind, base_path, template, scenario.input)
                if base is not None:
                    base_cache[scenario.name] = base

            verdict = count_verdict(scenario.name, base, pr, threshold_percent)
            verdicts.append(verdict)
            out(render_count(verdict))
    finally:
        runner.close()

    return verdicts


def render_count(verdict):
    if verdict.percent is None:
        return f"{verdict.name}: {verdict.status}"

    marker = {"regression": "  <-- REGRESSION", "improvement": "  (improved)"}
    return (
        f"{verdict.name}: {verdict.percent:+.2f}%  "
        f"(Ir base={verdict.base:,} pr={verdict.pr:,})"
        f"{marker.get(verdict.status, '')}"
    )


def time_scenarios(scenarios, templates, pr_binary, base_binary, cpu, rounds,
                   copies, threshold, min_ms=20.0, seed=0, out=print):
    """Paired, interleaved CPU time of every scenario; returns the verdicts.

    Returns (verdicts, notes): a verdict per judged scenario and a list of
    scenarios that were too short or failed.
    """
    runner = perf_timing.Runner(None, cpu)
    rng = random.Random(seed)
    verdicts, notes = [], []

    try:
        for scenario in scenarios:
            template = templates[scenario.name]
            pr_paths = runner.copies(pr_binary, copies)
            base_paths = runner.copies(base_binary, copies)

            def run(path, template=template, scenario=scenario):
                return run_cpu_ns(path, template, scenario.input,
                                  runner._prepare_child)

            base_warm = run(base_paths[0])  # also warms the page cache
            pr_warm = run(pr_paths[0])
            if pr_warm is None or base_warm is None:
                who = "the PR" if pr_warm is None else "the base"
                notes.append(f"{scenario.name}: the run of {who} failed")
                continue
            if base_warm < min_ms * 1e6:
                notes.append(f"{scenario.name}: {base_warm / 1e6:.1f} ms, too short")
                continue

            jobs = [("pr", i, p) for i, p in enumerate(pr_paths)]
            jobs += [("base", i, p) for i, p in enumerate(base_paths)]
            pr_rounds, base_rounds = [], []
            failed = False

            for _ in range(rounds):
                rng.shuffle(jobs)
                times = {"pr": [0] * copies, "base": [0] * copies}
                for kind, index, path in jobs:
                    ns = run(path)
                    if ns is None:
                        failed = True
                        break
                    times[kind][index] = ns
                if failed:
                    break
                pr_rounds.append(times["pr"])
                base_rounds.append(times["base"])

            if failed:
                notes.append(f"{scenario.name}: a run failed")
                continue

            verdict = perf_timing.judge(scenario.name, pr_rounds, base_rounds,
                                        threshold, rng)
            verdicts.append(verdict)
            out(render_time(verdict))
    finally:
        runner.close()

    return verdicts, notes


def render_time(verdict):
    marker = {"regression": "  <-- SLOWER", "improvement": "  (faster)",
              "noisy": "  (noisy: layout noise above the threshold)"}
    return (
        f"{verdict.name}: {verdict.base_ms:.0f} ms -> {verdict.pr_ms:.0f} ms, "
        f"{verdict.ratio - 1:+.1%} (95 % interval {verdict.low - 1:+.1%} to "
        f"{verdict.high - 1:+.1%}, {verdict.rounds} rounds)"
        f"{marker.get(verdict.status, '')}"
    )


def render_report(count_verdicts, time_verdicts, notes):
    """The results as markdown."""
    lines = ["### End-to-end scenarios", ""]

    if count_verdicts:
        lines += ["| scenario | base Ir | PR Ir | Δ | |", "|---|---|---|---|---|"]
        for v in count_verdicts:
            if v.percent is None:
                lines.append(f"| `{v.name}` | – | – | {v.status} | ⚠️ |")
                continue
            mark = {"regression": "❌", "improvement": "🎉"}.get(v.status, "✅")
            lines.append(f"| `{v.name}` | {v.base:,} | {v.pr:,} | {v.percent:+.2f}% | {mark} |")
        lines.append("")

    if time_verdicts:
        lines += ["| scenario | base (ms) | PR (ms) | Δ | 95 % interval | |",
                  "|---|---|---|---|---|---|"]
        for v in time_verdicts:
            mark = {"regression": "❌", "improvement": "🎉", "noisy": "⚠️"}.get(v.status, "✅")
            lines.append(
                f"| `{v.name}` | {v.base_ms:.0f} | {v.pr_ms:.0f} | {v.ratio - 1:+.1%} | "
                f"{v.low - 1:+.1%} to {v.high - 1:+.1%} | {mark} |"
            )
        lines.append("")

    lines += [f"- {note}" for note in notes]

    return "\n".join(lines) + "\n"
