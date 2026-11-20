#!/usr/bin/env python3
"""Check the performance of your working tree against a base, locally.

Builds the fixed-work benchmarks (benchmarks/perf) of your working tree
(uncommitted changes included) and of a base commit, and runs the two tools of
the CI performance gate on them:

* scripts/perf_gate.sh    instruction counts (callgrind), heap allocations;
                          exact and independent of the load of the machine
* scripts/perf_timing.py  CPU time of the measured loop, paired and
                          interleaved, with a confidence interval

Usage (from anywhere in the repository):

    scripts/perf_local.py                       # everything, base = merge-base with dev
    scripts/perf_local.py --only interWater     # only benchmarks whose name contains it
    scripts/perf_local.py --counts-only         # no timing: the fast check
    scripts/perf_local.py --base HEAD           # what did my uncommitted change do?
    scripts/perf_local.py --list                # the benchmarks and scenarios
    scripts/perf_local.py --suite scenarios     # whole PQ runs of examples and integration inputs
    scripts/perf_local.py --suite all           # benchmarks and scenarios

Two suites: the fixed-work benchmarks of benchmarks/perf (one part of the code
each, seconds) and the end-to-end scenarios of benchmarks/scenarios.json (the
real PQ on inputs of the examples and the integration tests, about a minute
each for the timing; see scripts/perf_scenarios.py).

Everything lives in build-perf-local/ (ignored by git). The base is exported
from git (never your working tree) and built once per commit; its instruction
counts are cached, so the next run only measures your side. Delete the
directory, or pass --rebuild-base, to start over.

Like CI the build is portable (-march is not native, so counts do not depend on
the CPU). --native builds what a user would run; compare native with native.

Timing needs a quiet machine: close other programs, and prefer a desktop that
does not throttle. Instruction counts do not care. The tool picks the CPU with
the least load whose hyper-thread sibling is idle, and warns about conditions
that make timings noisy.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_scenarios  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "build-perf-local"
PERF_CMAKE = Path("benchmarks") / "perf" / "CMakeLists.txt"
HELPER_TARGETS = ["perf_alloc_counter", "perf_timer"]
HELPER_LIBRARIES = ["libperf_alloc_counter.so", "libperf_timer.so"]


class Failure(Exception):
    """An expected problem; reported without a traceback."""


# ---------------------------------------------------------------- benchmarks


def benchmark_names(cmake_text):
    """Names (without the perf_ prefix) of the benchmarks of a CMakeLists.txt.

    The CMake file lists the sources (perfInterWater.cpp) and derives the
    target (perf_interWater) by lower-casing the first letter.
    """
    block = re.search(r"set\(perf_benchmarks(.*?)\)", cmake_text, re.S)
    if not block:
        return []

    names = []
    for stem in re.findall(r"perf([A-Za-z0-9]+)\.cpp", block.group(1)):
        names.append(stem[0].lower() + stem[1:])

    return names


def select(names, patterns):
    """The names that contain any pattern, case-insensitive; all if none."""
    if not patterns:
        return list(names)

    lowered = [p.lower() for p in patterns]
    return [n for n in names if any(p in n.lower() for p in lowered)]


# ---------------------------------------------------------------------- CPU


def parse_cpu_list(text):
    """'2,18' or '4-7,12' to a sorted list of CPU numbers."""
    cpus = set()
    for part in text.strip().split(","):
        if "-" in part:
            low, high = part.split("-")
            cpus.update(range(int(low), int(high) + 1))
        elif part:
            cpus.add(int(part))

    return sorted(cpus)


def parse_proc_stat(text):
    """cpu number -> (busy jiffies, total jiffies) from /proc/stat."""
    cpus = {}
    for line in text.splitlines():
        match = re.match(r"cpu(\d+)\s+(.*)", line)
        if not match:
            continue

        values = [int(v) for v in match.group(2).split()]
        total = sum(values[:8])
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        cpus[int(match.group(1))] = (total - idle, total)

    return cpus


def busy_fractions(before, after):
    result = {}
    for cpu, (busy, total) in after.items():
        if cpu not in before:
            continue

        d_busy = busy - before[cpu][0]
        d_total = total - before[cpu][1]
        result[cpu] = d_busy / d_total if d_total > 0 else 0.0

    return result


def choose_cpu(allowed, siblings, busy):
    """The CPU to pin the timing to.

    Hyper-thread siblings share the caches and execution units of a core, so a
    busy sibling slows a measurement down: the score of a CPU is the larger of
    its own load and the load of its siblings. Ties go to the highest number.
    """

    def score(cpu):
        others = [busy.get(s, 0.0) for s in siblings.get(cpu, []) if s != cpu]
        return max([busy.get(cpu, 0.0)] + others)

    return min(sorted(allowed, reverse=True), key=score)


def sample_cpu():
    allowed = sorted(os.sched_getaffinity(0))
    before = parse_proc_stat(Path("/proc/stat").read_text())
    time.sleep(1.0)
    after = parse_proc_stat(Path("/proc/stat").read_text())

    siblings = {}
    for cpu in allowed:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            siblings[cpu] = parse_cpu_list(path.read_text())
        except OSError:
            siblings[cpu] = [cpu]

    busy = busy_fractions(before, after)
    cpu = choose_cpu(allowed, siblings, busy)
    sibling_busy = max(
        [busy.get(s, 0.0) for s in siblings.get(cpu, []) if s != cpu] or [0.0]
    )

    others = sum(v for c, v in busy.items() if c != cpu)

    return cpu, busy.get(cpu, 0.0), sibling_busy, others


def environment_warnings(others_busy, governor, cpu_busy, sibling_busy):
    """Why timings from this machine may be noisy (counts are not affected).

    others_busy is the number of CPUs' worth of work that the rest of the
    machine did during the last second.
    """
    warnings = []

    if others_busy > 1.0:
        warnings.append(
            f"other programs keep about {others_busy:.1f} CPUs busy"
        )
    if governor not in (None, "performance"):
        warnings.append(
            f"CPU governor is '{governor}', not 'performance': the clock "
            "changes during a run"
        )
    if cpu_busy > 0.2 or sibling_busy > 0.2:
        warnings.append(
            f"the chosen CPU is {cpu_busy:.0%} busy and its sibling "
            f"{sibling_busy:.0%}"
        )

    return warnings


# ---------------------------------------------------------------------- git


def git(repo, *arguments, check=True):
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise Failure(
            f"git {' '.join(arguments)} failed in {repo}: "
            f"{result.stderr.strip()}"
        )

    return result.stdout.strip()


def resolve_base(repo, ref):
    """The commit to compare with: the ref, or the merge-base with dev."""
    if ref:
        return git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")

    for candidate in ("origin/dev", "dev"):
        if git(repo, "rev-parse", "--verify", "--quiet", candidate, check=False):
            return git(repo, "merge-base", "HEAD", candidate)

    raise Failure("no base: there is no dev branch, pass --base REF")


def export_tree(repo, commit, destination):
    """The files of a commit and of its submodules (recursively), no .git.

    Uses the submodule checkouts of the working repository, so nothing is
    downloaded; a submodule commit that is not there has to be fetched first.
    """
    destination.mkdir(parents=True, exist_ok=True)

    archive = subprocess.Popen(
        ["git", "-C", str(repo), "archive", commit], stdout=subprocess.PIPE
    )
    unpack = subprocess.run(
        ["tar", "-x", "-C", str(destination)], stdin=archive.stdout, check=False
    )
    archive.stdout.close()
    if archive.wait() != 0 or unpack.returncode != 0:
        raise Failure(f"could not export {commit} of {repo}")

    listing = git(repo, "ls-tree", "-r", commit)
    for line in listing.splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, sha = meta.split()
        if mode != "160000":
            continue

        sub = repo / path
        if not (sub / ".git").exists():
            raise Failure(
                f"submodule {path} is not checked out in {repo}: run "
                "'git submodule update --init --recursive'"
            )
        if not git(sub, "cat-file", "-t", sha, check=False):
            raise Failure(
                f"submodule {path} of the base has commit {sha[:12]}, which "
                f"this checkout does not have: 'git -C {sub} fetch'"
            )

        export_tree(sub, sha, destination / path)


# ------------------------------------------------------------------- builds


def run(command, **options):
    print("+ " + " ".join(str(c) for c in command), flush=True)
    result = subprocess.run([str(c) for c in command], check=False, **options)
    return result.returncode


def configure_and_build(source, build, targets, native, jobs, eigen):
    generator = ["-G", "Ninja"] if shutil.which("ninja") else []
    arguments = [
        "cmake", "-S", source, "-B", build, *generator,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DBUILD_WITH_NATIVE={'On' if native else 'Off'}",
        "-DBUILD_WITH_PERF_BENCH=On", "-DBUILD_WITH_ASE=Off",
        "-DBUILD_WITH_TESTS=Off", "-DBUILD_WITH_DOCS=Off",
    ]  # fmt: skip
    if eigen:
        arguments.append(f"-DFETCHCONTENT_SOURCE_DIR_EIGEN={eigen}")

    if run(arguments) != 0:
        raise Failure(f"cmake could not configure {source}")

    build_command = ["cmake", "--build", build, "--target", *targets, "-j", jobs]
    if run(build_command) != 0:
        raise Failure(f"the build of {source} failed")


def reusable_eigen(first_tree, first_build, second_tree):
    """Eigen of an existing build, if both trees pin the same version."""
    eigen = Path(first_build) / "_deps" / "eigen-src"
    pin = Path(".cmake") / "eigen.cmake"

    try:
        same = (first_tree / pin).read_text() == (second_tree / pin).read_text()
    except OSError:
        return None

    return eigen if same and eigen.is_dir() else None


def stage(directory, build, names):
    """A directory with links to the chosen binaries and the helper libraries."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)

    perf = Path(build) / "benchmarks" / "perf"
    linked = []
    for name in names:
        binary = perf / f"perf_{name}"
        if binary.exists():
            (directory / binary.name).symlink_to(binary)
            linked.append(name)

    for library in HELPER_LIBRARIES:
        if (perf / library).exists():
            (directory / library).symlink_to(perf / library)

    return linked


# --------------------------------------------------------------------- main


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Check the performance of the working tree against a base.",
        epilog="See the top of this file for the details.",
    )
    parser.add_argument("--base", metavar="REF",
                        help="commit to compare with (default: merge-base "
                        "with origin/dev)")
    parser.add_argument("--suite", choices=["benchmarks", "scenarios", "all"],
                        default="benchmarks",
                        help="fixed-work benchmarks (default), end-to-end "
                        "scenarios of examples and integration inputs, or both")
    parser.add_argument("--only", nargs="+", metavar="NAME", default=[],
                        help="only benchmarks and scenarios whose name "
                        "contains one of these")
    parser.add_argument("--list", action="store_true",
                        help="list the benchmarks and scenarios and exit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--counts-only", action="store_true",
                      help="instruction counts only (fast)")
    mode.add_argument("--timing-only", action="store_true",
                      help="timing only")
    parser.add_argument("--native", action="store_true",
                        help="build with -march=native (as a user would)")
    parser.add_argument("--sim", action="store_true",
                        help="also simulate caches and branch prediction "
                        "(about 3 times slower)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="count threshold in percent (default 1, as CI)")
    parser.add_argument("--timing-threshold", type=float, default=10.0,
                        help="timing threshold in percent (default 10, as CI)")
    parser.add_argument("--rounds", type=int, default=15,
                        help="timing rounds per benchmark (default 15)")
    parser.add_argument("--scenario-rounds", type=int, default=7,
                        help="timing rounds per scenario (default 7)")
    parser.add_argument("--scenario-copies", type=int, default=2,
                        help="copies of the PQ binary per round and side "
                        "(default 2, at least 2)")
    parser.add_argument("--jobs", default=str(os.cpu_count() or 4),
                        help="parallel build jobs")
    parser.add_argument("--rebuild-base", action="store_true",
                        help="rebuild the base and forget its cached counts")
    parser.add_argument("--report-only", action="store_true",
                        help="exit 0 even if the counts show a regression")
    options = parser.parse_args(argv)
    if options.scenario_copies < 2:
        parser.error("--scenario-copies must be at least 2")
    return options


def check_platform(options):
    if sys.platform != "linux":
        raise Failure("only Linux is supported (callgrind and LD_PRELOAD)")

    if not options.timing_only and not shutil.which("valgrind"):
        raise Failure(
            "valgrind is needed for the instruction counts: "
            "sudo apt install valgrind (or use --timing-only)"
        )

    for tool in ("cmake", "git", "tar"):
        if not shutil.which(tool):
            raise Failure(f"{tool} is not installed")


def compiler_version():
    return subprocess.run(
        [os.environ.get("CXX", "c++"), "--version"],
        capture_output=True, text=True, check=False,
    ).stdout  # fmt: skip


def cache_file(commit, options, selected):
    """Cached base counts: valid for one commit, flags, selection, compiler."""
    key = "|".join([
        commit, str(options.native), str(options.sim),
        ",".join(sorted(selected)), compiler_version(),
    ])  # fmt: skip
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return WORK / f"base-counts-{commit[:12]}-{digest}.tsv"


def scenario_cache_file(commit, options, scenarios):
    """Cached base counts of scenarios; changes with their definition."""
    key = "|".join([
        commit, str(options.native), compiler_version(),
        json.dumps([vars(s) for s in scenarios], sort_keys=True),
    ])  # fmt: skip
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return WORK / f"scenario-counts-{commit[:12]}-{digest}.json"


def pin_to_quiet_cpu():
    """The CPU to pin the timing to; prints what makes timings noisy."""
    cpu, cpu_busy, sibling_busy, others_busy = sample_cpu()
    governor_file = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
    governor = governor_file.read_text().strip() if governor_file.exists() else None
    for warning in environment_warnings(
        others_busy, governor, cpu_busy, sibling_busy
    ):
        print(f"warning: {warning}")
    print(f"pinned to CPU {cpu}\n", flush=True)

    return cpu


def run_benchmarks(options, selected, commit, pr_build, base_build, base_selected):
    """The gate and the timing of the fixed-work benchmarks; 1 on a regression."""
    pr_stage = stage(WORK / "stage-pr", pr_build, selected)
    stage(WORK / "stage-base", base_build, base_selected)
    if not pr_stage:
        raise Failure("no benchmark binary was built")

    status = 0

    if not options.timing_only:
        print("\n=== instruction counts " + "=" * 50, flush=True)
        environment = dict(
            os.environ,
            PERF_CACHE_SIM="1" if options.sim else "0",
            PERF_REPORT_FILE=str(WORK / "report-counts.md"),
            PERF_BASE_IR_FILE=str(cache_file(commit, options, selected)),
        )
        environment.pop("GITHUB_STEP_SUMMARY", None)
        code = run(
            ["bash", ROOT / "scripts" / "perf_gate.sh", WORK / "stage-pr",
             WORK / "stage-base", options.threshold],
            env=environment,
        )  # fmt: skip
        if code != 0:
            status = 1

    if not options.counts_only:
        print("\n=== timing " + "=" * 62, flush=True)
        cpu = pin_to_quiet_cpu()
        run(
            [sys.executable, ROOT / "scripts" / "perf_timing.py", WORK / "stage-pr",
             WORK / "stage-base", "--cpu", cpu, "--rounds", options.rounds,
             "--threshold", options.timing_threshold, "--report-only",
             "--append", WORK / "report-timing.md"],
            env=dict(os.environ, GITHUB_STEP_SUMMARY=""),
        )  # fmt: skip

    return status


def run_scenarios(options, chosen, commit, pr_build, base_build):
    """Counts and timing of whole PQ runs; 1 on a count regression."""
    pr_binary = pr_build / "apps" / "PQ"
    base_binary = base_build / "apps" / "PQ"
    for binary in (pr_binary, base_binary):
        if not binary.exists():
            raise Failure(f"{binary} was not built")

    count_verdicts, time_verdicts, notes = [], [], []
    status = 0

    if not options.timing_only:
        print("\n=== scenarios: instruction counts " + "=" * 39, flush=True)
        templates = {
            s.name: perf_scenarios.prepare(
                s, ROOT, WORK / "scenarios" / f"{s.name}-count", s.count_steps)
            for s in chosen
        }  # fmt: skip
        cache_path = scenario_cache_file(commit, options, chosen)
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        count_verdicts = perf_scenarios.count_scenarios(
            chosen, templates, pr_binary, base_binary,
            shutil.which("valgrind"), options.threshold, cache,
        )
        cache_path.write_text(json.dumps(cache, indent=1))
        if any(v.status == "regression" for v in count_verdicts):
            status = 1

    if not options.counts_only:
        print("\n=== scenarios: timing " + "=" * 51, flush=True)
        cpu = pin_to_quiet_cpu()
        templates = {
            s.name: perf_scenarios.prepare(
                s, ROOT, WORK / "scenarios" / f"{s.name}-time", s.steps)
            for s in chosen
        }  # fmt: skip
        time_verdicts, notes = perf_scenarios.time_scenarios(
            chosen, templates, pr_binary, base_binary, cpu,
            options.scenario_rounds, options.scenario_copies,
            options.timing_threshold / 100.0,
        )
        for note in notes:
            print(f"note: {note}")

    (WORK / "report-scenarios.md").write_text(
        perf_scenarios.render_report(count_verdicts, time_verdicts, notes)
    )

    return status


def main(argv=None):
    options = parse_arguments(sys.argv[1:] if argv is None else argv)
    check_platform(options)

    names = benchmark_names((ROOT / PERF_CMAKE).read_text())
    scenarios = perf_scenarios.load_scenarios(
        (ROOT / perf_scenarios.SCENARIOS_FILE).read_text()
    )
    scenario_names = [s.name for s in scenarios]

    if options.list:
        print("benchmarks (--suite benchmarks):")
        print("\n".join(f"  {n}" for n in names))
        print("scenarios (--suite scenarios):")
        print("\n".join(f"  {s.name}  {s.description}" for s in scenarios))
        return 0

    selected = (select(names, options.only)
                if options.suite in ("benchmarks", "all") else [])
    chosen_names = (select(scenario_names, options.only)
                    if options.suite in ("scenarios", "all") else [])
    chosen = [s for s in scenarios if s.name in chosen_names]

    if not selected and not chosen:
        hint = ""
        if options.suite == "benchmarks" and select(scenario_names, options.only):
            hint = " (it matches scenarios: add --suite scenarios)"
        elif options.suite == "scenarios" and select(names, options.only):
            hint = " (it matches benchmarks: use --suite benchmarks)"
        raise Failure(
            f"nothing contains {options.only}{hint}; see --list for the names"
        )

    commit = resolve_base(ROOT, options.base)
    short = commit[:12]
    flavour = "native" if options.native else "portable"
    print(f"base {short} ({git(ROOT, 'log', '-1', '--format=%s', commit)})")
    print(f"{len(selected)} benchmark(s), {len(chosen)} scenario(s), "
          f"{flavour} build\n")

    WORK.mkdir(exist_ok=True)
    for old in WORK.glob("report-*.md"):
        old.unlink()

    # your working tree, incrementally
    pr_build = WORK / f"pr-{flavour}"
    pr_targets = []
    if selected:
        pr_targets += (["perf_benchmarks"] if selected == names
                       else [f"perf_{n}" for n in selected] + HELPER_TARGETS)
    if chosen:
        pr_targets.append("PQ")
    configure_and_build(ROOT, pr_build, pr_targets, options.native,
                        options.jobs, os.environ.get("PQ_EIGEN_SRC"))

    # the base, exported from git, once per commit
    base_source = WORK / f"base-src-{short}"
    base_build = WORK / f"base-{short}-{flavour}"
    if options.rebuild_base:
        shutil.rmtree(base_source, ignore_errors=True)
        shutil.rmtree(base_build, ignore_errors=True)
        for pattern in (f"base-counts-{short}-*.tsv", f"scenario-counts-{short}-*.json"):
            for old in WORK.glob(pattern):
                old.unlink()
    if not base_source.exists():
        print(f"exporting {short} to {base_source}")
        try:
            export_tree(ROOT, commit, base_source)
        except Failure:
            shutil.rmtree(base_source, ignore_errors=True)
            raise

    base_names = benchmark_names((base_source / PERF_CMAKE).read_text())
    base_selected = [n for n in selected if n in base_names]
    new_only = [n for n in selected if n not in base_names]
    if new_only:
        print(f"not in the base (new, nothing to compare): {', '.join(new_only)}")

    base_targets = []
    if base_selected:
        base_targets += (["perf_benchmarks"] if base_selected == base_names
                         else [f"perf_{n}" for n in base_selected])
    if chosen:
        base_targets.append("PQ")
    if base_targets:
        eigen = os.environ.get("PQ_EIGEN_SRC") or reusable_eigen(
            ROOT, pr_build, base_source
        )
        configure_and_build(base_source, base_build, base_targets,
                            options.native, options.jobs, eigen)

    status = 0
    if selected:
        status |= run_benchmarks(options, selected, commit, pr_build,
                                 base_build, base_selected)
    if chosen:
        status |= run_scenarios(options, chosen, commit, pr_build, base_build)

    written = sorted(p for p in WORK.glob("report-*.md"))
    print(f"\nreports: {', '.join(str(p) for p in written)}")
    return 0 if options.report_only else status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Failure as error:
        print(f"perf_local: {error}", file=sys.stderr)
        sys.exit(2)
