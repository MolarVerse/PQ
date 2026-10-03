#!/usr/bin/env python3
"""Summarise one CI job's build: `.ninja_log` timings and ccache statistics.

Runs inside the CI job (Python standard library only) and writes, into --out:

  build-analysis.json  the summary (what the collector later turns into a
                       `kind: "build-analysis"` record, see SCHEMA.md)
  ninja_log.txt        the raw `.ninja_log`, if there was one
  ninja-includes.json  the fan-in of every project header (kind "ninja-includes-detail"),
                       from `ninja -t deps`; used to compare two builds (the clang comment)
  ccache-stats.txt     the raw `ccache --print-stats` output, if available

Every part is optional and this script never fails a job because of missing
inputs: it exits 0 and records what it could not read.
"""

import argparse
import json
import hashlib
import os
import platform
import posixpath
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1
SLOWEST = 20
SUPPORTED_LOG_VERSIONS = (5, 6, 7)
SUMMARY_NAME = "build-analysis.json"
BUILD_STATUSES = ("success", "failure", "cancelled", "skipped")
INCLUDES_NAME = "ninja-includes.json"
TOP_HEADERS = 30
MAX_FAN_IN_ENTRIES = 5000
OBJECT_SUFFIXES = (".o", ".obj")
SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx")

COMPILE_SUFFIXES = (".o", ".obj", ".gch", ".pch")
SHARED_LIBRARY = re.compile(r"\.(so(\.\d+)*|dylib|dll)$")


def build_succeeded(status):
    """True/False from the outcome of the job's build step, None if unknown."""
    if status == "success":
        return True
    if status in ("failure", "cancelled"):
        return False
    return None


def classify(target):
    """compile, archive, link or other, from the output file name."""
    name = Path(target).name
    if name.endswith(COMPILE_SUFFIXES):
        return "compile"
    if name.endswith(".a"):
        return "archive"
    if SHARED_LIBRARY.search(name) or "." not in name or name.endswith(".exe"):
        return "link"
    return "other"


def parse_ninja_log(text):
    """Return (steps, version, ignored_lines); steps are dicts with ms times.

    One step per distinct command run: outputs of one rule appear on several
    lines with identical times and hash, and a rebuilt output appears again
    later in the log (the last entry wins).
    """
    lines = text.splitlines()
    version = None
    if lines and lines[0].startswith("# ninja log v"):
        try:
            version = int(lines[0].rsplit("v", 1)[1])
        except ValueError:
            version = None
    latest = {}
    ignored = 0
    for line in lines[1:] if version is not None else lines:
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        try:
            start, end, _mtime, output, command_hash = (
                int(fields[0]),
                int(fields[1]),
                fields[2],
                fields[3],
                fields[4],
            )
        except (ValueError, IndexError):
            ignored += 1
            continue
        if end < start:
            ignored += 1
            continue
        latest[output] = (start, end, command_hash)
    steps = {}
    for output, (start, end, command_hash) in latest.items():
        key = (start, end, command_hash)
        if key not in steps:
            steps[key] = {"target": output, "start": start, "end": end}
    ordered = sorted(steps.values(), key=lambda step: (step["start"], step["target"]))
    return ordered, version, ignored


def seconds(milliseconds):
    return round(milliseconds / 1000, 1)


def summarise_ninja(text, build_ok):
    steps, version, ignored = parse_ninja_log(text)
    if version not in SUPPORTED_LOG_VERSIONS or not steps:
        return {
            "log_version": version,
            "complete": None,
            "steps": len(steps),
            "error": "no usable .ninja_log (unsupported version or no steps)",
        }
    for step in steps:
        step["kind"] = classify(step["target"])
        step["ms"] = step["end"] - step["start"]
    first = min(step["start"] for step in steps)
    last = max(step["end"] for step in steps)
    wall = last - first
    cpu = sum(step["ms"] for step in steps)
    compile_ends = [step["end"] for step in steps if step["kind"] == "compile"]

    by_kind = {}
    for kind in ("compile", "archive", "link", "other"):
        members = [step for step in steps if step["kind"] == kind]
        by_kind[kind] = {"steps": len(members), "cpu_s": seconds(sum(s["ms"] for s in members))}

    slowest = sorted(steps, key=lambda step: (-step["ms"], step["target"]))[:SLOWEST]
    summary = {
        "log_version": version,
        "complete": build_ok,
        "steps": len(steps),
        "wall_s": seconds(wall),
        "cpu_s": seconds(cpu),
        "parallelism": round(cpu / wall, 2) if wall else None,
        "tail_after_compile_s": seconds(last - max(compile_ends)) if compile_ends else None,
        "by_kind": by_kind,
        "slowest": [
            {"target": step["target"], "kind": step["kind"], "seconds": seconds(step["ms"])}
            for step in slowest
        ],
    }
    if ignored:
        summary["ignored_lines"] = ignored
    return summary


def parse_ccache_stats(text):
    """Parse `ccache --print-stats` (tab separated `name value` lines).

    Returns None if no counter could be read. All non-zero integer counters are
    kept as they are, because the uncacheable reasons (precompiled header,
    unsupported option, ...) are exactly what the data is for.
    """
    counters = {}
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or "timestamp" in fields[0]:
            continue
        try:
            value = int(fields[1])
        except ValueError:
            continue
        if value:
            counters[fields[0]] = value
    if not any(name in text for name in ("cache_miss", "direct_cache_hit", "preprocessed_cache_hit")):
        return None
    hits = counters.get("direct_cache_hit", 0) + counters.get("preprocessed_cache_hit", 0)
    misses = counters.get("cache_miss", 0)
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / (hits + misses), 4) if hits + misses else None,
        "counters": dict(sorted(counters.items())),
    }


def read_runner(cpuinfo="/proc/cpuinfo", machine=None, cores=None):
    """CPU model and core count of this machine, or None.

    Only x86 is recorded: the `model name` line of /proc/cpuinfo is how the runner
    hardware generations (which differ in speed by up to 1.8x) tell themselves apart.
    """
    machine = platform.machine() if machine is None else machine
    if machine not in ("x86_64", "AMD64"):
        return None
    try:
        text = Path(cpuinfo).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        name, _, value = line.partition(":")
        model = " ".join(value.split())
        if name.strip() == "model name" and model:
            if cores is None:
                cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
            return {"cpu_model": model, "cores": cores or 1}
    return None


def run(command):
    """The finished process of a command, or None if it cannot be run."""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None


def parse_ninja_deps(text):
    """{target: [dependency, ...]} from the output of `ninja -t deps`.

    Each target line (`obj.o: #deps 12, deps mtime 123 (VALID)`) is followed by
    its indented dependencies.
    """
    targets = {}
    current = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():
            current = line.split(": #deps", 1)[0] if ": #deps" in line else None
            if current is not None:
                targets[current] = []
        elif current is not None:
            targets[current].append(line.strip())
    return targets


def normalise_dependency(path, build_dir, source_root):
    """(path, is_project_file): relative to the source root if it is a project file.

    Relative dependencies are relative to the build directory. A file inside the
    build directory (generated headers) is not a project file.
    """
    raw = path if posixpath.isabs(path) else posixpath.join(build_dir, path)
    path = posixpath.normpath(raw)
    in_build = path == build_dir or path.startswith(build_dir + "/")
    if not in_build and path.startswith(source_root + "/"):
        return path[len(source_root) + 1 :], True
    if in_build and path.startswith(source_root + "/"):
        return path[len(source_root) + 1 :], False
    return path, False


def include_graph(targets, build_dir, source_root):
    """(summary, {project file: fan-in}) from parsed `ninja -t deps` output.

    Only object files count as translation units, and a translation unit's own
    source file is not one of its dependencies here. The fan-in of a file is the
    number of translation units that depend on it (for example include it);
    unlike clang's trace it does not depend on how long anything took, so two
    builds of the same code give the same numbers.
    """
    build_dir = posixpath.normpath(str(build_dir))
    source_root = posixpath.normpath(str(source_root))
    pairs = set()
    project = set()
    objects = 0
    for target, dependencies in targets.items():
        if not target.endswith(OBJECT_SUFFIXES):
            continue
        objects += 1
        name = posixpath.normpath(target)
        for dependency in dependencies:
            if dependency.endswith(SOURCE_SUFFIXES):
                continue
            path, is_project = normalise_dependency(dependency, build_dir, source_root)
            pairs.add((name, path))
            if is_project:
                project.add(path)
    fan_in = {}
    project_pairs = 0
    for _, path in pairs:
        if path in project:
            fan_in[path] = fan_in.get(path, 0) + 1
            project_pairs += 1
    digest = hashlib.sha256("\n".join(f"{t}\t{d}" for t, d in sorted(pairs)).encode()).hexdigest()
    top = sorted(fan_in.items(), key=lambda item: (-item[1], item[0]))[:TOP_HEADERS]
    summary = {
        "objects": objects,
        "unique_files": len({path for _, path in pairs}),
        "include_pairs": len(pairs),
        "project_files": len(fan_in),
        "project_pairs": project_pairs,
        "digest": digest,
        "top_project_files": [{"file": file, "fan_in": count} for file, count in top],
    }
    return summary, fan_in


def read_includes(build_dir, source_root):
    """(summary, fan-in map) from `ninja -t deps`, or (None, None) if it cannot be read."""
    result = run(["ninja", "-C", str(build_dir), "-t", "deps"])
    if result is None or result.returncode != 0:
        return None, None
    targets = parse_ninja_deps(result.stdout)
    if not any(target.endswith(OBJECT_SUFFIXES) for target in targets):
        return None, None
    return include_graph(targets, Path(build_dir).resolve().as_posix(), Path(source_root).resolve().as_posix())


def build_summary(args, env):
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "build-analysis",
        "run_id": int(env["GITHUB_RUN_ID"]) if env.get("GITHUB_RUN_ID") else None,
        "run_attempt": int(env["GITHUB_RUN_ATTEMPT"]) if env.get("GITHUB_RUN_ATTEMPT") else None,
        "job_id": args.job_id,
        "job_key": env.get("GITHUB_JOB"),
        "artifact": args.name,
        "runner": read_runner(),
        "ninja": None,
        "ccache": None,
        "includes": None,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ninja_log = Path(args.build_dir) / ".ninja_log"
    if ninja_log.is_file():
        text = ninja_log.read_text(encoding="utf-8", errors="replace")
        shutil.copyfile(ninja_log, out / "ninja_log.txt")
        summary["ninja"] = summarise_ninja(text, build_succeeded(args.build_status))
        includes, fan_in = read_includes(args.build_dir, args.source_root)
        if includes is not None:
            summary["includes"] = includes
            detail = {
                "schema_version": SCHEMA_VERSION,
                "kind": "ninja-includes-detail",
                "include_pairs": includes["include_pairs"],
                "project_pairs": includes["project_pairs"],
                "digest": includes["digest"],
                "fan_in": dict(sorted(fan_in.items(), key=lambda item: (-item[1], item[0]))[:MAX_FAN_IN_ENTRIES]),
            }
            (out / INCLUDES_NAME).write_text(json.dumps(detail, separators=(",", ":")) + "\n", encoding="utf-8")

    if args.ccache:
        result = run(["ccache", "--print-stats"])
        if result is not None and result.returncode == 0:
            (out / "ccache-stats.txt").write_text(result.stdout, encoding="utf-8")
            summary["ccache"] = parse_ccache_stats(result.stdout)
    return summary


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Summarise a job's build timings.")
    parser.add_argument("--out", required=True, help="directory for the summary and raw files")
    parser.add_argument("--name", required=True, help="artifact name this summary is uploaded under")
    parser.add_argument("--build-dir", default="build", help="directory holding .ninja_log (default: build)")
    parser.add_argument(
        "--build-status",
        choices=BUILD_STATUSES,
        default=None,
        help="outcome of the job's build step; decides `ninja.complete` (default: unknown)",
    )
    parser.add_argument("--source-root", default=".", help="repository root, to tell project files from system headers")
    parser.add_argument("--ccache", action="store_true", help="also record `ccache --print-stats`")
    parser.add_argument("--job-id", type=int, default=None, help="REST API id of this job (job.check_run_id)")
    return parser.parse_args(argv)


def main(argv=None, env=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    env = os.environ if env is None else env
    summary = build_summary(args, env)
    path = Path(args.out) / SUMMARY_NAME
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    ninja = summary["ninja"] or {}
    ccache = summary["ccache"] or {}
    print(
        f"wrote {path}: ninja steps={ninja.get('steps', '-')} complete={ninja.get('complete', '-')}, "
        f"ccache hit rate={ccache.get('hit_rate', '-')}, "
        f"cpu={(summary['runner'] or {}).get('cpu_model', '-')}, "
        f"include pairs={(summary['includes'] or {}).get('include_pairs', '-')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
