#!/usr/bin/env python3
"""Summarise one CI job's build: `.ninja_log` timings and ccache statistics.

Runs inside the CI job (Python standard library only) and writes, into --out:

  build-analysis.json  the summary (what the collector later turns into a
                       `kind: "build-analysis"` record, see SCHEMA.md)
  ninja_log.txt        the raw `.ninja_log`, if there was one
  ccache-stats.txt     the raw `ccache --print-stats` output, if available

Every part is optional and this script never fails a job because of missing
inputs: it exits 0 and records what it could not read.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1
SLOWEST = 20
SUPPORTED_LOG_VERSIONS = (5, 6, 7)
SUMMARY_NAME = "build-analysis.json"

COMPILE_SUFFIXES = (".o", ".obj", ".gch", ".pch")
SHARED_LIBRARY = re.compile(r"\.(so(\.\d+)*|dylib|dll)$")
STEP_COUNT = re.compile(r"^\[\d+/(\d+)\]")


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


def summarise_ninja(text, pending_steps):
    steps, version, ignored = parse_ninja_log(text)
    if version not in SUPPORTED_LOG_VERSIONS or not steps:
        return {
            "log_version": version,
            "complete": None,
            "pending_steps": pending_steps,
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
        "complete": None if pending_steps is None else pending_steps == 0,
        "pending_steps": pending_steps,
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


def run(command):
    """stdout of a command, or None if it is missing or fails."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def pending_ninja_steps(build_dir):
    """Steps `ninja -n` would still run: 0 means the build is complete.

    After `ninja -k 0` with failures, the failed targets and everything that
    depends on them are still pending.
    """
    output = run(["ninja", "-C", str(build_dir), "-n"])
    if output is None:
        return None
    if "no work to do" in output:
        return 0
    for line in output.splitlines():
        match = STEP_COUNT.match(line)
        if match:
            return int(match.group(1))
    return None


def build_summary(args, env):
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "build-analysis",
        "run_id": int(env["GITHUB_RUN_ID"]) if env.get("GITHUB_RUN_ID") else None,
        "run_attempt": int(env["GITHUB_RUN_ATTEMPT"]) if env.get("GITHUB_RUN_ATTEMPT") else None,
        "job_id": args.job_id,
        "job_key": env.get("GITHUB_JOB"),
        "artifact": args.name,
        "ninja": None,
        "ccache": None,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ninja_log = Path(args.build_dir) / ".ninja_log"
    if ninja_log.is_file():
        text = ninja_log.read_text(encoding="utf-8", errors="replace")
        shutil.copyfile(ninja_log, out / "ninja_log.txt")
        summary["ninja"] = summarise_ninja(text, pending_ninja_steps(args.build_dir))

    if args.ccache:
        stats = run(["ccache", "--print-stats"])
        if stats is not None:
            (out / "ccache-stats.txt").write_text(stats, encoding="utf-8")
            summary["ccache"] = parse_ccache_stats(stats)
    return summary


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Summarise a job's build timings.")
    parser.add_argument("--out", required=True, help="directory for the summary and raw files")
    parser.add_argument("--name", required=True, help="artifact name this summary is uploaded under")
    parser.add_argument("--build-dir", default="build", help="directory holding .ninja_log (default: build)")
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
        f"ccache hit rate={ccache.get('hit_rate', '-')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
