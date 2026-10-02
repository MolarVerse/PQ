#!/usr/bin/env python3
"""Summarise clang `-ftime-trace` output of a build (standard library only).

clang writes one Chrome-trace JSON file next to every object file
(`foo.cpp.o` -> `foo.cpp.json`). This script reads them all and writes, into
--out:

  clang-traces.json   the summary (see SCHEMA.md "Clang trace summary")
  traces/             the raw traces of the slowest files, flattened names

The summary has the totals (compiler, frontend, backend), the slowest files,
the heaviest headers by inclusion time, the most expensive template
instantiations and two counts that depend little on runner noise. Times are
seconds. It never fails a job: unreadable traces are counted and skipped.

Time notes. Events shorter than clang's trace granularity (0.5 ms by default)
are not in the trace, so the counts are of events above that size. "Inclusive"
time contains the time of nested events (an included header's own includes),
"self" time does not; header and template rankings use self time so that a
header is not blamed for what it includes.
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA_VERSION = 1
SUMMARY_NAME = "clang-traces.json"
TRACE_NAME = re.compile(r"\.(cpp|cc|cxx|c)\.json$")

TOP_FILES = 20
TOP_HEADERS = 30
TOP_TEMPLATES = 30
RAW_TRACES = 20
MAX_NAME_CHARS = 200

SOURCE = "Source"
TEMPLATE_EVENTS = ("InstantiateClass", "InstantiateFunction")


def load_trace(path):
    """The complete events (`ph` == "X") of one trace file; ValueError if unusable."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error
    events = data.get("traceEvents") if isinstance(data, dict) else None
    if not isinstance(events, list):
        raise ValueError(f"{path}: no traceEvents")
    return [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("ph") == "X"
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
        and isinstance(event.get("name"), str)
    ]


def detail(event):
    args = event.get("args")
    value = args.get("detail") if isinstance(args, dict) else None
    return value if isinstance(value, str) else ""


def with_self_times(events):
    """[(event, depth, self_us)] for events that nest by time (one family).

    An event is the child of the innermost open event of the list that contains
    it; its duration is taken out of the parent's self time. Timestamps are
    whole microseconds, so a child may end a microsecond after its parent.
    """
    ordered = sorted(events, key=lambda e: (e.get("tid", 0), e["ts"], -e["dur"]))
    result = []
    stack = []  # [end, tid, index into result]
    for event in ordered:
        start, end, tid = event["ts"], event["ts"] + event["dur"], event.get("tid", 0)
        while stack and (stack[-1][1] != tid or stack[-1][0] <= start):
            stack.pop()
        depth = len(stack)
        self_us = float(event["dur"])
        if stack:
            parent = result[stack[-1][2]]
            parent[2] -= min(event["dur"], stack[-1][0] - start)
        result.append([event, depth, self_us])
        stack.append((end, tid, len(result) - 1))
    return [(event, depth, max(self_us, 0.0)) for event, depth, self_us in result]


def analyse_trace(events):
    """Figures of one translation unit."""
    named = defaultdict(list)
    for event in events:
        named[event["name"]].append(event)

    def total(name):
        return sum(e["dur"] for e in named.get(name, [])) / 1e6 or None

    sources = with_self_times(named.get(SOURCE, []))
    top_level = [item for item in sources if item[1] == 0]
    main = max(top_level, key=lambda item: item[0]["dur"], default=None)
    headers = [
        (detail(event), event["dur"] / 1e6, self_us / 1e6)
        for event, depth, self_us in sources
        if detail(event) and (main is None or event is not main[0])
    ]
    templates = [
        (detail(event), event["dur"] / 1e6, self_us / 1e6)
        for event, _, self_us in with_self_times(
            [e for name in TEMPLATE_EVENTS for e in named.get(name, [])]
        )
        if detail(event)
    ]
    frontend, backend = total("Frontend"), total("Backend")
    compiler = total("ExecuteCompiler")
    if compiler is None and (frontend or backend):
        compiler = (frontend or 0) + (backend or 0)
    return {
        "main": detail(main[0]) if main else "",
        "total_s": compiler,
        "frontend_s": frontend,
        "backend_s": backend,
        "headers": headers,
        "templates": templates,
        "source_events": len(sources),
        "instantiation_events": len(templates),
    }


def clean_path(path, roots):
    """Path relative to the first matching root, or the path itself."""
    for root in roots:
        prefix = root.rstrip("/") + "/"
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def clip(text, limit=MAX_NAME_CHARS):
    text = "".join(char if char.isprintable() else " " for char in text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def find_traces(build_dir):
    return sorted(
        path
        for path in Path(build_dir).rglob("*.json")
        if TRACE_NAME.search(path.name) and path.is_file()
    )


def round_s(value):
    return None if value is None else round(value, 3)


def summarise(build_dir, source_root):
    """(summary dict, [(total seconds, trace path)] of the readable traces)."""
    roots = [str(Path(source_root).resolve()), str(Path(build_dir).resolve())]
    files, unreadable = [], 0
    header_totals = defaultdict(lambda: [0.0, 0.0, 0, set()])  # inclusive, self, events, files
    template_totals = defaultdict(lambda: [0.0, 0.0, 0])
    sums = {"total_s": 0.0, "frontend_s": 0.0, "backend_s": 0.0}
    source_events = instantiation_events = 0
    heaviest = []

    for path in find_traces(build_dir):
        try:
            result = analyse_trace(load_trace(path))
        except ValueError:
            unreadable += 1
            continue
        if result["total_s"] is None:
            unreadable += 1
            continue
        name = clean_path(result["main"], roots) or clean_path(str(path.resolve().with_suffix("")), [roots[1]])
        for key in sums:
            sums[key] += result[key] or 0.0
        source_events += result["source_events"]
        instantiation_events += result["instantiation_events"]
        for header, inclusive, own in result["headers"]:
            entry = header_totals[clean_path(header, roots)]
            entry[0] += inclusive
            entry[1] += own
            entry[2] += 1
            entry[3].add(name)
        for template, inclusive, own in result["templates"]:
            entry = template_totals[template]
            entry[0] += inclusive
            entry[1] += own
            entry[2] += 1
        files.append((result["total_s"], name, result["frontend_s"], result["backend_s"]))
        heaviest.append((result["total_s"], path))

    files.sort(key=lambda item: (-item[0], item[1]))
    headers = sorted(header_totals.items(), key=lambda item: (-item[1][1], item[0]))[:TOP_HEADERS]
    templates = sorted(template_totals.items(), key=lambda item: (-item[1][1], item[0]))[:TOP_TEMPLATES]
    count = len(files)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "clang-trace-summary",
        "files": count,
        "unreadable": unreadable,
        "total_s": round_s(sums["total_s"]),
        "frontend_s": round_s(sums["frontend_s"]),
        "backend_s": round_s(sums["backend_s"]),
        "source_events": source_events,
        "instantiation_events": instantiation_events,
        "slowest_files": [
            {"file": clip(name), "total_s": round_s(total), "frontend_s": round_s(front), "backend_s": round_s(back)}
            for total, name, front, back in files[:TOP_FILES]
        ],
        "headers": [
            {
                "header": clip(header),
                "inclusive_s": round_s(inclusive),
                "self_s": round_s(own),
                "events": events,
                "files": len(including),
            }
            for header, (inclusive, own, events, including) in headers
        ],
        "templates": [
            {"name": clip(template), "count": events, "inclusive_s": round_s(inclusive), "self_s": round_s(own)}
            for template, (inclusive, own, events) in templates
        ],
    }
    return summary, sorted(heaviest, key=lambda item: (-item[0], str(item[1])))


def copy_raw_traces(heaviest, build_dir, out):
    target = Path(out) / "traces"
    target.mkdir(parents=True, exist_ok=True)
    base = Path(build_dir).resolve()
    for _, path in heaviest[:RAW_TRACES]:
        try:
            relative = str(path.resolve().relative_to(base))
        except ValueError:
            relative = path.name
        shutil.copyfile(path, target / relative.replace(os.sep, "__"))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Summarise clang -ftime-trace output.")
    parser.add_argument("--build-dir", default="build", help="build directory with the *.cpp.json traces")
    parser.add_argument("--source-root", default=".", help="repository root, stripped from paths")
    parser.add_argument("--out", required=True, help="directory for the summary and the raw traces")
    parser.add_argument("--no-raw", action="store_true", help="do not copy the heaviest raw traces")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, heaviest = summarise(args.build_dir, args.source_root)
    (out / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.no_raw and heaviest:
        copy_raw_traces(heaviest, args.build_dir, out)
    print(
        f"wrote {out / SUMMARY_NAME}: {summary['files']} traces ({summary['unreadable']} unreadable), "
        f"total {summary['total_s']} s (frontend {summary['frontend_s']} s, backend {summary['backend_s']} s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
