"""Clang `-ftime-trace` data of a build: where the compiler spends its time, per file, header and template.

clang writes one Chrome-trace JSON file next to every object file (`foo.cpp.o` -> `foo.cpp.json`). This module reads
them (standard library only) and keeps what answers the questions the Ninja data cannot:

  * frontend (parsing, semantic analysis, template instantiation) versus backend (optimisation, code generation),
    per file and so per module,
  * the time spent in every header (self: without what it includes; inclusive: with it), summed over all files,
  * the most expensive template instantiations (self and inclusive time, and how often).

Format notes (checked on clang 20 traces). Most events are complete events (`ph` "X"). Include events (`Source`)
are async begin/end pairs (`ph` "b" and "e", id 0), written next to each other when the include finishes, so they
are paired by order. The main source file is not an event. Events shorter than clang's trace granularity (0.5 ms by
default) are not in the trace, so counts are counts of events above that size and, like the times, depend on how
fast the machine was. "Inclusive" time contains nested events, "self" time does not; rankings use self time so that
a header is not blamed for what it includes.
"""

import json
import posixpath
import re
from collections import defaultdict
from pathlib import Path

SOURCE = "Source"
TEMPLATE_EVENTS = ("InstantiateClass", "InstantiateFunction")
TOP_HEADERS = 300
TOP_TEMPLATES = 200
MAX_NAME_CHARS = 200
MAX_TRACE_BYTES = 200_000_000
SYSTEM_PREFIXES = (
    (re.compile(r"^/usr/include/c\+\+/\d+/"), "<std>/"),
    (re.compile(r"^/usr/lib/llvm-\d+/lib/clang/\d+/include/"), "<clang>/"),
    (re.compile(r"^/usr/include/"), "<system>/"),
    (re.compile(r"^.*/eigen-src/"), "<eigen>/"),
)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def complete_events(raw):
    """Complete events of a trace: "X" events plus paired async "b"/"e" events."""
    events = []
    open_events = defaultdict(list)
    for event in raw:
        if not isinstance(event, dict) or not isinstance(event.get("name"), str) or not is_number(event.get("ts")):
            continue
        phase = event.get("ph")
        if phase == "X" and is_number(event.get("dur")):
            events.append(event)
            continue
        key = (event.get("pid"), event.get("tid"), event.get("cat"), event.get("name"), event.get("id"))
        if phase == "b":
            open_events[key].append(event)
        elif phase == "e" and open_events[key]:
            begin = open_events[key].pop()
            if event["ts"] >= begin["ts"]:
                events.append({"ph": "X", "name": begin["name"], "ts": begin["ts"], "dur": event["ts"] - begin["ts"],
                               "tid": begin.get("tid", 0), "args": begin.get("args") if isinstance(begin.get("args"), dict) else {}})
    return events


def detail_of(event):
    args = event.get("args")
    value = args.get("detail") if isinstance(args, dict) else None
    return value if isinstance(value, str) else ""


def with_self_times(events):
    """[(event, self microseconds)] for events that nest by time.

    An event is the child of the innermost open event that contains it; its duration is taken out of the parent's
    self time. Timestamps are whole microseconds, so a child may end a microsecond after its parent.
    """
    ordered = sorted(events, key=lambda e: (e.get("tid", 0), e["ts"], -e["dur"]))
    result, stack = [], []   # stack: (end, tid, index into result)
    for event in ordered:
        start, end, tid = event["ts"], event["ts"] + event["dur"], event.get("tid", 0)
        while stack and (stack[-1][1] != tid or stack[-1][0] <= start):
            stack.pop()
        self_us = float(event["dur"])
        if stack:
            parent = result[stack[-1][2]]
            parent[1] -= min(event["dur"], stack[-1][0] - start)
        result.append([event, self_us])
        stack.append((end, tid, len(result) - 1))
    return [(event, max(self_us, 0.0)) for event, self_us in result]


def clean_path(path, source_root):
    """A header path as the report shows it: relative to the repository, or `<std>/`, `<system>/`, `<eigen>/`, ..."""
    path = posixpath.normpath(path) if path else path
    root = source_root.rstrip("/") + "/"
    if path.startswith(root):
        return path[len(root):]
    for pattern, label in SYSTEM_PREFIXES:
        if pattern.match(path):
            return pattern.sub(label, path, count=1)
    return path


def clip(text, limit=MAX_NAME_CHARS):
    text = "".join(char if char.isprintable() else " " for char in text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def load_events(path):
    """The complete events of one trace file; ValueError if it is unusable."""
    try:
        if Path(path).stat().st_size > MAX_TRACE_BYTES:
            raise ValueError(f"{path}: too large")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error
    raw = data.get("traceEvents") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise ValueError(f"{path}: no traceEvents")
    return complete_events(raw)


def analyse(events, source_root):
    """The figures of one translation unit: times in seconds, headers and templates as (name, self, inclusive)."""
    named = defaultdict(list)
    for event in events:
        named[event["name"]].append(event)

    def total(name):
        return sum(e["dur"] for e in named.get(name, [])) / 1e6

    headers = [(clean_path(detail_of(event), source_root), self_us / 1e6, event["dur"] / 1e6)
               for event, self_us in with_self_times(named.get(SOURCE, [])) if detail_of(event)]
    templates = [(clip(detail_of(event)), self_us / 1e6, event["dur"] / 1e6)
                 for event, self_us in with_self_times([e for name in TEMPLATE_EVENTS for e in named.get(name, [])])
                 if detail_of(event)]
    frontend, backend = total("Frontend"), total("Backend")
    return {"total_s": total("ExecuteCompiler") or frontend + backend, "frontend_s": frontend, "backend_s": backend,
            "headers": headers, "templates": templates}


def trace_path(object_path):
    """`dir/foo.cpp.o` -> `dir/foo.cpp.json`, or None for an output that is not an object file."""
    return object_path[:-2] + ".json" if object_path.endswith(".o") else None


def collect(build_dir, source_root, objects, source_of, log=print):
    """The trace summary of the given object files (paths relative to the build directory), or None without traces.

    source_of maps an object path to its source file (repository relative). Missing and unreadable traces are
    counted, not fatal.
    """
    build_dir, source_root = Path(build_dir), str(Path(source_root).resolve())
    files, headers, templates = [], defaultdict(lambda: [0.0, 0.0, 0, 0]), defaultdict(lambda: [0.0, 0.0, 0, 0])
    missing = unreadable = 0
    for obj in sorted(objects):
        relative = trace_path(obj)
        path = build_dir / relative if relative else None
        if path is None or not path.is_file():
            missing += 1
            continue
        try:
            figures = analyse(load_events(path), source_root)
        except ValueError as error:
            unreadable += 1
            log(f"warning: skipping trace: {error}")
            continue
        main = source_of(obj)
        files.append({"src": main, "total_s": round(figures["total_s"], 3), "frontend_s": round(figures["frontend_s"], 3),
                      "backend_s": round(figures["backend_s"], 3), "inclusions": len(figures["headers"]),
                      "instantiations": len(figures["templates"])})
        for table, rows in ((headers, figures["headers"]), (templates, figures["templates"])):
            seen = set()
            for name, self_s, inclusive_s in rows:
                entry = table[name]
                entry[0] += self_s
                entry[1] += inclusive_s
                entry[2] += 1
                if name not in seen:
                    seen.add(name)
                    entry[3] += 1
    if not files:
        return None

    def top(table, limit):
        ranked = sorted(table.items(), key=lambda item: (-item[1][0], item[0]))[:limit]
        return [{"f": name, "self_s": round(v[0], 3), "incl_s": round(v[1], 3), "n": v[2], "tus": v[3]} for name, v in ranked]

    return {
        "files": files, "headers": top(headers, TOP_HEADERS), "templates": top(templates, TOP_TEMPLATES),
        "missing": missing, "unreadable": unreadable,
        "totals": {"total_s": round(sum(f["total_s"] for f in files), 2), "frontend_s": round(sum(f["frontend_s"] for f in files), 2),
                   "backend_s": round(sum(f["backend_s"] for f in files), 2),
                   "header_self_s": round(sum(v[0] for v in headers.values()), 2),
                   "template_self_s": round(sum(v[0] for v in templates.values()), 2),
                   "inclusions": sum(v[2] for v in headers.values()), "instantiations": sum(v[2] for v in templates.values()),
                   "distinct_headers": len(headers), "distinct_templates": len(templates)},
    }
