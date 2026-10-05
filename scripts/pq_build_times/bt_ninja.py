"""Readers for what Ninja leaves behind: `.ninja_log` and `ninja -t deps`.

Deliberately independent of the CI metrics scripts (.github/ci-metrics), so that
changes to the CI pipeline cannot break local build-time tracking.
"""

import hashlib
import os
import re
from collections import Counter, namedtuple

LogEntry = namedtuple("LogEntry", "start end mtime output hash")
COMPILE_SUFFIXES = (".o", ".obj", ".gch", ".pch")
OBJECT_SUFFIXES = (".o", ".obj")
SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx", ".c")
HEADER_SUFFIXES = (".hpp", ".h", ".hxx", ".tpp")
DEPS_HEADER = re.compile(r"^(?P<target>\S.*?): #deps \d+, deps mtime")


def log_lines(text):
    """The data lines of a .ninja_log, as a set (to tell which ones a build added)."""
    return {line for line in text.splitlines() if line and not line.startswith("#")}


def parse_log(lines):
    entries = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        try:
            entries.append(LogEntry(int(parts[0]), int(parts[1]), int(parts[2]), parts[3], parts[4]))
        except ValueError:
            continue
    return entries


def summarise_steps(entries):
    """Steps, CPU seconds, link seconds and the tail after the last compile of one build."""
    unique = {(entry.start, entry.end, entry.hash): entry for entry in entries}.values()
    steps = list(unique)
    compiles = [step for step in steps if step.output.endswith(COMPILE_SUFFIXES)]
    cpu = sum(step.end - step.start for step in steps) / 1000
    link = sum(step.end - step.start for step in steps if step not in compiles) / 1000
    tail = 0.0
    if compiles and steps:
        tail = (max(step.end for step in steps) - max(step.end for step in compiles)) / 1000
    return {"steps": len(steps), "cpu_s": round(cpu, 3), "link_s": round(link, 3), "tail_s": round(tail, 3)}


def parse_deps(text):
    """{target: [dependency, ...]} from `ninja -t deps`."""
    result = {}
    current = None
    for line in text.splitlines():
        if not line.strip():
            current = None
            continue
        header = DEPS_HEADER.match(line)
        if header and not line.startswith(" "):
            current = header.group("target")
            result[current] = []
        elif current is not None and line.startswith("    "):
            result[current].append(line.strip())
    return result


def _under(path, root):
    return path == root or path.startswith(root + os.sep)


def include_graph(deps, source_root, build_dir):
    """The exact include graph of the project files and numbers that do not depend on machine speed.

    Returns {"metrics": {...}, "fan_in": {file: objects including it}, "sources": {object: source}, "sizes":
    {object: number of project files it depends on}}. Paths are relative to the source root; system headers,
    generated files in the build directory and `external/` are not project files.
    """
    source_root = os.path.realpath(source_root)
    build_dir = os.path.realpath(build_dir)
    external = os.path.join(source_root, "external")

    def project(path):
        absolute = os.path.normpath(path if os.path.isabs(path) else os.path.join(build_dir, path))
        if not _under(absolute, source_root) or _under(absolute, build_dir) or _under(absolute, external):
            return None
        return os.path.relpath(absolute, source_root)

    per_object = {}
    all_files = set()
    pairs = 0
    for target, dependencies in deps.items():
        if not target.endswith(OBJECT_SUFFIXES):
            continue
        all_files.update(dependencies)
        pairs += len(set(dependencies))
        per_object[target] = [name for name in (project(dep) for dep in dependencies) if name]
    fan_in = Counter()
    sources = {}
    sizes = {}
    digest = hashlib.sha256()
    for target in sorted(per_object):
        files = sorted(set(per_object[target]))
        sizes[target] = len(files)
        fan_in.update(files)
        for name in files:
            digest.update(f"{target}\0{name}\n".encode("utf-8"))
        source = next((name for name in per_object[target] if name.endswith(SOURCE_SUFFIXES)), None)
        if source:
            sources[target] = source
    metrics = {
        "objects": len(per_object),
        "unique_files": len(all_files),
        "include_pairs": pairs,
        "project_files": len(fan_in),
        "project_pairs": sum(fan_in.values()),
        "digest": digest.hexdigest(),
    }
    return {"metrics": metrics, "fan_in": dict(fan_in), "sources": sources, "sizes": sizes}


def choose_targets(graph):
    """Deterministic touch targets: the most widely included header, a median header and a median source.

    A source file is "median" if its object depends on a median number of project files among the objects
    built from files under src/; a header is "median" among the project headers included at least twice. With an
    even number of candidates the lower middle one is taken, so the median header differs from the top one.
    """
    fan_in = graph["fan_in"]
    headers = sorted((name for name in fan_in if name.endswith(HEADER_SUFFIXES)), key=lambda n: (-fan_in[n], n))
    shared = sorted((name for name in headers if fan_in[name] >= 2), key=lambda n: (fan_in[n], n))
    leaves = sorted(
        (
            (graph["sizes"][obj], source)
            for obj, source in graph["sources"].items()
            if source.startswith("src" + os.sep)
        )
    )
    return {
        "leaf": leaves[(len(leaves) - 1) // 2][1] if leaves else None,
        "header_top": headers[0] if headers else None,
        "header_median": shared[(len(shared) - 1) // 2] if shared else None,
    }
