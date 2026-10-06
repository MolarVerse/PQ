"""Which area, group and kind a path belongs to. Edit AREA_RULES to change the splits; everything else follows.

area  : where in the repository (src, include, tests, perf, ...)
group : a coarser view of the areas (production, tests, perf, ci, docs, build, data, other)
kind  : what the file is (header, tpp, source, cmake, python, shell, config, doc, other)
class : header-like (header + tpp: compiled into every includer) or source or other
"""

import os

# The first matching prefix wins. A prefix ending in "/" matches a directory, anything else the exact path.
# Entries with area None are left out of all counts (submodules are not part of the code base).
AREA_RULES = (
    ("external/", None, None),
    ("benchmarks/perf/", "perf", "perf"),
    ("benchmarks/", "benchmarks", "perf"),
    ("tests/", "tests", "tests"),
    ("integration_tests/", "integration_tests", "tests"),
    ("src/", "src", "production"),
    ("include/", "include", "production"),
    ("apps/", "apps", "production"),
    (".github/workflows/", "ci_workflows", "ci"),
    (".github/ci-metrics/data/", "ci_data", "data"),
    (".github/", "ci_other", "ci"),
    ("scripts/", "scripts", "other"),
    ("docs/", "docs", "docs"),
    ("changes/", "changelog", "docs"),
    ("CHANGELOG.md", "changelog", "docs"),
    ("DEV-CHANGELOG.md", "changelog", "docs"),
    (".cmake/", "cmake", "build"),
    ("cmake/", "cmake", "build"),
    ("CMakeLists.txt", "cmake", "build"),
)
DEFAULT_AREA = ("other", "other")

KIND_BY_SUFFIX = {
    ".hpp": "header", ".h": "header", ".hxx": "header", ".hh": "header",
    ".tpp": "tpp",
    ".cpp": "source", ".cc": "source", ".cxx": "source", ".c": "source",
    ".cmake": "cmake",
    ".py": "python",
    ".sh": "shell", ".bash": "shell",
    ".yml": "config", ".yaml": "config", ".toml": "config", ".json": "config", ".jsonl": "config",
    ".cfg": "config", ".ini": "config",
    ".md": "doc", ".rst": "doc", ".txt": "doc",
}
CLASS_OF_KIND = {"header": "header_like", "tpp": "header_like", "source": "source"}
CPP_KINDS = ("header", "tpp", "source")
GROUPS = ("production", "tests", "perf", "ci", "docs", "build", "data", "other")
AREAS = ("src", "include", "apps", "tests", "integration_tests", "perf", "benchmarks", "ci_workflows", "ci_other",
         "ci_data", "scripts", "docs", "changelog", "cmake", "other")


def kind_of(path):
    name = os.path.basename(path)
    if name == "CMakeLists.txt":
        return "cmake"
    return KIND_BY_SUFFIX.get(os.path.splitext(name)[1].lower(), "other")


def classify(path):
    """(area, group, kind) of a path, or None if it is not counted."""
    for prefix, area, group in AREA_RULES:
        matches = path.startswith(prefix) if prefix.endswith("/") else path == prefix
        if matches:
            return None if area is None else (area, group, kind_of(path))
    return DEFAULT_AREA + (kind_of(path),)


def class_of(kind):
    return CLASS_OF_KIND.get(kind, "other")


def group_of_area(area):
    for _, rule_area, group in AREA_RULES:
        if rule_area == area:
            return group
    return DEFAULT_AREA[1]
