#!/usr/bin/env python3
"""Render the clang build time comment of a pull request (standard library only).

Compares the clang-traces-detail.json of the pull request (see
summarise_traces.py) with the one of a recent `dev` run and writes Markdown
for a single, updated-in-place pull request comment. Informational only: it
never fails and always writes a report, also when there is no baseline or no
trace data.

Timings on shared runners are noisy (the same code took 882 s to 1707 s of
compiler time in five runs), so the per-header, per-file and per-template
changes are *speed-adjusted*: the baseline value is scaled by the ratio of the
two total compile times before the difference is taken. This is only a
first-order correction: clang leaves events shorter than 0.5 ms out of the
trace, so a faster run records fewer events, and the counts of header
inclusions and template instantiations (and the self time of headers made of
many small events) are speed-dependent too. Counts are therefore shown without
a percentage change.

The include graph (from `ninja -t deps`, see summarise_build.py) is exact and
the same for two builds of the same code, so its changes need no threshold or
speed adjustment: the totals and the headers whose fan-in (number of files that
include them) changed are shown as they are.
"""

import argparse
import json
import re
import sys
from pathlib import Path

MARKER = "<!-- clang-build-times-comment -->"
DETAIL_KIND = "clang-trace-detail"
INCLUDES_KIND = "ninja-includes-detail"
CPU_MODEL = re.compile(r"[A-Za-z0-9 ()@.,_+/-]{1,80}")
SUMMARY_KINDS = (DETAIL_KIND, "clang-trace-summary")

# A change is listed only if it is both large enough in seconds and in percent.
HEADER_MIN_S, HEADER_MIN_REL = 0.5, 0.15
FILE_MIN_S, FILE_MIN_REL = 1.0, 0.20
TEMPLATE_MIN_S, TEMPLATE_MIN_REL = 0.5, 0.20
SHOW_HEAVIER = 10
SHOW_LIGHTER = 5
SHOW_FAN_IN = 10
SHOW_FILES = 5
SHOW_TEMPLATES = 5
DETAIL_FILES, DETAIL_HEADERS, DETAIL_TEMPLATES = 20, 30, 30
MAX_COMMENT_CHARS = 60_000
NAME_CHARS = 140

HEADLINE = (
    ("Compiler time (CPU, all files)", "total_s", "time"),
    ("Frontend (parsing, templates)", "frontend_s", "time"),
    ("Backend (optimisation, code generation)", "backend_s", "time"),
    ("Header inclusions (events of at least 0.5 ms)", "source_events", "count"),
    ("Template instantiation events (at least 0.5 ms)", "instantiation_events", "count"),
)


def load_summary(path):
    """The parsed detail or summary file, or None if it is missing or not usable."""
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("kind") not in SUMMARY_KINDS:
        return None
    for key in ("total_s", "frontend_s", "backend_s", "source_events", "instantiation_events", "files"):
        if not is_number(data.get(key)):
            return None
    return data


def load_runner(path):
    """The CPU model recorded in a build-analysis.json, or None if missing or not usable."""
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = data["runner"]["cpu_model"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return model if isinstance(model, str) and CPU_MODEL.fullmatch(model) else None


def runner_line(model, base_model):
    """One sentence on the CPUs of this run and of the baseline, or None without data."""
    if model is None:
        return None
    if base_model is None:
        return f"Runner CPU: `{model}`."
    if model == base_model:
        return f"Both runs used the runner CPU `{model}`."
    return f"Runner CPU: `{model}` here, `{base_model}` for the baseline. Runner CPUs differ in speed, so part of the change can be the hardware."


def load_includes(path):
    """The parsed include graph detail, or None if missing or not usable."""
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("kind") != INCLUDES_KIND:
        return None
    fan_in = data.get("fan_in")
    if not is_number(data.get("include_pairs")) or not is_number(data.get("project_pairs")) or not isinstance(fan_in, dict):
        return None
    clean = {name: count for name, count in fan_in.items() if isinstance(name, str) and is_number(count) and count >= 0}
    return dict(data, fan_in=clean)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def entries(data, key, name_key, fields):
    """The usable entries of one list of a summary: dicts with a name and numbers."""
    result = []
    for entry in data.get(key) or []:
        if isinstance(entry, dict) and isinstance(entry.get(name_key), str) and all(is_number(entry.get(f)) for f in fields):
            result.append(entry)
    return result


def limit_of(data, key):
    limits = data.get("limits")
    value = limits.get(key) if isinstance(limits, dict) else None
    return value if isinstance(value, int) and value > 0 else None


def format_seconds(value):
    if value is None:
        return "-"
    if abs(value) < 60:
        return f"{value:.1f} s"
    seconds = int(round(abs(value)))
    minutes, seconds = divmod(seconds, 60)
    sign = "-" if value < 0 else ""
    return f"{sign}{minutes}m {seconds:02d}s"


def format_signed_seconds(value):
    return f"{'+' if value >= 0 else '-'}{abs(value):.1f} s"


def format_percent_change(base, new):
    if not base:
        return "n/a"
    return f"{(new - base) / base * 100:+.1f}%"


def code(text, limit=NAME_CHARS):
    """Inline code for an untrusted name: no backticks, pipes or line breaks, clipped."""
    text = "".join(ch if ch.isprintable() else " " for ch in str(text)).replace("`", "'").replace("|", "/")
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return f"`{text}`"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def speed_factor(pr, base):
    if base and base["total_s"] > 0 and pr["total_s"] > 0:
        return pr["total_s"] / base["total_s"]
    return 1.0


def changes(base_values, pr_values, scale, base_cut, min_s, min_rel):
    """Speed-adjusted changes: [(name, base value or None, pr value, delta, relative or None)].

    A name missing from the baseline list was either not there or below the
    list's cut-off (`base_cut`, its smallest value if the list was full, else 0).
    """
    rows = []
    for name, value in pr_values.items():
        if name in base_values:
            expected = base_values[name] * scale
            delta = value - expected
            relative = delta / expected if expected > 0 else None
            if abs(delta) >= min_s and (relative is None or abs(relative) >= min_rel):
                rows.append((name, base_values[name], value, delta, relative))
        else:
            floor = base_cut * scale
            delta = value - floor
            if delta >= min_s and (floor == 0 or delta / floor >= min_rel):
                rows.append((name, None, value, delta, None))
    return sorted(rows, key=lambda row: (-row[3], row[0]))


def cut_off(data, key, values):
    limit = limit_of(data, key)
    if limit is not None and len(values) >= limit and values:
        return min(values.values())
    return 0.0


def change_cell(row):
    _, base, value, delta, relative = row
    if base is None:
        return f"{format_signed_seconds(delta)} (new in the list)"
    return f"{format_signed_seconds(delta)} ({relative * 100:+.0f}%)" if relative is not None else format_signed_seconds(delta)


def headline_rows(pr, base):
    rows = []
    for label, key, kind in HEADLINE:
        value = pr[key]
        shown = format_seconds(value) if kind == "time" else f"{int(value):,}"
        if base is None:
            rows.append([label, shown])
            continue
        before = format_seconds(base[key]) if kind == "time" else f"{int(base[key]):,}"
        change = format_percent_change(base[key], value) if kind == "time" else "-"
        rows.append([label, before, shown, change])
    return rows


def header_section(pr, base, scale, includes=None, base_includes=None):
    pr_self = {e["header"]: e["self_s"] for e in entries(pr, "headers", "header", ("self_s",))}
    pr_files = {e["header"]: e.get("files") for e in entries(pr, "headers", "header", ("self_s",))}
    base_entries = entries(base, "headers", "header", ("self_s",))
    base_self = {e["header"]: e["self_s"] for e in base_entries}
    base_files = {e["header"]: e.get("files") for e in base_entries}
    rows = changes(base_self, pr_self, scale, cut_off(base, "headers", base_self), HEADER_MIN_S, HEADER_MIN_REL)

    exact_before = base_includes["fan_in"] if base_includes else {}
    exact_after = includes["fan_in"] if includes else {}

    def files_cell(name):
        if name in exact_before or name in exact_after:
            before, after = exact_before.get(name), exact_after.get(name)
            return f"{before if before is not None else '-'} → {after if after is not None else '-'}"
        before, after = base_files.get(name), pr_files.get(name)
        return f"~{before if before is not None else '-'} → ~{after if after is not None else '-'}"

    def render(selected):
        return table(
            ["Header", "Self time before", "Self time now", "Change (speed-adjusted)", "Files including it (exact, ~ = from the trace)"],
            [
                [
                    code(row[0]),
                    format_seconds(row[1]) if row[1] is not None else "-",
                    format_seconds(row[2]),
                    change_cell(row),
                    files_cell(row[0]),
                ]
                for row in selected
            ],
        )

    heavier = [row for row in rows if row[3] > 0][:SHOW_HEAVIER]
    lighter = [row for row in reversed(rows) if row[3] < 0][:SHOW_LIGHTER]
    return heavier, lighter, render


def simple_changes(pr, base, scale, key, name_key, value_key, minimum, relative):
    pr_values = {e[name_key]: e[value_key] for e in entries(pr, key, name_key, (value_key,))}
    base_values = {e[name_key]: e[value_key] for e in entries(base, key, name_key, (value_key,))}
    return changes(base_values, pr_values, scale, cut_off(base, key, base_values), minimum, relative)


def fan_in_changes(base, pr):
    """[(header, before or None, now or None, delta)] for every project header whose fan-in changed."""
    rows = []
    for name in set(base) | set(pr):
        before, after = base.get(name), pr.get(name)
        delta = (after or 0) - (before or 0)
        if delta:
            rows.append((name, before, after, delta))
    return sorted(rows, key=lambda row: (-abs(row[3]), row[0]))


def signed(value):
    return f"{value:+,}"


def include_section(includes, base_includes):
    """The exact include graph part of the comment (Markdown lines)."""
    lines = ["#### Include graph (exact, from `ninja -t deps`)", ""]
    pairs, project = includes["include_pairs"], includes["project_pairs"]
    if base_includes is None:
        lines += table(["", "This pull request"], [["Include pairs (translation unit, header)", f"{int(pairs):,}"], ["... of which project headers", f"{int(project):,}"]])
        return lines
    if includes.get("digest") and includes.get("digest") == base_includes.get("digest"):
        lines += [f"The include graph is identical to dev's ({int(pairs):,} include pairs, {int(project):,} of them project headers)."]
        return lines
    lines += table(
        ["", "dev", "This pull request", "Change"],
        [
            ["Include pairs (translation unit, header)", f"{int(base_includes['include_pairs']):,}", f"{int(pairs):,}", f"{signed(int(pairs - base_includes['include_pairs']))} ({format_percent_change(base_includes['include_pairs'], pairs)})"],
            ["... of which project headers", f"{int(base_includes['project_pairs']):,}", f"{int(project):,}", signed(int(project - base_includes["project_pairs"]))],
        ],
    )
    changed = fan_in_changes(base_includes["fan_in"], includes["fan_in"])
    lines.append("")
    if not changed:
        lines.append("No project header changed the number of files that include it (the graph differs elsewhere, for example in system headers).")
        return lines
    lines += [f"**Project headers whose fan-in (number of files including them) changed** ({len(changed)} in total, the {min(SHOW_FAN_IN, len(changed))} largest changes):", ""]
    lines += table(
        ["Header", "Included by before", "Included by now", "Change"],
        [
            [code(name), "-" if before is None else str(int(before)), "-" if after is None else str(int(after)), signed(int(delta))]
            for name, before, after, delta in changed[:SHOW_FAN_IN]
        ],
    )
    return lines


def detail_tables(pr):
    lines = ["<details><summary>Slowest files, heaviest headers and templates of this pull request</summary>", ""]
    files = entries(pr, "slowest_files", "file", ("total_s", "frontend_s", "backend_s"))[:DETAIL_FILES]
    headers = entries(pr, "headers", "header", ("self_s", "inclusive_s"))[:DETAIL_HEADERS]
    templates = entries(pr, "templates", "name", ("self_s", "count"))[:DETAIL_TEMPLATES]
    lines += [f"**Slowest {len(files)} files**", ""]
    lines += table(
        ["File", "Total", "Frontend", "Backend"],
        [[code(e["file"]), format_seconds(e["total_s"]), format_seconds(e["frontend_s"]), format_seconds(e["backend_s"])] for e in files],
    )
    lines += ["", f"**Heaviest {len(headers)} headers by self time** (summed over all files)", ""]
    lines += table(
        ["Header", "Self", "Inclusive", "Files"],
        [[code(e["header"]), format_seconds(e["self_s"]), format_seconds(e["inclusive_s"]), str(e.get("files", "-"))] for e in headers],
    )
    lines += ["", f"**Top {len(templates)} template instantiations by self time**", ""]
    lines += table(
        ["Instantiation", "Self", "Count"],
        [[code(e["name"]), format_seconds(e["self_s"]), f"{int(e['count']):,}"] for e in templates],
    )
    lines += ["", "</details>"]
    return lines


NOTES = (
    "<sub>Informational, never a failing check. Clang 20, Debug, no ccache, on a shared runner: "
    "the same code took between 15 and 28 minutes of compiler time in different runs, so changes of "
    "single headers, files and templates are speed-adjusted (the baseline is scaled by the ratio of "
    "the total compile times). That is only a first-order correction: clang leaves events shorter "
    "than 0.5 ms out of the trace, so a faster run also records fewer events. The counts, the "
    "files-including-it column and the self time of headers made of many small events therefore "
    "depend on runner speed too, and small changes in them are not meaningful. "
    "Only the longest lists of both builds are compared, so a header can appear as new when it "
    "crossed the cut-off of the baseline's list.</sub>"
)


def render(pr, base, *, baseline_sha=None, run_url=None, includes=None, base_includes=None, runner=None, base_runner=None):
    """The Markdown comment for a pull request with trace data."""
    lines = [MARKER, "### Clang build times (clang-20, Debug), informational", ""]
    note = runner_line(runner, base_runner if base is not None else None)
    if base is None:
        lines += ["No baseline yet: no `dev` run has stored its clang summary (or it expired), so only this pull request's numbers are shown.", ""]
        lines += [note, ""] if note else []
        lines += table(["", "This pull request"], headline_rows(pr, None))
        if includes is not None:
            lines += [""] + include_section(includes, None)
    else:
        sha = f"`{str(baseline_sha)[:7]}`" if baseline_sha else "a recent run"
        lines += [f"Compared with `dev` {sha}.", ""] + ([note, ""] if note else [])
        lines += table(["", "dev", "This pull request", "Change"], headline_rows(pr, base))
        scale = speed_factor(pr, base)
        lines += ["", f"The compiler time of this run was {scale:.2f} times the baseline's; per-item changes below are measured after scaling the baseline by that factor.", ""]
        if includes is not None:
            lines += include_section(includes, base_includes) + ["", "<sub>The baseline is the newest dev build, so commits merged to dev since then also show up here.</sub>", ""]
        heavier, lighter, render_headers = header_section(pr, base, scale, includes, base_includes)
        lines += ["#### Headers that got heavier", ""]
        if heavier:
            lines += render_headers(heavier)
        else:
            lines.append(f"None: no header changed by at least {HEADER_MIN_S:g} s and {HEADER_MIN_REL * 100:.0f}% (speed-adjusted).")
        lines.append("")
        file_rows = simple_changes(pr, base, scale, "slowest_files", "file", "total_s", FILE_MIN_S, FILE_MIN_REL)
        template_rows = simple_changes(pr, base, scale, "templates", "name", "self_s", TEMPLATE_MIN_S, TEMPLATE_MIN_REL)
        more = []
        if lighter:
            more += ["**Headers that got lighter**", ""] + render_headers(lighter) + [""]
        shown_files = file_rows[:SHOW_FILES]
        if shown_files:
            more += ["**Files that changed most**", ""] + simple_rows(shown_files, "File") + [""]
        shown_templates = template_rows[:SHOW_TEMPLATES]
        if shown_templates:
            more += ["**Template instantiations that changed most**", ""] + simple_rows(shown_templates, "Instantiation") + [""]
        if more:
            lines += ["<details><summary>More changes</summary>", ""] + more + ["</details>", ""]
    if lines[-1] != "":
        lines.append("")
    lines += detail_tables(pr) + ["", NOTES]
    if run_url:
        lines += ["", f"[Workflow run]({run_url})"]
    return fit("\n".join(lines) + "\n")


def simple_rows(rows, label):
    return table(
        [label, "Before", "Now", "Change (speed-adjusted)"],
        [[code(r[0]), format_seconds(r[1]) if r[1] is not None else "-", format_seconds(r[2]), change_cell(r)] for r in rows],
    )


def render_unavailable(status, run_url=None):
    """The comment when there is no usable trace data."""
    reason = {
        "failure": "The clang build failed, so there are no build times for this commit.",
        "cancelled": "The clang build was cancelled before it produced build times.",
    }.get(status, "The clang build did not produce usable `-ftime-trace` data for this commit.")
    lines = [MARKER, "### Clang build times (clang-20, Debug), informational", "", reason]
    if run_url:
        lines += ["", f"[Workflow run]({run_url})"]
    return "\n".join(lines) + "\n"


def fit(text):
    """Keep a comment within GitHub's size limit by dropping the details first."""
    if len(text) <= MAX_COMMENT_CHARS:
        return text
    cut = text.find("<details><summary>Slowest files")
    if cut != -1:
        text = text[:cut] + "\n" + NOTES + "\n"
    return text if len(text) <= MAX_COMMENT_CHARS else text[: MAX_COMMENT_CHARS - 20] + "\n…(truncated)\n"


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Render the clang build time comment.")
    parser.add_argument("--pr", required=True, help="clang-traces-detail.json of this build")
    parser.add_argument("--baseline", help="clang-traces-detail.json of a recent dev build")
    parser.add_argument("--baseline-sha", help="dev commit of the baseline")
    parser.add_argument("--includes", help="ninja-includes.json of this build (exact include graph)")
    parser.add_argument("--baseline-includes", help="ninja-includes.json of the dev build")
    parser.add_argument("--runner", help="build-analysis.json of this build (runner CPU)")
    parser.add_argument("--baseline-runner", help="build-analysis.json of the dev build")
    parser.add_argument("--run-url", help="link to this workflow run")
    parser.add_argument("--status", default="success", help="outcome of the clang build step")
    parser.add_argument("--out", required=True, help="Markdown file to write")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    pr = load_summary(args.pr)
    if pr is None:
        text = render_unavailable(args.status, args.run_url)
    else:
        text = render(
            pr,
            load_summary(args.baseline),
            baseline_sha=args.baseline_sha,
            run_url=args.run_url,
            includes=load_includes(args.includes),
            base_includes=load_includes(args.baseline_includes),
            runner=load_runner(args.runner),
            base_runner=load_runner(args.baseline_runner),
        )
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text):,} characters, {'no trace data' if pr is None else 'baseline ' + ('found' if args.baseline and load_summary(args.baseline) else 'missing')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
