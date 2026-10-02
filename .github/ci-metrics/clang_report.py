#!/usr/bin/env python3
"""Render the clang build time comment of a pull request (standard library only).

Compares the clang-traces-detail.json of the pull request (see
summarise_traces.py) with the one of a recent `dev` run and writes Markdown
for a single, updated-in-place pull request comment. Informational only: it
never fails and always writes a report, also when there is no baseline or no
trace data.

Timings on shared runners are noisy (the same build differed by about 30%
between two runs), so the per-header, per-file and per-template changes are
*speed-adjusted*: the baseline value is scaled by the ratio of the two total
compile times before the difference is taken. The counts of header inclusions
and template instantiations do not depend on runner speed.
"""

import argparse
import json
import sys
from pathlib import Path

MARKER = "<!-- clang-build-times-comment -->"
DETAIL_KIND = "clang-trace-detail"
SUMMARY_KINDS = (DETAIL_KIND, "clang-trace-summary")

# A change is listed only if it is both large enough in seconds and in percent.
HEADER_MIN_S, HEADER_MIN_REL = 0.5, 0.15
FILE_MIN_S, FILE_MIN_REL = 1.0, 0.20
TEMPLATE_MIN_S, TEMPLATE_MIN_REL = 0.5, 0.20
SHOW_HEAVIER = 10
SHOW_LIGHTER = 5
SHOW_FILES = 5
SHOW_TEMPLATES = 5
DETAIL_FILES, DETAIL_HEADERS, DETAIL_TEMPLATES = 20, 30, 30
MAX_COMMENT_CHARS = 60_000
NAME_CHARS = 140

HEADLINE = (
    ("Compiler time (CPU, all files)", "total_s", "time"),
    ("Frontend (parsing, templates)", "frontend_s", "time"),
    ("Backend (optimisation, code generation)", "backend_s", "time"),
    ("Header inclusions", "source_events", "count"),
    ("Template instantiation events", "instantiation_events", "count"),
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
        rows.append([label, before, shown, format_percent_change(base[key], value)])
    return rows


def header_section(pr, base, scale):
    pr_self = {e["header"]: e["self_s"] for e in entries(pr, "headers", "header", ("self_s",))}
    pr_files = {e["header"]: e.get("files") for e in entries(pr, "headers", "header", ("self_s",))}
    base_entries = entries(base, "headers", "header", ("self_s",))
    base_self = {e["header"]: e["self_s"] for e in base_entries}
    base_files = {e["header"]: e.get("files") for e in base_entries}
    rows = changes(base_self, pr_self, scale, cut_off(base, "headers", base_self), HEADER_MIN_S, HEADER_MIN_REL)

    def files_cell(name):
        before, after = base_files.get(name), pr_files.get(name)
        return f"{before if before is not None else '-'} → {after if after is not None else '-'}"

    def render(selected):
        return table(
            ["Header", "Self time before", "Self time now", "Change (speed-adjusted)", "Files including it"],
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
    "the same build differed by about 30% between two runs, so changes of single headers, files and "
    "templates are speed-adjusted (the baseline is scaled by the ratio of the total compile times); "
    "the counts of header inclusions and template instantiations do not depend on runner speed. "
    "Only the longest lists of both builds are compared, so a header can appear as new when it "
    "crossed the cut-off of the baseline's list.</sub>"
)


def render(pr, base, *, baseline_sha=None, run_url=None):
    """The Markdown comment for a pull request with trace data."""
    lines = [MARKER, "### Clang build times (clang-20, Debug), informational", ""]
    if base is None:
        lines += ["No baseline yet: no `dev` run has stored its clang summary (or it expired), so only this pull request's numbers are shown.", ""]
        lines += table(["", "This pull request"], headline_rows(pr, None))
    else:
        sha = f"`{str(baseline_sha)[:7]}`" if baseline_sha else "a recent run"
        lines += [f"Compared with `dev` {sha}.", ""]
        lines += table(["", "dev", "This pull request", "Change"], headline_rows(pr, base))
        scale = speed_factor(pr, base)
        lines += ["", f"The compiler time of this run was {scale:.2f} times the baseline's; per-item changes below are measured after scaling the baseline by that factor.", ""]
        heavier, lighter, render_headers = header_section(pr, base, scale)
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
        text = render(pr, load_summary(args.baseline), baseline_sha=args.baseline_sha, run_url=args.run_url)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text):,} characters, {'no trace data' if pr is None else 'baseline ' + ('found' if args.baseline and load_summary(args.baseline) else 'missing')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
