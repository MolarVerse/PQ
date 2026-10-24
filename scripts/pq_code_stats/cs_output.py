"""Tidy CSV outputs: changes (one row per change), splits and state (long form), metrics (state per change), weekly.

Column names are stable so the files can be loaded straight into pandas, R, a spreadsheet or gnuplot.
"""

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cs_classify

SCOPES = ("prod", "tests", "perf")   # C++ kinds only: production = src+include+apps, tests = tests, perf = perf+benchmarks
SCOPE_OF_AREA = {"src": "prod", "include": "prod", "apps": "prod", "tests": "tests", "perf": "perf", "benchmarks": "perf"}
CLASSES = ("header_like", "source")
CI_AREAS = {"ci_workflows": "ci_workflow", "ci_other": "ci_other"}


def iso(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def week_of(moment):
    year, week, _ = moment.isocalendar()
    return f"{year}-W{week:02d}"


def week_start(moment):
    return (moment.date() - timedelta(days=moment.weekday())).isoformat()


def cpp_scope(area, kind):
    """(scope, class) of a C++ file (production / tests / perf, header-like / source), else None."""
    if kind in cs_classify.CPP_KINDS and area in SCOPE_OF_AREA:
        return SCOPE_OF_AREA[area], cs_classify.class_of(kind)
    return None


def change_columns():
    columns = ["sha", "date", "week", "kind", "pr", "title", "author", "files_added", "files_deleted",
               "files_modified", "lines_added", "lines_deleted", "lines_net"]
    for group in cs_classify.GROUPS:
        columns += [f"{group}_added", f"{group}_deleted"]
    for scope in SCOPES:
        for cls in CLASSES:
            columns += [f"{scope}_{cls}_added", f"{scope}_{cls}_deleted"]
    return columns + ["ci_runs", "ci_attempts", "ci_failed_runs"]


def change_row(record, ci_by_pr):
    change = record.change
    row = {column: 0 for column in change_columns()}
    row.update(sha=change.sha, date=iso(change.date), week=week_of(change.date), kind=change.kind,
               pr=change.pr or "", title=change.title, author=change.author)
    for (area, kind), split in record.splits.items():
        group = cs_classify.group_of_area(area)
        row["files_added"] += split.files_added
        row["files_deleted"] += split.files_deleted
        row["files_modified"] += split.files_modified
        row["lines_added"] += split.lines_added
        row["lines_deleted"] += split.lines_deleted
        row[f"{group}_added"] += split.lines_added
        row[f"{group}_deleted"] += split.lines_deleted
        scope = cpp_scope(area, kind)
        if scope:
            row[f"{scope[0]}_{scope[1]}_added"] += split.lines_added
            row[f"{scope[0]}_{scope[1]}_deleted"] += split.lines_deleted
    row["lines_net"] = row["lines_added"] - row["lines_deleted"]
    ci = ci_by_pr.get(change.pr) if ci_by_pr is not None and change.pr else None
    if ci_by_pr is None:
        row.update(ci_runs="", ci_attempts="", ci_failed_runs="")
    else:
        row.update(ci_runs=ci["runs"] if ci else 0, ci_attempts=ci["attempts"] if ci else 0,
                   ci_failed_runs=ci["failed"] if ci else 0)
    return row


def split_rows(record):
    change = record.change
    for (area, kind), split in sorted(record.splits.items()):
        yield {
            "sha": change.sha, "date": iso(change.date), "pr": change.pr or "", "change_kind": change.kind,
            "area": area, "group": cs_classify.group_of_area(area), "file_kind": kind,
            "code_class": cs_classify.class_of(kind), **split._asdict(),
        }


def state_rows(record):
    change = record.change
    for (area, kind), (files, lines) in sorted(record.state.items()):
        yield {
            "sha": change.sha, "date": iso(change.date), "pr": change.pr or "", "area": area,
            "group": cs_classify.group_of_area(area), "file_kind": kind,
            "code_class": cs_classify.class_of(kind), "files": files, "lines": lines,
        }


def metrics_columns():
    columns = ["sha", "date", "pr"]
    for scope in SCOPES:
        for cls in CLASSES:
            columns += [f"{scope}_{cls}_files", f"{scope}_{cls}_lines"]
    columns += ["prod_source_share", "tests_to_prod", "perf_to_prod", "tests_perf_to_prod",
                "ci_workflow_files", "ci_workflow_lines", "ci_other_files", "ci_other_lines", "ci_files", "ci_lines",
                "integration_files", "integration_lines"]
    for group in cs_classify.GROUPS:
        columns += [f"{group}_files", f"{group}_lines"]
    return columns


def ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else ""


def metrics_row(record):
    row = {column: 0 for column in metrics_columns()}
    row.update(sha=record.change.sha, date=iso(record.change.date), pr=record.change.pr or "")
    for (area, kind), (files, lines) in record.state.items():
        group = cs_classify.group_of_area(area)
        row[f"{group}_files"] += files
        row[f"{group}_lines"] += lines
        scope = cpp_scope(area, kind)
        if scope:
            row[f"{scope[0]}_{scope[1]}_files"] += files
            row[f"{scope[0]}_{scope[1]}_lines"] += lines
        if area in CI_AREAS:
            row[f"{CI_AREAS[area]}_files"] += files
            row[f"{CI_AREAS[area]}_lines"] += lines
        if area == "integration_tests":
            row["integration_files"] += files
            row["integration_lines"] += lines
    row["ci_files"] = row["ci_workflow_files"] + row["ci_other_files"]
    row["ci_lines"] = row["ci_workflow_lines"] + row["ci_other_lines"]
    prod = row["prod_header_like_lines"] + row["prod_source_lines"]
    tests = row["tests_header_like_lines"] + row["tests_source_lines"]
    perf = row["perf_header_like_lines"] + row["perf_source_lines"]
    row["prod_source_share"] = ratio(row["prod_source_lines"], prod)
    row["tests_to_prod"] = ratio(tests, prod)
    row["perf_to_prod"] = ratio(perf, prod)
    row["tests_perf_to_prod"] = ratio(tests + perf, prod)
    return row


def weekly_columns():
    columns = ["week", "week_start", "changes", "prs"]
    for group in cs_classify.GROUPS:
        columns += [f"{group}_added", f"{group}_deleted", f"{group}_net"]
    for scope in SCOPES:
        for cls in CLASSES:
            columns += [f"{scope}_{cls}_added", f"{scope}_{cls}_deleted", f"{scope}_{cls}_net"]
    return columns + ["ci_runs", "ci_failed_runs", "ci_attempts"]


def weekly_rows(records, ci_daily, ci_window=None):
    """Flows per ISO week from the change rows (sums), plus the CI runs of that week.

    Weeks inside the CI window without runs are zero; weeks outside it (or without CI data at all) stay empty.
    The window is (first date, last date), by default the first and last day with a run.
    """
    rows = {}
    for record in records:
        change_values = change_row(record, None)
        key = change_values["week"]
        row = rows.setdefault(key, {column: 0 for column in weekly_columns()})
        row.update(week=key, week_start=week_start(record.change.date))
        row["changes"] += 1
        row["prs"] += 1 if record.change.kind == "pr" else 0
        for column, value in change_values.items():
            if column.endswith("_added") or column.endswith("_deleted"):
                if column in row:
                    row[column] += value
    first_day = min(datetime.fromisoformat(row["week_start"]) for row in rows.values()) if rows else None
    last_day = max(datetime.fromisoformat(row["week_start"]) for row in rows.values()) if rows else None
    day = first_day
    while day is not None and day <= last_day:   # weeks without any change are zeros, not gaps
        key = week_of(day)
        if key not in rows:
            rows[key] = {column: 0 for column in weekly_columns()}
            rows[key].update(week=key, week_start=day.date().isoformat())
        day += timedelta(days=7)
    for row in rows.values():
        for column in list(row):
            if column.endswith("_added"):
                stem = column[:-len("_added")]
                if f"{stem}_net" in row:
                    row[f"{stem}_net"] = row[column] - row[f"{stem}_deleted"]
    if ci_daily is not None:
        by_week = defaultdict(lambda: [0, 0, 0])
        for day in ci_daily:
            moment = datetime.strptime(day["date"], "%Y-%m-%d")
            totals = by_week[week_of(moment)]
            totals[0] += int(day["runs"])
            totals[1] += int(day["runs"]) if day["conclusion"] == "failure" else 0
            totals[2] += int(day["attempts"])
        if ci_window is None and ci_daily:
            days = sorted(day["date"] for day in ci_daily)
            ci_window = (datetime.strptime(days[0], "%Y-%m-%d").date(), datetime.strptime(days[-1], "%Y-%m-%d").date())
        for key, row in rows.items():
            start = datetime.fromisoformat(row["week_start"]).date()
            inside = ci_window is not None and start + timedelta(days=6) >= ci_window[0] and start <= ci_window[1]
            row["ci_runs"], row["ci_failed_runs"], row["ci_attempts"] = (by_week[key] if key in by_week else (0, 0, 0)) if inside else ("", "", "")
    else:
        for row in rows.values():
            row["ci_runs"] = row["ci_failed_runs"] = row["ci_attempts"] = ""
    return [rows[key] for key in sorted(rows)]


def write_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


SPLIT_COLUMNS = ["sha", "date", "pr", "change_kind", "area", "group", "file_kind", "code_class", "files_added",
                 "files_deleted", "files_modified", "lines_added", "lines_deleted"]
STATE_COLUMNS = ["sha", "date", "pr", "area", "group", "file_kind", "code_class", "files", "lines"]
CI_DAILY_COLUMNS = ["date", "workflow", "event", "conclusion", "runs", "attempts"]


def write_all(out, records, ci_daily=None, ci_by_pr=None, ci_window=None):
    """Write every CSV; returns {name: path}."""
    out = Path(out)
    return {
        "changes": write_csv(out / "changes.csv", change_columns(), [change_row(r, ci_by_pr) for r in records]),
        "splits": write_csv(out / "splits.csv", SPLIT_COLUMNS, [row for r in records for row in split_rows(r)]),
        "state": write_csv(out / "state.csv", STATE_COLUMNS, [row for r in records for row in state_rows(r)]),
        "metrics": write_csv(out / "metrics.csv", metrics_columns(), [metrics_row(r) for r in records]),
        "weekly": write_csv(out / "weekly.csv", weekly_columns(), weekly_rows(records, ci_daily, ci_window)),
        **({"ci_daily": write_csv(out / "ci_daily.csv", CI_DAILY_COLUMNS, ci_daily)} if ci_daily is not None else {}),
    }
