#!/usr/bin/env python3
"""Generate CI_TIMINGS.md, an overview of CI timings, from the JSONL shards.

Reads every `data/*.jsonl` shard written by collect.py (Python standard
library only) and writes a Markdown page that GitHub renders directly. The
output depends only on the data: "now" is the newest record, so regenerating
from unchanged data produces identical bytes and never causes a commit.

See README.md (Overview report) for how to read the page.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA_VERSION = 1
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
OUTPUT_NAME = "CI_TIMINGS.md"

DEFAULT_WINDOW_DAYS = 14
DEFAULT_REGRESSION_PERCENT = 20.0
DEFAULT_MIN_SAMPLES = 5
MIN_REGRESSION_DELTA_S = 30

PUSH_BRANCH = "dev"
EVENTS = ("push", "pull_request")
EVENT_LABEL = {"push": "push (dev)", "pull_request": "pull_request"}

CHART_WORKFLOW = "BUILD"
CHART_MIN_RUNS = 3


@dataclass(frozen=True)
class Job:
    workflow: str
    name: str
    event: str
    branch: str
    run_id: int
    attempt: int
    conclusion: str
    created: datetime
    completed: datetime
    queue_s: int
    duration_s: int
    eigen_hit: bool | None
    is_rerun: bool
    infra_failure: bool | None


@dataclass(frozen=True)
class Analysis:
    workflow: str
    job: str
    event: str
    branch: str
    attempt: int
    conclusion: str
    created: datetime
    ninja: dict | None
    ccache: dict | None


@dataclass
class LoadStats:
    files: int = 0
    records: int = 0
    ignored: int = 0
    invalid: int = 0
    analyses: int = 0
    analysis_records: list = field(default_factory=list)


def parse_timestamp(text):
    return datetime.strptime(text, TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)


def parse_job(record):
    """Return a Job, or None for a record of another kind/schema version."""
    if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != "job":
        return None
    flags = record["flags"]
    return Job(
        workflow=record["workflow"],
        name=record["job"],
        event=record["event"],
        branch=record["branch"],
        run_id=int(record["run_id"]),
        attempt=int(record["run_attempt"]),
        conclusion=record["conclusion"],
        created=parse_timestamp(record["created_at"]),
        completed=parse_timestamp(record["completed_at"]),
        queue_s=int(record["queue_s"]),
        duration_s=int(record["duration_s"]),
        eigen_hit=flags.get("eigen_cache_hit"),
        is_rerun=bool(flags.get("is_rerun")),
        infra_failure=flags.get("infra_failure"),
    )


def parse_analysis(record):
    """An Analysis from a build-analysis record; None for another schema version."""
    if record.get("schema_version") != SCHEMA_VERSION:
        return None
    return Analysis(
        workflow=record["workflow"],
        job=record["job"],
        event=record["event"],
        branch=record["branch"],
        attempt=int(record["run_attempt"]),
        conclusion=record["conclusion"],
        created=parse_timestamp(record["created_at"]),
        ninja=record["ninja"],
        ccache=record["ccache"],
    )


def load_jobs(data_dir):
    """Read all shards. Returns (jobs, LoadStats); bad lines are counted, not fatal."""
    jobs = []
    stats = LoadStats()
    for shard in sorted(Path(data_dir).glob("*.jsonl")):
        stats.files += 1
        with open(shard, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                stats.records += 1
                try:
                    record = json.loads(line)
                    if isinstance(record, dict) and record.get("kind") == "build-analysis":
                        analysis = parse_analysis(record)
                        if analysis is None:
                            stats.ignored += 1
                        else:
                            stats.analyses += 1
                            stats.analysis_records.append(analysis)
                        continue
                    job = parse_job(record)
                except (ValueError, KeyError, TypeError, AttributeError):
                    stats.invalid += 1
                    continue
                if job is None:
                    stats.ignored += 1
                else:
                    jobs.append(job)
    return jobs, stats


def exclusion_reason(job):
    """Why a job is left out of the typical-duration statistics (None = kept).

    Each job is counted under the first matching reason.
    """
    if job.event not in EVENTS:
        return "event other than push or pull_request"
    if job.event == "push" and job.branch != PUSH_BRANCH:
        return f"push to a branch other than {PUSH_BRANCH}"
    if job.conclusion == "cancelled":
        return "cancelled"
    if job.infra_failure is True:
        return "infrastructure failure"
    if job.is_rerun:
        return "rerun (attempt > 1)"
    if job.conclusion != "success":
        return "failed or other non-success conclusion"
    return None


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare(current, previous, *, regression_percent, min_samples):
    """Return (relative change or None, is_regression) for two median samples."""
    cur_n, cur_median = current
    prev_n, prev_median = previous
    if cur_n < min_samples or prev_n < min_samples or not prev_median:
        return None, False
    change = cur_median / prev_median - 1
    regression = (
        change * 100 > regression_percent
        and cur_median - prev_median >= MIN_REGRESSION_DELTA_S
    )
    return change, regression


def format_duration(seconds):
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m {seconds:02d}s"


def format_change(change, regression):
    if change is None:
        return "n/a"
    text = f"{change * 100:+.0f}%"
    return f"{text} **regression**" if regression else text


def format_percent(part, whole):
    return f"{100 * part / whole:.0f}%" if whole else "-"


def iso_week(moment):
    year, week, _ = moment.isocalendar()
    return f"{year}-W{week:02d}"


def cell(text):
    return str(text).replace("|", "\\|")


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return lines


@dataclass(frozen=True)
class Run:
    workflow: str
    event: str
    start: datetime
    end: datetime
    last_job: str

    @property
    def wall_s(self):
        return (self.end - self.start).total_seconds()


def is_structural(job):
    """`changes` and `*-gate` jobs exist in every run; they are not real work."""
    return job.name == "changes" or job.name.endswith("-gate")


def collect_runs(jobs):
    """Wall-clock per run. Returns (runs, Counter of run exclusions).

    Kept: first attempt, allowed event, every job successful, and at least one
    real job (a run where the paths filter skipped everything only has the
    structural jobs and would drag the medians down).
    """
    by_run = defaultdict(list)
    for job in jobs:
        if job.attempt != 1 or job.event not in EVENTS:
            continue
        if job.event == "push" and job.branch != PUSH_BRANCH:
            continue
        by_run[job.run_id].append(job)
    runs = []
    dropped = Counter()
    for members in by_run.values():
        if any(job.conclusion != "success" for job in members):
            dropped["run with a non-successful job"] += 1
            continue
        if all(is_structural(job) for job in members):
            dropped["run where the paths filter skipped every real job"] += 1
            continue
        last = max(members, key=lambda job: (job.completed, job.name))
        first = members[0]
        runs.append(
            Run(
                workflow=first.workflow,
                event=first.event,
                start=min(job.created for job in members),
                end=last.completed,
                last_job=last.name,
            )
        )
    return runs, dropped


class Windows:
    """Current window [mid, end] and the equally long previous window [start, mid)."""

    def __init__(self, newest, days):
        self.mid = newest - timedelta(days=days)
        self.start = self.mid - timedelta(days=days)
        self.newest = newest

    def current(self, moment):
        return moment >= self.mid

    def previous(self, moment):
        return self.start <= moment < self.mid


def summarise(values):
    if not values:
        return 0, None, None
    return len(values), percentile(values, 50), percentile(values, 90)


def job_rows(kept, windows, options):
    """One row per (workflow, job, event) with data in the current window."""
    current = defaultdict(lambda: ([], []))
    previous = defaultdict(list)
    for job in kept:
        key = (job.workflow, job.name, job.event)
        if windows.current(job.created):
            current[key][0].append(job.duration_s)
            current[key][1].append(job.queue_s)
        elif windows.previous(job.created):
            previous[key].append(job.duration_s)
    rows = []
    for key in sorted(current):
        durations, queues = current[key]
        n, median, p90 = summarise(durations)
        prev_n, prev_median, _ = summarise(previous.get(key, []))
        change, regression = compare(
            (n, median),
            (prev_n, prev_median),
            regression_percent=options.regression_percent,
            min_samples=options.min_samples,
        )
        _, queue_median, queue_p90 = summarise(queues)
        rows.append(
            {
                "workflow": key[0],
                "job": key[1],
                "event": key[2],
                "n": n,
                "median": median,
                "p90": p90,
                "prev_median": prev_median,
                "prev_n": prev_n,
                "change": change,
                "regression": regression,
                "queue_median": queue_median,
                "queue_p90": queue_p90,
            }
        )
    return rows


def path_rows(runs, windows, options):
    """One row per (workflow, event): run wall-clock and its usual last job."""
    current = defaultdict(list)
    previous = defaultdict(list)
    for run in runs:
        key = (run.workflow, run.event)
        if windows.current(run.start):
            current[key].append(run)
        elif windows.previous(run.start):
            previous[key].append(run)
    rows = []
    for key in sorted(current):
        n, median, p90 = summarise([run.wall_s for run in current[key]])
        prev_n, prev_median, _ = summarise([run.wall_s for run in previous.get(key, [])])
        change, regression = compare(
            (n, median),
            (prev_n, prev_median),
            regression_percent=options.regression_percent,
            min_samples=options.min_samples,
        )
        last_jobs = Counter(run.last_job for run in current[key])
        name, count = sorted(last_jobs.items(), key=lambda item: (-item[1], item[0]))[0]
        rows.append(
            {
                "workflow": key[0],
                "event": key[1],
                "n": n,
                "median": median,
                "p90": p90,
                "prev_median": prev_median,
                "change": change,
                "regression": regression,
                "last_job": name,
                "last_share": format_percent(count, n),
            }
        )
    return rows


def eigen_weeks(kept):
    """Per ISO week and event: [hits, known] of the Eigen cache flag."""
    weeks = defaultdict(lambda: {event: [0, 0] for event in EVENTS})
    for job in kept:
        if job.eigen_hit is None:
            continue
        counts = weeks[iso_week(job.created)][job.event]
        counts[1] += 1
        counts[0] += 1 if job.eigen_hit else 0
    return dict(sorted(weeks.items()))


def weekly_medians(runs, workflow, event):
    weeks = defaultdict(list)
    for run in runs:
        if run.workflow == workflow and run.event == event:
            weeks[iso_week(run.start)].append(run.wall_s)
    return {
        week: percentile(values, 50)
        for week, values in sorted(weeks.items())
        if len(values) >= CHART_MIN_RUNS
    }


def chart(title, axis_label, labels, values, maximum):
    """A Mermaid xychart-beta line chart (rendered natively by GitHub)."""
    years = {label[:4] for label in labels}
    shown = [label if len(years) > 1 else label[5:] for label in labels]
    return [
        "```mermaid",
        "xychart-beta",
        f'    title "{title}"',
        "    x-axis [" + ", ".join(f'"{label}"' for label in shown) + "]",
        f'    y-axis "{axis_label}" 0 --> {maximum}',
        "    line [" + ", ".join(f"{value:g}" for value in values) + "]",
        "```",
    ]


def chart_maximum(values, step):
    return int(max(values) // step + 1) * step


PCH_COUNTER = "could_not_use_precompiled_header"
CACHE_FULL_RATIO = 0.95
SLOWEST_SHOWN = 10


def analysis_kept(analysis):
    """Build analyses compared like jobs: first attempts, not cancelled, allowed events."""
    if analysis.event not in EVENTS or analysis.attempt != 1 or analysis.conclusion == "cancelled":
        return False
    return analysis.event != "push" or analysis.branch == PUSH_BRANCH


def ccache_figures(analysis):
    """(hit rate, PCH-blocked calls, PCH-blocked share, cache full) of one build, or None."""
    ccache = analysis.ccache
    if not ccache:
        return None
    cacheable = ccache["hits"] + ccache["misses"]
    blocked = ccache["counters"].get(PCH_COUNTER, 0)
    counters = ccache["counters"]
    size, limit = counters.get("cache_size_kibibyte"), counters.get("max_cache_size_kibibyte")
    return (
        ccache["hits"] / cacheable if cacheable else None,
        blocked,
        blocked / (cacheable + blocked) if cacheable + blocked else None,
        size >= CACHE_FULL_RATIO * limit if size is not None and limit else None,
    )


def fmt_seconds(value):
    if value is None:
        return "-"
    return f"{value:.1f}s" if value < 60 else format_duration(value)


def format_points(current, previous, current_n, previous_n, min_samples):
    if current is None or previous is None or current_n < min_samples or previous_n < min_samples:
        return "n/a"
    return f"{(current - previous) * 100:+.1f} pp"


def median_of(values):
    values = [value for value in values if value is not None]
    return percentile(values, 50) if values else None


def ccache_rows(analyses, windows, options):
    current, previous = defaultdict(list), defaultdict(list)
    for analysis in analyses:
        figures = ccache_figures(analysis)
        if figures is None:
            continue
        key = (analysis.workflow, analysis.job, analysis.event)
        if windows.current(analysis.created):
            current[key].append(figures)
        elif windows.previous(analysis.created):
            previous[key].append(figures)
    rows = []
    for key in sorted(current):
        now = current[key]
        rates = [f[0] for f in now if f[0] is not None]
        before = [f[0] for f in previous.get(key, []) if f[0] is not None]
        full = [f[3] for f in now if f[3] is not None]
        rows.append(
            [
                f"{key[0]} / {key[1]}",
                EVENT_LABEL[key[2]],
                len(now),
                format_percent(median_of(rates) or 0, 1) if rates else "-",
                format_percent(median_of(before) or 0, 1) if before else "-",
                format_points(median_of(rates), median_of(before), len(rates), len(before), options.min_samples),
                f"{median_of([f[1] for f in now]):.0f}",
                format_percent(median_of([f[2] for f in now]) or 0, 1) if any(f[2] is not None for f in now) else "-",
                f"{sum(full)}/{len(full)}" if full else "-",
            ]
        )
    return rows


def weekly_blocked_share(analyses):
    """ISO week -> median share of compiler calls blocked by the PCH (>= CHART_MIN_RUNS builds)."""
    weeks = defaultdict(list)
    for analysis in analyses:
        figures = ccache_figures(analysis)
        if figures is not None and figures[2] is not None:
            weeks[iso_week(analysis.created)].append(figures[2])
    return {w: median_of(v) for w, v in sorted(weeks.items()) if len(v) >= CHART_MIN_RUNS}


def ninja_rows(analyses, windows):
    """Complete ninja builds per (workflow, job, event) in the current window."""
    groups = defaultdict(list)
    skipped = 0
    for analysis in analyses:
        ninja = analysis.ninja
        if not ninja or "error" in ninja or not windows.current(analysis.created):
            continue
        if ninja["complete"] is not True:
            skipped += 1
            continue
        groups[(analysis.workflow, analysis.job, analysis.event)].append(ninja)
    rows = []
    for key in sorted(groups):
        builds = groups[key]
        link_share = [b["by_kind"]["link"]["cpu_s"] / b["cpu_s"] for b in builds if b["cpu_s"]]
        rows.append(
            [
                f"{key[0]} / {key[1]}",
                EVENT_LABEL[key[2]],
                len(builds),
                fmt_seconds(median_of([b["wall_s"] for b in builds])),
                fmt_seconds(median_of([b["cpu_s"] for b in builds])),
                f"{median_of([b['parallelism'] for b in builds]):.1f}" if any(b["parallelism"] for b in builds) else "-",
                fmt_seconds(median_of([b["tail_after_compile_s"] for b in builds])),
                format_percent(median_of(link_share) or 0, 1) if link_share else "-",
            ]
        )
    return rows, skipped


def slowest_steps(analyses, windows):
    """Per (workflow, job): rows of the slowest steps of dev pushes plus build counts."""
    builds = defaultdict(list)
    for analysis in analyses:
        if analysis.event != "push" or not windows.current(analysis.created):
            continue
        if analysis.ninja and "error" not in analysis.ninja:
            builds[(analysis.workflow, analysis.job)].append(analysis.ninja)
    result = {}
    for key, members in sorted(builds.items()):
        seen = defaultdict(list)
        kinds = {}
        for ninja in members:
            for step in ninja["slowest"]:
                seen[step["target"]].append(step["seconds"])
                kinds[step["target"]] = step["kind"]
        ranked = sorted(seen.items(), key=lambda item: (-median_of(item[1]), item[0]))[:SLOWEST_SHOWN]
        complete = sum(1 for ninja in members if ninja["complete"] is True)
        result[key] = (
            len(members),
            complete,
            [
                [f"`{target}`", kinds[target], f"{len(values)}/{len(members)}", fmt_seconds(median_of(values)), fmt_seconds(max(values))]
                for target, values in ranked
            ],
        )
    return result


def build_analysis_lines(analyses, windows, options):
    lines = ["## Build analysis", ""]
    kept = [analysis for analysis in analyses if analysis_kept(analysis)]
    if not kept:
        lines += [
            "No build analysis records yet. They come from the `build-timings-*` artifacts of "
            "`BUILD` and `LINT` runs, which the collector ingests from the day those jobs started "
            "uploading them (nothing can be backfilled).",
            "",
        ]
        return lines
    lines += [
        f"From {len(kept):,} build analysis records (first attempts, not cancelled; `push` only "
        f"to `{PUSH_BRANCH}`), over the current window.",
        "",
        "### ccache",
        "",
        "Per job: the median hit rate of cacheable calls, and the calls that never reach ccache "
        f"because they use the precompiled header (`{PCH_COUNTER}`; see #732). "
        "\"Cache full\" counts builds whose cache was at least 95% of its size limit, which "
        "means it is evicting entries.",
        "",
    ]
    rows = ccache_rows(kept, windows, options)
    if rows:
        lines += table(
            ["Job", "Event", "Builds", "Hit rate", "Previous", "Change", "PCH-blocked calls", "PCH-blocked share", "Cache full"],
            rows,
        )
    else:
        lines.append("No ccache statistics in the current window.")
    lines.append("")
    shares = weekly_blocked_share(kept)
    if shares:
        values = [round(100 * value, 1) for value in shares.values()]
        lines += chart(
            "Share of compiler calls blocked by the PCH, weekly median (%)", "percent", list(shares), values, 100
        )
        lines.append("")

    rows, skipped = ninja_rows(kept, windows)
    lines += ["### Ninja builds", ""]
    lines += [
        "Complete builds only. `Tail` is the time from the last compile step finishing to the "
        "end of the build (linking); `Link share` is the part of all step time spent linking; "
        "`Parallelism` is CPU time divided by wall time. A large tail with low parallelism "
        "means a bigger runner would not help much.",
        "",
    ]
    if rows:
        lines += table(["Job", "Event", "Builds", "Wall", "CPU", "Parallelism", "Tail", "Link share"], rows)
    else:
        lines.append("No complete ninja builds in the current window.")
    if skipped:
        lines += [
            "",
            f"Left out because they are not full builds: {skipped:,} with `complete` false or unknown "
            "(for example `lint`, which builds with `-k 0` and tolerates errors).",
        ]
    lines.append("")

    lines += [f"### Slowest build steps on `{PUSH_BRANCH}`", ""]
    lines += [
        "From `push` builds in the current window. Each build records only its 20 slowest steps, "
        "so \"In\" says in how many builds a step made that list. With a warm ccache the compiles "
        "that really ran are the uncacheable ones, so this is what is rebuilt every time, not a "
        "cold-build ranking.",
        "",
    ]
    steps = slowest_steps(kept, windows)
    if not steps:
        lines += [f"No ninja builds of pushes to `{PUSH_BRANCH}` in the current window.", ""]
    for (workflow, job), (count, complete, rows) in steps.items():
        lines += [f"#### {workflow} / {job} ({complete} of {count} builds complete)", ""]
        lines += table(["Step", "Kind", "In", "Median", "Max"], rows)
        lines.append("")
    return lines


def render(jobs, stats, options):
    lines = ["# CI timings", ""]
    if not jobs:
        lines += [
            "No timing data found yet. This page is generated by "
            "`.github/ci-metrics/report.py` from `.github/ci-metrics/data/*.jsonl`.",
            "",
        ]
        return "\n".join(lines)

    newest = max(job.created for job in jobs)
    oldest = min(job.created for job in jobs)
    windows = Windows(newest, options.window_days)
    reasons = Counter(exclusion_reason(job) for job in jobs)
    kept = [job for job in jobs if exclusion_reason(job) is None]
    runs, dropped_runs = collect_runs(jobs)
    jobs_table = job_rows(kept, windows, options)
    paths_table = path_rows(runs, windows, options)

    lines += [
        "Generated by `.github/ci-metrics/report.py` from the job records in "
        "`.github/ci-metrics/data/`. Do not edit by hand; it is regenerated after "
        "every collector run.",
        "",
        f"- Data: {len(jobs):,} job records from {oldest:%Y-%m-%d} to {newest:%Y-%m-%d}.",
        f"- Current window: the {options.window_days} days up to {newest:%Y-%m-%d %H:%M} UTC "
        f"(from {windows.mid:%Y-%m-%d %H:%M}). Previous window: the "
        f"{options.window_days} days before it.",
        "- `push (dev)` runs are compared apart from `pull_request` runs: they read "
        "different cache scopes (pull requests can read `dev`'s caches, `dev` pushes cannot "
        "read caches created on pull request branches).",
        "- Durations are the time a job ran; queue time (waiting for a runner) is shown "
        "separately. Medians and p90 (90th percentile) are over the jobs in the window.",
        "",
    ]

    flagged = [row for row in paths_table + jobs_table if row["regression"]]
    lines += ["## Regressions", ""]
    lines += [
        f"A median that rose by more than {options.regression_percent:g}% and at least "
        f"{MIN_REGRESSION_DELTA_S} seconds against the previous window, with at least "
        f"{options.min_samples} samples in both windows.",
        "",
    ]
    if flagged:
        for row in flagged:
            subject = f"{row['workflow']} / {row['job']}" if "job" in row else f"{row['workflow']} (whole run)"
            lines.append(
                f"- {subject}, {EVENT_LABEL[row['event']]}: median "
                f"{format_duration(row['prev_median'])} to {format_duration(row['median'])} "
                f"({row['change'] * 100:+.0f}%, {row['n']} runs)"
            )
    else:
        lines.append("None.")
    lines.append("")

    lines += ["## Wall-clock per workflow (critical path)", ""]
    lines += [
        "Time from the first job of a run being created to the last job finishing "
        "(includes queue time and job dependencies such as `changes` before `build`). "
        "Only first-attempt runs whose jobs all succeeded and in which at least one real job "
        "ran (see Exclusions). \"Usually last\" is the "
        "job that finished last most often, i.e. the job a pull request waits for.",
        "",
    ]
    lines += table(
        ["Workflow", "Event", "Runs", "Median", "p90", "Previous median", "Change", "Usually last"],
        [
            [
                row["workflow"],
                EVENT_LABEL[row["event"]],
                row["n"],
                format_duration(row["median"]),
                format_duration(row["p90"]),
                format_duration(row["prev_median"]),
                format_change(row["change"], row["regression"]),
                f"{row['last_job']} ({row['last_share']})",
            ]
            for row in paths_table
        ],
    )
    lines.append("")

    lines += ["## Trend", ""]
    lines += [f"Weekly median wall-clock of {CHART_WORKFLOW} runs; weeks with fewer than {CHART_MIN_RUNS} runs are omitted.", ""]
    drawn = False
    for event in EVENTS:
        medians = weekly_medians(runs, CHART_WORKFLOW, event)
        if not medians:
            continue
        drawn = True
        minutes = [round(value / 60, 1) for value in medians.values()]
        lines += chart(
            f"{CHART_WORKFLOW} wall-clock, {EVENT_LABEL[event]}, weekly median (minutes)",
            "minutes",
            list(medians),
            minutes,
            chart_maximum(minutes, 5),
        )
        lines.append("")
    if not drawn:
        lines += [f"Not enough {CHART_WORKFLOW} runs yet for a trend chart.", ""]

    lines += ["## Jobs", ""]
    lines += [
        f"Jobs with no run in the current window are omitted. \"Previous median\" is n/a "
        f"with fewer than {options.min_samples} samples in either window.",
        "",
    ]
    for workflow in sorted({row["workflow"] for row in jobs_table}):
        lines += [f"### {workflow}", ""]
        lines += table(
            [
                "Job", "Event", "Runs", "Median", "p90", "Previous median", "Change",
                "Queue median", "Queue p90",
            ],
            [
                [
                    row["job"],
                    EVENT_LABEL[row["event"]],
                    row["n"],
                    format_duration(row["median"]),
                    format_duration(row["p90"]),
                    format_duration(row["prev_median"]),
                    format_change(row["change"], row["regression"]),
                    format_duration(row["queue_median"]),
                    format_duration(row["queue_p90"]),
                ]
                for row in jobs_table
                if row["workflow"] == workflow
            ],
        )
        lines.append("")

    lines += build_analysis_lines(stats.analysis_records, windows, options)

    lines += ["## Eigen cache hit rate", ""]
    weeks = eigen_weeks(kept)
    if weeks:
        lines += [
            "Share of jobs that restored the Eigen source from the cache instead of cloning "
            "it from GitLab (only jobs that have the `Cache Eigen source` step; see "
            "`eigen_cache_hit` in SCHEMA.md). Weeks are ISO weeks of the job's creation.",
            "",
        ]
        rows = []
        for week, per_event in weeks.items():
            total_hits = sum(counts[0] for counts in per_event.values())
            total_known = sum(counts[1] for counts in per_event.values())
            rows.append(
                [week]
                + [
                    f"{format_percent(*per_event[event])} ({per_event[event][1]})"
                    if per_event[event][1]
                    else "-"
                    for event in EVENTS
                ]
                + [f"{format_percent(total_hits, total_known)} ({total_known})"]
            )
        lines += table(
            ["Week", "push (dev), hit rate (jobs)", "pull_request, hit rate (jobs)", "All, hit rate (jobs)"],
            rows,
        )
        lines.append("")
        labels = list(weeks)
        rates = [
            round(
                100
                * sum(counts[0] for counts in per_event.values())
                / sum(counts[1] for counts in per_event.values())
            )
            for per_event in weeks.values()
        ]
        lines += chart("Eigen cache hit rate, all events, weekly (%)", "percent", labels, rates, 100)
    else:
        lines.append("No jobs with the Eigen cache steps yet.")
    lines.append("")

    lines += ["## Exclusions", ""]
    lines += [
        "Left out of the medians, p90 and the trend (each job counted once, under the first "
        "matching reason):",
        "",
    ]
    excluded = sorted((reason, count) for reason, count in reasons.items() if reason)
    lines += table(["Reason", "Jobs"], [[reason, f"{count:,}"] for reason, count in excluded]) if excluded else ["None."]
    lines += [
        "",
        f"Kept: {reasons.get(None, 0):,} of {len(jobs):,} jobs. Jobs the collector "
        "skipped (`skipped` conclusion) were never recorded.",
        "",
        "Whole-run wall-clock additionally leaves out these runs (counted once each):",
        "",
    ]
    lines += (
        table(["Reason", "Runs"], [[reason, f"{count:,}"] for reason, count in sorted(dropped_runs.items())])
        if dropped_runs
        else ["None."]
    )
    lines += [
        "",
        f"The wall-clock figures cover {len(runs):,} runs. A run where only `changes` and "
        "`*-gate` jobs ran is one in which the paths filter skipped all real jobs (for "
        "example a pull request without C++ changes).",
    ]
    if stats.ignored or stats.invalid:
        lines += [
            "",
            f"Unread records: {stats.ignored:,} of an unknown schema version or kind, "
            f"{stats.invalid:,} malformed.",
        ]
    lines.append("")
    return "\n".join(lines)


def write_if_changed(path, text):
    """Atomic write; returns True if the file content changed."""
    path = Path(path)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return True


def parse_args(argv):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate the CI timings overview page.")
    parser.add_argument("--data-dir", type=Path, default=here / "data")
    parser.add_argument("--out", type=Path, default=None, help=f"default: <data-dir>/{OUTPUT_NAME}")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--regression-percent", type=float, default=DEFAULT_REGRESSION_PERCENT)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    options = parser.parse_args(argv)
    if options.window_days < 1 or options.min_samples < 1 or options.regression_percent < 0:
        parser.error("--window-days and --min-samples must be >= 1, --regression-percent >= 0")
    if options.out is None:
        options.out = options.data_dir / OUTPUT_NAME
    return options


def main(argv=None):
    options = parse_args(sys.argv[1:] if argv is None else argv)
    if not options.data_dir.is_dir():
        print(f"error: data directory not found: {options.data_dir}", file=sys.stderr)
        return 1
    jobs, stats = load_jobs(options.data_dir)
    changed = write_if_changed(options.out, render(jobs, stats, options))
    verb = "wrote" if changed else "unchanged:"
    print(f"{verb} {options.out} ({len(jobs):,} jobs from {stats.files} shards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
