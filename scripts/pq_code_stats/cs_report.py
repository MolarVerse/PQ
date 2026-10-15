"""A self-contained HTML report with SVG charts, built from the CSV files (stdlib only, no network)."""

import html
from datetime import datetime, timezone

PALETTE = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")
WIDTH, HEIGHT = 560, 270
LEFT, RIGHT, TOP, BOTTOM = 62, 14, 30, 36

STYLE = """
body{font-family:system-ui,sans-serif;margin:24px;color:#1d2330}h1{font-size:20px;margin:0 0 4px}
h2{font-size:16px;margin:26px 0 6px}h3{font-size:13px;margin:0 0 2px}p.meta{color:#566;margin:2px 0;font-size:13px}
.wrap{display:flex;flex-wrap:wrap;gap:12px}.panel{border:1px solid #dde2ea;border-radius:6px;padding:8px 10px}
.empty{color:#889;font-size:12px}svg .grid{stroke:#e5e9f0}svg .axis{font-size:10px;fill:#566}svg .legend{font-size:11px}
table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #dde2ea;padding:2px 7px;text-align:right}
th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}
"""


def to_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def epoch(text):
    text = text.replace("Z", "+00:00")
    moment = datetime.fromisoformat(text if "T" in text else text + "T00:00:00+00:00")
    return moment.astimezone(timezone.utc).timestamp()


def nice_ticks(low, high, count=4):
    span = (high - low) or 1.0
    raw = span / count
    magnitude = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 10 ** -(len(str(int(1 / raw))))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    tick = (low // step) * step
    ticks = []
    while tick <= high + step * 0.999:
        ticks.append(tick)
        tick += step
    return ticks


def format_value(value):
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e4:
        return f"{value / 1e3:.0f}k"
    if abs(value) >= 1000:
        return f"{value / 1e3:.1f}k"
    return f"{value:.2f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))


def time_chart(title, series, formatter=format_value, zero=True, step=False):
    """Lines over time (step=True holds a value until the next change). series: [(name, [(epoch seconds, value), ...])]."""
    series = [(name, [(t, v) for t, v in points if v is not None]) for name, points in series]
    series = [(name, points) for name, points in series if points]
    head = f"<h3>{html.escape(title)}</h3>"
    if not series:
        return f'<div class="panel">{head}<p class="empty">no data</p></div>'
    times = [t for _, points in series for t, _ in points]
    values = [v for _, points in series for _, v in points]
    t_min, t_max = min(times), max(times)
    low = min(values + ([0] if zero else []))
    ticks = nice_ticks(low, max(values))
    y_min, y_max = ticks[0], ticks[-1]
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def x(t):
        return LEFT + (plot_w / 2 if t_max == t_min else plot_w * (t - t_min) / (t_max - t_min))

    def y(v):
        return TOP + plot_h * (1 - (v - y_min) / ((y_max - y_min) or 1))

    parts = [f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="{html.escape(title)}">']
    for tick in ticks:
        parts.append(f'<line x1="{LEFT}" y1="{y(tick):.1f}" x2="{WIDTH - RIGHT}" y2="{y(tick):.1f}" class="grid"/>')
        parts.append(f'<text x="{LEFT - 6}" y="{y(tick) + 4:.1f}" class="axis" text-anchor="end">{html.escape(formatter(tick))}</text>')
    for index in range(5):
        t = t_min + (t_max - t_min) * index / 4
        label = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m")
        anchor = "start" if index == 0 else ("end" if index == 4 else "middle")
        parts.append(f'<text x="{x(t):.1f}" y="{HEIGHT - 18}" class="axis" text-anchor="{anchor}">{label}</text>')
    for number, (name, points) in enumerate(series):
        color = PALETTE[number % len(PALETTE)]
        if step:
            coordinates = [f"{x(points[0][0]):.1f},{y(points[0][1]):.1f}"]
            for (_, previous), (t, v) in zip(points, points[1:]):
                coordinates += [f"{x(t):.1f},{y(previous):.1f}", f"{x(t):.1f},{y(v):.1f}"]
        else:
            coordinates = [f"{x(t):.1f},{y(v):.1f}" for t, v in points]
        path = " ".join(coordinates)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        parts.append(f'<rect x="{LEFT + 8 + number * 138}" y="{TOP - 20}" width="10" height="10" fill="{color}"/>'
                     f'<text x="{LEFT + 22 + number * 138}" y="{TOP - 11}" class="legend">{html.escape(name)}</text>')
    parts.append("</svg>")
    return f'<div class="panel">{head}{"".join(parts)}</div>'


def total(row, names):
    numbers = [to_number(row.get(name)) for name in names]
    return None if any(number is None for number in numbers) else sum(numbers)


def column(rows, *names):
    """(time, sum of the named columns) per row; a row where one of them is empty gives None."""
    return [(epoch(row["date"]), total(row, names)) for row in rows]


def weekly_column(rows, *names):
    return [(epoch(row["week_start"]), total(row, names)) for row in rows]


def both(scope, what):
    """The header-like and the source column of a scope, for example both("prod", "lines")."""
    return (f"{scope}_header_like_{what}", f"{scope}_source_{what}")


def recent_table(changes, count=25):
    header = ["date", "PR", "title", "files", "+lines", "-lines", "prod hdr", "prod src", "tests", "perf", "CI runs"]
    lines = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>"]
    for row in changes[-count:][::-1]:
        def net(a, d):
            return int(row[a]) - int(row[d])
        cells = [
            row["date"][:10], f"#{row['pr']}" if row["pr"] else row["kind"], html.escape(row["title"][:60]),
            str(int(row["files_added"]) + int(row["files_deleted"]) + int(row["files_modified"])),
            row["lines_added"], row["lines_deleted"],
            f"{net('prod_header_like_added', 'prod_header_like_deleted'):+d}", f"{net('prod_source_added', 'prod_source_deleted'):+d}",
            f"{net('tests_header_like_added', 'tests_header_like_deleted') + net('tests_source_added', 'tests_source_deleted'):+d}",
            f"{net('perf_header_like_added', 'perf_header_like_deleted') + net('perf_source_added', 'perf_source_deleted'):+d}",
            row["ci_runs"] or "-",
        ]
        lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "".join(lines) + "</table>"


def render(metrics, weekly, changes, ci_daily, meta):
    """The report page. metrics/weekly/changes/ci_daily are lists of CSV row dicts."""
    out = [
        f'<!doctype html><html><head><meta charset="utf-8"><title>PQ code statistics</title><style>{STYLE}</style></head><body>',
        "<h1>PQ code statistics</h1>",
        f'<p class="meta">{html.escape(str(meta.get("ref", "")))} at {html.escape(str(meta.get("head", ""))[:12])}; '
        f'{len(changes)} changes ({sum(1 for c in changes if c["kind"] == "pr")} pull requests); '
        f'generated {html.escape(str(meta.get("generated_at", "")))}</p>',
        '<p class="meta">C++ means header, tpp and source files. production = src + include + apps, tests = tests (unit tests, without '
        'integration reference data), perf = benchmarks/perf + benchmarks. header-like = header + tpp.</p>',
        "<h2>State over time</h2><div class='wrap'>",
        time_chart("Production C++ lines", [("header-like", column(metrics, "prod_header_like_lines")), ("source", column(metrics, "prod_source_lines")),
                                           ("total", column(metrics, *both("prod", "lines")))], step=True),
        time_chart("Source share of production C++ lines (higher = less in headers)", [("source share", column(metrics, "prod_source_share"))], formatter=lambda v: f"{v:.2f}", zero=False, step=True),
        time_chart("C++ lines: production, tests, perf", [("production", column(metrics, *both("prod", "lines"))), ("tests", column(metrics, *both("tests", "lines"))),
                                                         ("perf", column(metrics, *both("perf", "lines")))], step=True),
        time_chart("Test and perf C++ lines per production line", [("tests", column(metrics, "tests_to_prod")), ("perf", column(metrics, "perf_to_prod")),
                                                                  ("tests + perf", column(metrics, "tests_perf_to_prod"))], formatter=lambda v: f"{v:.2f}", step=True),
        time_chart("C++ files", [("prod header-like", column(metrics, "prod_header_like_files")), ("prod source", column(metrics, "prod_source_files")),
                                 ("tests", column(metrics, *both("tests", "files"))), ("perf", column(metrics, *both("perf", "files")))], step=True),
        time_chart("CI files", [("workflows", column(metrics, "ci_workflow_files")), ("other .github", column(metrics, "ci_other_files"))], step=True),
        time_chart("CI lines", [("workflows", column(metrics, "ci_workflow_lines")), ("other .github", column(metrics, "ci_other_lines"))], step=True),
        "</div><h2>Flows per week</h2><div class='wrap'>",
        time_chart("Net C++ lines per week", [("production", weekly_column(weekly, *both("prod", "net"))), ("tests", weekly_column(weekly, *both("tests", "net"))),
                                              ("perf", weekly_column(weekly, *both("perf", "net")))], zero=False),
        time_chart("Net production C++ lines per week: header-like vs source", [("header-like", weekly_column(weekly, "prod_header_like_net")),
                                                                              ("source", weekly_column(weekly, "prod_source_net"))], zero=False),
        time_chart("Added production C++ lines per week", [("header-like", weekly_column(weekly, "prod_header_like_added")), ("source", weekly_column(weekly, "prod_source_added"))]),
        time_chart("Pull requests merged per week", [("PRs", weekly_column(weekly, "prs"))]),
    ]
    if ci_daily is not None:
        out.append(time_chart("CI runs per week", [("runs", weekly_column(weekly, "ci_runs")), ("failed", weekly_column(weekly, "ci_failed_runs"))]))
    out.append("</div><h2>Latest changes</h2>" + recent_table(changes) + "</body></html>")
    return "".join(out)
