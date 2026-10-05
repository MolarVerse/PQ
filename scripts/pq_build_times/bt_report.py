"""The over-time report of one fingerprint: self-contained HTML with inline SVG, and a terminal table.

Standard library only. Every text that came from a snapshot (notes, file names) is HTML-escaped.
"""

import html

import bt_fingerprint

WALL_PANELS = (
    ("cold", "Cold build (build step)"),
    ("noop", "No-op build"),
    ("touch_leaf", "Touch a median source file"),
    ("touch_header_top", "Touch the most widely included header"),
    ("touch_header_median", "Touch a median header"),
)
STEP_PANELS = (
    ("touch_header_top", "Steps rebuilt: most widely included header"),
    ("touch_header_median", "Steps rebuilt: median header"),
    ("touch_leaf", "Steps rebuilt: median source file"),
)
GRAPH_PANELS = (("include_pairs", "Include pairs (object, file)"), ("project_pairs", "Project include pairs"))
WIDTH, HEIGHT = 440, 230
LEFT, RIGHT, TOP, BOTTOM = 58, 14, 30, 38


def format_duration(seconds):
    if seconds is None:
        return "-"
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int(seconds % 3600 // 60):02d}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m {seconds % 60:04.1f}s"
    return f"{seconds:.2f}s" if seconds < 10 else f"{seconds:.1f}s"


def format_count(value):
    return "-" if value is None else f"{value:,}"


def short_date(snapshot):
    return snapshot["created_at"][:10]


def wall_value(snapshot, scenario):
    return snapshot.get("scenarios", {}).get(scenario, {}).get("wall_s")


def steps_value(snapshot, scenario):
    return snapshot.get("scenarios", {}).get(scenario, {}).get("steps")


def graph_value(snapshot, key):
    return snapshot.get("include_graph", {}).get(key)


def nice_ticks(maximum, count=4):
    if maximum <= 0:
        return [0.0, 1.0]
    raw = maximum / count
    magnitude = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 10 ** -(len(str(int(1 / raw))))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    ticks, tick = [], 0.0
    while tick <= maximum + step * 0.999:
        ticks.append(tick)
        tick += step
    return ticks


def svg_chart(title, snapshots, value_of, label_of, baseline_value, formatter):
    """One line chart: a point per snapshot (evenly spaced, in order), the baseline as a dashed line, notes as markers."""
    points = [(index, value_of(snapshot)) for index, snapshot in enumerate(snapshots)]
    drawn = [(i, v) for i, v in points if v is not None]
    head = f'<h3>{html.escape(title)}</h3>'
    if not drawn:
        return f'<div class="panel">{head}<p class="empty">no data</p></div>'
    top = max([v for _, v in drawn] + ([baseline_value] if baseline_value else []))
    ticks = nice_ticks(top * 1.05)
    y_max = ticks[-1] or 1.0
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def x(index):
        return LEFT + (plot_w / 2 if len(snapshots) == 1 else plot_w * index / (len(snapshots) - 1))

    def y(value):
        return TOP + plot_h * (1 - value / y_max)

    parts = [f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="{html.escape(title)}">']
    for tick in ticks:
        parts.append(f'<line x1="{LEFT}" y1="{y(tick):.1f}" x2="{WIDTH - RIGHT}" y2="{y(tick):.1f}" class="grid"/>')
        parts.append(f'<text x="{LEFT - 6}" y="{y(tick) + 4:.1f}" class="axis" text-anchor="end">{html.escape(formatter(tick))}</text>')
    for index in sorted({0, len(snapshots) - 1}):
        anchor = "start" if index == 0 and len(snapshots) > 1 else ("end" if len(snapshots) > 1 else "middle")
        parts.append(f'<text x="{x(index):.1f}" y="{HEIGHT - 18}" class="axis" text-anchor="{anchor}">{short_date(snapshots[index])}</text>')
    if baseline_value:
        parts.append(f'<line x1="{LEFT}" y1="{y(baseline_value):.1f}" x2="{WIDTH - RIGHT}" y2="{y(baseline_value):.1f}" class="baseline"><title>baseline {html.escape(formatter(baseline_value))}</title></line>')
    for index, snapshot in enumerate(snapshots):
        if snapshot.get("note"):
            parts.append(
                f'<g><line x1="{x(index):.1f}" y1="{TOP}" x2="{x(index):.1f}" y2="{TOP + plot_h}" class="note"/>'
                f'<polygon points="{x(index) - 4:.1f},{TOP - 2} {x(index) + 4:.1f},{TOP - 2} {x(index):.1f},{TOP + 6}" class="marker"/>'
                f'<title>{html.escape(short_date(snapshot))}: {html.escape(snapshot["note"])}</title></g>'
            )
    if len(drawn) > 1:
        path = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in drawn)
        parts.append(f'<polyline points="{path}" class="line"/>')
    for index, value in drawn:
        parts.append(f'<circle cx="{x(index):.1f}" cy="{y(value):.1f}" r="3.5" class="point"><title>{html.escape(label_of(snapshots[index], value))}</title></circle>')
    parts.append("</svg>")
    return f'<div class="panel">{head}{"".join(parts)}</div>'


def snapshot_label(snapshot, text):
    git = snapshot.get("git", {})
    commit = git.get("commit", "?") + ("*" if git.get("dirty") else "")
    note = f" - {snapshot['note']}" if snapshot.get("note") else ""
    return f"{snapshot['created_at']}  {commit}  {text}{note}"


STYLE = """
body{font-family:system-ui,sans-serif;margin:24px;color:#1d2330;background:#fff}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}h3{font-size:13px;margin:0 0 4px}
p.meta{color:#566;margin:2px 0;font-size:13px}p.warn{color:#9a3412;font-size:13px}
.grid-wrap{display:flex;flex-wrap:wrap;gap:12px}.panel{border:1px solid #dde2ea;border-radius:6px;padding:8px 10px}
.empty{color:#889;font-size:12px}svg .grid{stroke:#e5e9f0}svg .axis{font-size:10px;fill:#566}
svg .line{fill:none;stroke:#2563eb;stroke-width:2}svg .point{fill:#2563eb}svg .baseline{stroke:#b45309;stroke-dasharray:5 4;stroke-width:1.5}
svg .note{stroke:#9ca3af;stroke-dasharray:2 3}svg .marker{fill:#7c3aed}
table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #dde2ea;padding:3px 8px;text-align:right}th:first-child,td:first-child,td.note{text-align:left}
"""


def render_html(snapshots, baseline, fingerprint_id):
    """The whole report page for the snapshots of one fingerprint (oldest first)."""
    latest = snapshots[-1]
    title = f"PQ local build times - {fingerprint_id}"
    out = [
        f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{STYLE}</style></head><body>',
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="meta">{html.escape(bt_fingerprint.describe(latest["fingerprint"]))}</p>',
        f'<p class="meta">{len(snapshots)} snapshots, {short_date(snapshots[0])} to {short_date(latest)}; '
        f'baseline: {html.escape(baseline["id"]) if baseline else "none yet (pin one with: pqbt baseline set)"}</p>',
        '<p class="meta">Dashed line: baseline. Purple markers: snapshot notes. Hover over points for details.</p>',
    ]
    targets = {tuple(sorted(s.get("targets", {}).items())) for s in snapshots}
    if len(targets) > 1:
        out.append('<p class="warn">The touched files differ between snapshots; the touch panels mix different files.</p>')
    out.append("<h2>Wall time</h2><div class='grid-wrap'>")
    for scenario, title_text in WALL_PANELS:
        out.append(svg_chart(
            title_text, snapshots, lambda s, sc=scenario: wall_value(s, sc),
            lambda s, v: snapshot_label(s, format_duration(v)),
            wall_value(baseline, scenario) if baseline else None, format_duration,
        ))
    out.append("</div><h2>Deterministic figures (do not depend on machine speed)</h2><div class='grid-wrap'>")
    for scenario, title_text in STEP_PANELS:
        out.append(svg_chart(
            title_text, snapshots, lambda s, sc=scenario: steps_value(s, sc),
            lambda s, v: snapshot_label(s, f"{v} steps"),
            steps_value(baseline, scenario) if baseline else None, format_count,
        ))
    for key, title_text in GRAPH_PANELS:
        out.append(svg_chart(
            title_text, snapshots, lambda s, k=key: graph_value(s, k),
            lambda s, v: snapshot_label(s, format_count(v)),
            graph_value(baseline, key) if baseline else None, format_count,
        ))
    out.append("</div><h2>All snapshots</h2><pre>" + html.escape(render_table(snapshots, baseline)) + "</pre></body></html>")
    return "".join(out)


def delta(value, reference):
    if value is None or not reference:
        return ""
    return f" ({(value / reference - 1) * 100:+.0f}%)"


def render_table(snapshots, baseline):
    """A fixed-width table, one row per snapshot, with the change against the baseline in brackets."""
    columns = [("cold", "cold"), ("noop", "no-op"), ("touch_leaf", "leaf"), ("touch_header_top", "top hdr"), ("touch_header_median", "mid hdr")]
    rows = [["date", "commit", *[name for _, name in columns], "note"]]
    for snapshot in snapshots:
        git = snapshot.get("git", {})
        row = [snapshot["created_at"][:16].replace("T", " "), git.get("commit", "?")[:8] + ("*" if git.get("dirty") else "")]
        for scenario, _ in columns:
            value = wall_value(snapshot, scenario)
            reference = wall_value(baseline, scenario) if baseline and baseline["id"] != snapshot["id"] else None
            row.append(format_duration(value) + delta(value, reference) if value is not None else "-")
        row.append(snapshot.get("note", ""))
        rows.append(row)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)
