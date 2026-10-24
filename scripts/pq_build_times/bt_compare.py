"""Compare two snapshots of the same fingerprint.

Deterministic figures (steps rebuilt, include graph) come first because they do not depend on machine speed.
Timings are judged against the noise of the two snapshots (their run-to-run spread) and an absolute floor, so a
difference is only called a change when it is larger than what repeating the same build produces.
"""

import bt_fingerprint
import bt_report

SCENARIO_ORDER = ("cold", "noop", "touch_leaf", "touch_header_top", "touch_header_median")
SCENARIO_NAMES = {
    "cold": "cold build", "noop": "no-op build", "touch_leaf": "touch source file",
    "touch_header_top": "touch top header", "touch_header_median": "touch median header",
}
GRAPH_KEYS = (
    ("objects", "objects"), ("unique_files", "files (all)"), ("include_pairs", "include pairs"),
    ("project_files", "project files"), ("project_pairs", "project include pairs"),
)
# Other processes' CPU during a build. On an idle machine 24-26 s of a 1,220 s clang cold build are not attributed to the
# build's own processes (kernel and I/O work), about 2%; a clearly disturbed run uses a multiple of that.
FOREIGN_MIN_S = 10.0
FOREIGN_SHARE = 0.05
NOISE_FLOOR = 0.05          # relative, whatever the spread says
SINGLE_RUN_FLOOR = 0.10     # relative, when a snapshot has no repetitions to judge noise from
ABSOLUTE_FLOOR_S = 0.05     # seconds; below this nothing is a change (a no-op build takes ~0.01 s)
BUSY_LOAD_FRACTION = 0.25


class CompareError(ValueError):
    pass


def interference(snapshot):
    """[(scenario, foreign CPU seconds, the build's CPU seconds)] for builds during which other processes used a lot of CPU."""
    found = []
    for scenario in SCENARIO_ORDER:
        data = snapshot.get("scenarios", {}).get(scenario)
        if not data or data.get("foreign_cpu_s") is None:
            continue
        if data["foreign_cpu_s"] >= max(FOREIGN_MIN_S, FOREIGN_SHARE * data["cpu_s"]):
            found.append((scenario, data["foreign_cpu_s"], data["cpu_s"]))
    return found


def interference_warnings(snapshot):
    return [f"warning: other processes used about {foreign:.0f} s of CPU during the {SCENARIO_NAMES[scenario]} of {snapshot['id']} "
            f"({foreign / cpu:.0%} of the build's CPU); its timings are inflated, repeat it on an idle machine"
            for scenario, foreign, cpu in interference(snapshot)]


def fingerprint_differences(a, b):
    """The id keys in which two fingerprints differ: [(key, value in a, value in b)]."""
    return [
        (key, a.get(key), b.get(key)) for key in bt_fingerprint.ID_KEYS if a.get(key) != b.get(key)
    ]


def resolve(snapshots, baseline, reference):
    """A snapshot of one series by 'baseline', 'latest', 'previous', an id or a unique id prefix."""
    if not snapshots:
        raise CompareError("no snapshots in this series")
    if reference == "latest":
        return snapshots[-1]
    if reference == "previous":
        if len(snapshots) < 2:
            raise CompareError("there is no snapshot before the latest one")
        return snapshots[-2]
    if reference == "baseline":
        if baseline is None:
            raise CompareError("no baseline pinned for this series (pin one with: pqbt.py baseline set)")
        return baseline
    matches = [snapshot for snapshot in snapshots if snapshot["id"].startswith(reference)]
    exact = [snapshot for snapshot in matches if snapshot["id"] == reference]
    if exact:
        return exact[0]
    if len(matches) == 1:
        return matches[0]
    raise CompareError(f"{'ambiguous' if matches else 'no such'} snapshot '{reference}' in this series")


def noise_threshold(a, b):
    """Relative change below which two timings are not told apart, and whether it is only a floor."""
    spreads = [x.get("spread") for x in (a, b)]
    single = any(len(x.get("runs", [])) < 2 for x in (a, b))
    floor = SINGLE_RUN_FLOOR if single else NOISE_FLOOR
    measured = max((s for s in spreads if s is not None), default=0.0)
    return max(floor, 2 * measured), single


def relative(before, after):
    return None if not before else after / before - 1


def judge(steps_a, steps_b, wall_a, wall_b, threshold):
    """(timing direction, verdict text) of one scenario."""
    significant = abs(wall_b - wall_a) > max(threshold * wall_a, ABSOLUTE_FLOOR_S)
    direction = ("slower" if wall_b > wall_a else "faster") if significant else "same"
    fewer, more = steps_b < steps_a, steps_b > steps_a
    if direction == "same":
        if fewer:
            return direction, "fewer steps, time within noise"
        if more:
            return direction, "more steps, time within noise"
        return direction, "unchanged"
    if direction == "faster":
        if fewer:
            return direction, "improved"
        if more:
            return direction, "more steps but faster: check noise"
        return direction, "faster with the same steps: check noise"
    if more:
        return direction, "regressed"
    if fewer:
        return direction, "fewer steps but slower: check noise"
    return direction, "slower with the same steps: check noise"


def compare(a, b):
    """The comparison of snapshot a (before) and b (after); raises CompareError for different fingerprints."""
    if a["fingerprint_id"] != b["fingerprint_id"]:
        details = "; ".join(f"{k}: {x!r} vs {y!r}" for k, x, y in fingerprint_differences(a["fingerprint"], b["fingerprint"]))
        raise CompareError(
            f"different fingerprints ({a['fingerprint_id']} vs {b['fingerprint_id']}): {details}. "
            "Timings of different machines or configurations are not comparable."
        )
    rows = []
    for scenario in SCENARIO_ORDER:
        sa, sb = a["scenarios"].get(scenario), b["scenarios"].get(scenario)
        if sa is None or sb is None:
            continue
        row = {"scenario": scenario, "a": sa, "b": sb}
        if sa.get("file") != sb.get("file"):
            row.update(comparable=False, verdict=f"not comparable: touched {sa.get('file')} vs {sb.get('file')}")
        else:
            threshold, single = noise_threshold(sa, sb)
            direction, verdict = judge(sa["steps"], sb["steps"], sa["wall_s"], sb["wall_s"], threshold)
            row.update(comparable=True, threshold=threshold, single_run=single, direction=direction, verdict=verdict,
                       wall_change=relative(sa["wall_s"], sb["wall_s"]), cpu_change=relative(sa["cpu_s"], sb["cpu_s"]),
                       steps_change=relative(sa["steps"], sb["steps"]))
        rows.append(row)
    graph = []
    for key, label in GRAPH_KEYS:
        va, vb = a["include_graph"].get(key), b["include_graph"].get(key)
        if va is not None and vb is not None:
            graph.append({"key": key, "label": label, "a": va, "b": vb, "change": relative(va, vb)})
    cores = a["fingerprint"].get("logical_cores") or 1
    busy = [x["id"] for x in (a, b) if x.get("load", {}).get("start", 0) > cores * BUSY_LOAD_FRACTION]
    return {
        "a": a, "b": b, "rows": rows, "graph": graph,
        "digest_changed": a["include_graph"].get("digest") != b["include_graph"].get("digest"),
        "busy": busy,
    }


def percent(change):
    return "" if change is None else f"{change * 100:+.1f}%"


def format_table(rows):
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)


def describe_snapshot(label, snapshot):
    git = snapshot.get("git", {})
    commit = git.get("commit", "?")[:8] + ("*" if git.get("dirty") else "")
    note = f"  \"{snapshot['note']}\"" if snapshot.get("note") else ""
    return f"{label}: {snapshot['id']}  {commit}{note}"


def render(result):
    """The comparison as text."""
    a, b = result["a"], result["b"]
    out = [
        f"fingerprint {a['fingerprint_id']}: {bt_fingerprint.describe(a['fingerprint'])}",
        describe_snapshot("before", a),
        describe_snapshot("after ", b),
        "",
        "Deterministic figures (independent of machine speed)",
    ]
    out.append(format_table(
        [["", "before", "after", "change"]]
        + [[row["label"], bt_report.format_count(row["a"]), bt_report.format_count(row["b"]), percent(row["change"])]
           for row in result["graph"]]
        + [["include graph", a["include_graph"].get("digest", "")[:12], b["include_graph"].get("digest", "")[:12],
            "changed" if result["digest_changed"] else "identical"]]
    ))
    out += ["", "Scenarios (steps are deterministic, times are judged against the noise)"]
    table = [["scenario", "steps", "wall", "change", "cpu", "change", "noise", "verdict"]]
    for row in result["rows"]:
        sa, sb = row["a"], row["b"]
        if not row["comparable"]:
            table.append([SCENARIO_NAMES[row["scenario"]], f"{sa['steps']} -> {sb['steps']}", "", "", "", "", "", row["verdict"]])
            continue
        table.append([
            SCENARIO_NAMES[row["scenario"]],
            f"{sa['steps']} -> {sb['steps']}" + (f" ({percent(row['steps_change'])})" if row["steps_change"] else ""),
            f"{bt_report.format_duration(sa['wall_s'])} -> {bt_report.format_duration(sb['wall_s'])}",
            percent(row["wall_change"]),
            f"{bt_report.format_duration(sa['cpu_s'])} -> {bt_report.format_duration(sb['cpu_s'])}",
            percent(row["cpu_change"]),
            f"+-{row['threshold'] * 100:.0f}%" + ("*" if row["single_run"] else ""),
            row["verdict"],
        ])
    out.append(format_table(table))
    notes = []
    if any(row.get("single_run") for row in result["rows"]):
        notes.append("* a snapshot with a single run per scenario (for example the cold build): noise cannot be measured, so a wider threshold is used.")
    if result["busy"]:
        notes.append(f"warning: the machine was busy when {', '.join(result['busy'])} started; its timings are less reliable.")
    for snapshot in (a, b):
        notes += interference_warnings(snapshot)
    unstable = [SCENARIO_NAMES[row["scenario"]] for row in result["rows"]
                if not (row["a"].get("steps_stable", True) and row["b"].get("steps_stable", True))]
    if unstable:
        notes.append(f"warning: the number of steps varied between repetitions for: {', '.join(unstable)}.")
    return "\n".join(out + ([""] + notes if notes else []))
