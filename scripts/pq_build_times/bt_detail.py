"""Detail of one snapshot: where the build time goes and what a change costs.

The numbers come from what Ninja records (.ninja_log: when every step ran, .ninja_deps: which files every object
includes) and from git (in how many PRs a file changed lately). Nothing here needs a special compiler; parse versus
code generation time per header needs clang's -ftime-trace and is not part of this.

The central figure is the *rebuild cost* of a header: the compile CPU time of every translation unit that includes
it, which is what touching the header makes you wait for. Multiplied by how often the header changed it is the
*burden*, the compile CPU you actually burn on it.
"""

import hashlib
import os
from collections import Counter, defaultdict

import bt_fingerprint
import bt_ninja

SCHEMA_VERSION = 1
KIND = "local-build-detail"
TOP_HEADERS_STORED = 300
CHURN_DAYS = 180
CONCURRENCY_BUCKETS = 10
TAIL_FRACTION = 0.10
LINK_KINDS = ("library", "executable")


def module_of(source):
    """The part of the code base a source file belongs to (src/<module>, tests, perf, third-party, ...)."""
    if not source:
        return "other"
    parts = source.split("/")
    head = parts[0]
    if head in ("external", "_deps"):
        return "third-party"
    if head == "tests":
        return "tests"
    if head == "benchmarks":
        return "perf" if len(parts) > 1 and parts[1] == "perf" else "benchmarks"
    if head == "apps":
        return "apps"
    if head == "src":
        return f"src/{parts[1]}" if len(parts) > 2 else "src"
    return "other"


def group_of(module):
    return "src (project)" if module == "src" or module.startswith("src/") else module


def git_churn(source_root, days=CHURN_DAYS, run=bt_fingerprint.default_run):
    """({file: number of changes of the branch that touched it}, ref) over the last days, or (None, None).

    A change is a merged PR, another merge or a direct commit on the first-parent history, so a PR counts once
    however many commits it had.
    """
    for ref in ("origin/dev", "dev", "HEAD"):
        if not run(["git", "-C", source_root, "rev-parse", "--verify", "--quiet", ref + "^{commit}"]).strip():
            continue
        text = run(["git", "-c", "core.quotepath=false", "-C", source_root, "log", "--first-parent", "-m",
                    f"--since={days}.days", "--name-only", "--format=%x01", ref])
        counts = Counter()
        for block in text.split("\x01")[1:]:
            counts.update({line.strip() for line in block.splitlines() if line.strip()})
        return counts, ref
    return None, None


def build_steps(entries, deps, source_root, build_dir, churn):
    """One dict per executed command: output, kind, seconds, start and end offset, and for compiles more."""
    steps = bt_ninja.unique_steps(entries)
    origin = min((entry.start for entry in steps), default=0)
    result = []
    for entry in sorted(steps, key=lambda e: (e.start, e.output)):
        kind = bt_ninja.kind_of_output(entry.output)
        step = {"o": entry.output, "k": kind, "s": round((entry.end - entry.start) / 1000, 3),
                "a": round((entry.start - origin) / 1000, 3), "e": round((entry.end - origin) / 1000, 3)}
        if kind == "compile":
            source = bt_ninja.source_of_object(entry.output)
            step["src"] = source
            dependencies = deps.get(entry.output)
            if dependencies is not None:
                unique = set(dependencies)
                step["n"] = len(unique)
                repo = {rel[0] for rel in (bt_ninja.repo_relative(d, source_root, build_dir) for d in unique) if rel}
                step["p"] = len(repo - {source})
            if churn is not None and source and not source.startswith(("external/", "_deps/")):
                step["ch"] = churn.get(source, 0)   # third-party sources are not ours to change
        result.append(step)
    return result


def header_costs(deps, seconds, source_root, build_dir):
    """(objects, fan-in, compile CPU of the includers, external?, includer set id) per header, over the objects built.

    The includer set id is a short hash of the objects that include the header: headers with the same id are
    included by exactly the same files, which is how the report folds them into one row.
    """
    fan_in, cpu, external = Counter(), defaultdict(float), {}
    includers = defaultdict(list)
    objects = 0
    for target, dependencies in deps.items():
        if target not in seconds:
            continue
        objects += 1
        seen = set()
        for dependency in dependencies:
            found = bt_ninja.repo_relative(dependency, source_root, build_dir)
            if found is None or found[0] in seen or found[0].endswith(bt_ninja.SOURCE_SUFFIXES):
                continue
            seen.add(found[0])
            fan_in[found[0]] += 1
            cpu[found[0]] += seconds[target]
            external[found[0]] = found[1]
            includers[found[0]].append(target)
    sets = {name: hashlib.sha1("\n".join(sorted(objs)).encode("utf-8")).hexdigest()[:10] for name, objs in includers.items()}
    return objects, fan_in, cpu, external, sets


def concurrency(steps, buckets=CONCURRENCY_BUCKETS):
    """The average number of steps running in each tenth of the build's wall time."""
    wall = max((step["e"] for step in steps), default=0.0)
    if wall <= 0:
        return []
    width = wall / buckets
    running = [0.0] * buckets
    for step in steps:
        for index in range(buckets):
            low, high = index * width, (index + 1) * width
            running[index] += max(0.0, min(step["e"], high) - max(step["a"], low))
    return [round(value / width, 2) for value in running]


def make_detail(snapshot_id, fingerprint_id, jobs, source_name, cold_entries, deps, touches, source_root, build_dir,
                churn, churn_ref, churn_days=CHURN_DAYS):
    """The detail of a snapshot. touches: {scenario: {"file", "wall_s", "entries"}}."""
    steps = build_steps(cold_entries, deps, source_root, build_dir, churn)
    seconds = {step["o"]: step["s"] for step in steps if step["k"] == "compile"}
    objects, fan_in, cpu, external, sets = header_costs(deps, seconds, source_root, build_dir)
    ranked = sorted(cpu, key=lambda name: (-cpu[name], name))[:TOP_HEADERS_STORED]
    headers = []
    for name in ranked:
        header = {"f": name, "fan_in": fan_in[name], "cpu_s": round(cpu[name], 2), "x": external[name], "g": sets[name]}
        if churn is not None:
            # a header of a submodule changes when the submodule pointer is bumped, which is a change of "external/<name>"
            header["ch"] = churn.get("/".join(name.split("/")[:2]) if external[name] else name, 0)
        headers.append(header)
    touch_detail = {}
    for scenario, touch in touches.items():
        entries = bt_ninja.unique_steps(touch["entries"])
        touch_detail[scenario] = {
            "file": touch["file"], "wall_s": touch["wall_s"],
            "steps": sorted(({"o": e.output, "k": bt_ninja.kind_of_output(e.output), "s": round((e.end - e.start) / 1000, 3)}
                             for e in entries), key=lambda s: (-s["s"], s["o"])),
        }
    return {
        "schema_version": SCHEMA_VERSION, "kind": KIND, "id": snapshot_id, "fingerprint_id": fingerprint_id,
        "source": source_name, "jobs": jobs, "objects": objects, "steps": steps, "headers": headers,
        "fan_in": {name: fan_in[name] for name in sorted(fan_in)},   # all headers, exact: the top list is cut by timing
        "touch": touch_detail, "churn_days": churn_days if churn is not None else None, "churn_ref": churn_ref,
    }


# --- the report ---------------------------------------------------------------------------------------------------

def duration(seconds):
    if seconds is None:
        return "-"
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int(seconds % 3600 // 60):02d}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{seconds:.1f}s" if seconds >= 0.1 else f"{seconds:.2f}s"


def percent(part, whole):
    return f"{100 * part / whole:.1f}%" if whole else "-"


def table(rows, left=()):
    """Fixed-width table; the first row is the header. Columns in `left` (and the last one) are left aligned."""
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    last = len(widths) - 1

    def cell(row, i):
        text = str(row[i])
        return text.ljust(widths[i]) if i in left or i == last else text.rjust(widths[i])

    lines = ["  ".join(cell(row, i) for i in range(len(widths))).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)


def concentration(compiles):
    """How many files make up half and four fifths of the compile CPU: (n50, n80, files)."""
    times = sorted((step["s"] for step in compiles), reverse=True)
    total = sum(times)
    reached, counts = 0.0, {}
    for number, value in enumerate(times, 1):
        reached += value
        for share in (0.5, 0.8):
            counts.setdefault(share, number) if reached >= share * total else None
    return counts.get(0.5, 0), counts.get(0.8, 0), len(times)


def bound_by(efficiency):
    """What limits the wall time, from how many of the job slots were busy on average."""
    if efficiency >= 0.85:
        return "CPU throughput: the cores are busy all the time, so only less compile work makes the build faster"
    if efficiency >= 0.6:
        return "mostly CPU throughput, with some idle time (see the tail and the running steps per tenth)"
    return "idle cores: a serial phase or a few long steps limit the wall time (see the PCH, the tail and the long steps below)"


def section_glance(detail, steps, wall, cpu):
    jobs = detail["jobs"]
    compiles = [s for s in steps if s["k"] == "compile"]
    cold = detail["source"] == "cold"
    lines = [f"{'cold build' if cold else 'all steps'}: " + (f"wall {duration(wall)}, " if cold else "")
             + f"CPU {duration(cpu)}, {len(steps)} steps ({len(compiles)} compiles), {jobs} parallel jobs"]
    if not cold:
        lines.append("no timeline (wall time, parallelism, tail): Ninja stores times per build, so a snapshot needs the 'cold' scenario for them")
        return lines
    efficiency = cpu / (wall * jobs) if wall and jobs else 0
    lines.append(f"parallelism: {cpu / wall:.1f} running steps on average = {percent(cpu, wall * jobs)} of the {jobs} job slots"
                 if wall else "parallelism: -")
    if wall:
        lines.append(f"the wall time is bound by {bound_by(efficiency)}")
    for step in sorted((s for s in steps if s["k"] == "pch"), key=lambda s: s["e"]):
        name = next((part[:-4] for part in step["o"].split("/") if part.endswith(".dir")), step["o"])
        lines.append(f"precompiled header {name}: "
                     f"{duration(step['s'])}, ready after {duration(step['e'])} ({percent(step['e'], wall)} of the wall time)")
    tail = sorted((s for s in steps if s["e"] >= wall * (1 - TAIL_FRACTION)), key=lambda s: -s["s"])[:3]
    if tail:
        lines.append(f"tail: steps still running in the last {TAIL_FRACTION:.0%} of the build: "
                     + "; ".join(f"{os.path.basename(s['o'])} ({duration(s['s'])})" for s in tail))
    levels = concurrency(steps)
    if levels:
        lines.append("running steps per tenth of the build: " + " ".join(f"{value:.0f}" for value in levels))
    return lines


def section_groups(steps, cpu, top):
    groups = defaultdict(lambda: [0, 0.0])
    modules = defaultdict(lambda: [0, 0.0])
    for step in steps:
        if step["k"] == "compile":
            module = module_of(step.get("src"))
            name = group_of(module)
            groups[name][0] += 1
            groups[name][1] += step["s"]
            if name == "src (project)":
                modules[module][0] += 1
                modules[module][1] += step["s"]
        else:
            name = {"pch": "precompiled header", "library": "linking libraries", "executable": "linking executables"}.get(step["k"], "other steps")
            groups[name][0] += 1
            groups[name][1] += step["s"]
    rows = [["", "steps", "CPU", "share"]]
    for name, (count, seconds) in sorted(groups.items(), key=lambda item: -item[1][1]):
        rows.append([name, count, duration(seconds), percent(seconds, cpu)])
        if name == "src (project)":
            ordered = sorted(modules.items(), key=lambda item: -item[1][1])
            for module, (mcount, mseconds) in ordered[:top]:
                rows.append([f"  {module}", mcount, duration(mseconds), percent(mseconds, cpu)])
            rest = ordered[top:]
            if rest:
                rows.append([f"  ... {len(rest)} more modules", sum(c for _, (c, _) in rest), duration(sum(s for _, (_, s) in rest)),
                             percent(sum(s for _, (_, s) in rest), cpu)])
    return [table(rows, left=(0,))]


def section_slowest(detail, steps, wall, cpu, top):
    cold = detail["source"] == "cold"
    compiles = sorted((s for s in steps if s["k"] == "compile"), key=lambda s: (-s["s"], s["o"]))
    has_churn = any("ch" in s for s in compiles)
    header = ["time", "% CPU"] + (["ran"] if cold else []) + ["repo files"] + (["PRs"] if has_churn else []) + ["file"]
    rows = [header]
    for step in compiles[:top]:
        late = "*" if cold and wall and step["e"] >= wall * (1 - TAIL_FRACTION) else " "
        row = [duration(step["s"]), percent(step["s"], cpu)] + ([f"{step['a']:.0f}-{step['e']:.0f}s{late}"] if cold else []) + [step.get("p", "-")]
        if has_churn:
            row.append(step.get("ch", "-"))
        row.append(step.get("src") or step["o"])
        rows.append(row)
    n50, n80, files = concentration(compiles)
    lines = [table(rows)]
    lines.append(f"concentration: {n50} files ({percent(n50, files)}) make up half of the compile CPU, {n80} files ({percent(n80, files)}) four fifths")
    if cold:
        lines.append("* ends in the last tenth of the build: the build waits for it (a long pole)")
    heavy = sorted((s for s in compiles if "p" in s), key=lambda s: (-s["p"], s["o"]))[:min(top, 8)]
    if heavy:
        lines += ["", "Translation units that pull in the most repository files (parse load)",
                  table([["repo files", "time", "file"]] + [[s["p"], duration(s["s"]), s.get("src") or s["o"]] for s in heavy])]
    return lines


def section_headers(detail, steps, top):
    compiles = [s for s in steps if s["k"] == "compile"]
    compile_cpu = sum(s["s"] for s in compiles)
    objects = detail["objects"] or 1
    headers = detail["headers"]
    if not headers:
        return ["no include data (ninja -t deps was empty)"]
    has_churn = any("ch" in h for h in headers)

    def label(members):
        names = [m["f"] for m in members]
        text = names[0] + (" (external)" if members[0].get("x") else "")
        if len(names) > 1:
            text += f"  +{len(names) - 1} with the same includers: " + ", ".join(os.path.basename(n) for n in names[1:4]) + (", ..." if len(names) > 4 else "")
        return text

    def row(members):
        h = members[0]
        base = [h["fan_in"], percent(h["fan_in"], objects), duration(h["cpu_s"]), percent(h["cpu_s"], compile_cpu)]
        if has_churn:
            base += ["-", "-"] if "ch" not in h else [h["ch"], duration(h["ch"] * h["cpu_s"])]
        return base + [label(members)]

    head = ["fan-in", "of TUs", "rebuild CPU", "% compile CPU"] + (["PRs", "burden"] if has_churn else []) + ["header"]
    lines = []
    if has_churn:
        by_burden = sorted((h for h in headers if h.get("ch")), key=lambda h: (-h["ch"] * h["cpu_s"], -h["cpu_s"], h["f"]))
        burden_groups = {}
        for h in by_burden:   # the same includers and the same number of changes: one row
            burden_groups.setdefault((h.get("g") or (h["fan_in"], h["cpu_s"]), h["ch"]), []).append(h)
        lines += [f"By burden: rebuild CPU times the number of PRs that changed the header in the last {detail['churn_days']} days "
                  f"(history of {detail['churn_ref']}; for a header of a submodule the changes are bumps of the submodule)",
                  table([head] + [row(members) for members in list(burden_groups.values())[:top]]), ""]
    by_cost = sorted(headers, key=lambda h: (-h["cpu_s"], h["f"]))
    groups = {}
    for h in by_cost:   # headers included by exactly the same files are one row
        groups.setdefault(h.get("g") or (h["fan_in"], h["cpu_s"]), []).append(h)
    rows = [["fan-in", "of TUs", "rebuild CPU", "% compile CPU", "header"]]
    for members in list(groups.values())[:top]:
        h = members[0]
        rows.append([h["fan_in"], percent(h["fan_in"], objects), duration(h["cpu_s"]), percent(h["cpu_s"], compile_cpu), label(members)])
    lines += ["By rebuild cost: the compile CPU of every translation unit that includes the header",
              table(rows)]
    return lines


def section_touches(detail, top):
    lines = []
    names = {"touch_leaf": "touching a source file", "touch_header_top": "touching the most widely included header",
             "touch_header_median": "touching a median header"}
    for scenario in ("touch_leaf", "touch_header_top", "touch_header_median"):
        touch = detail.get("touch", {}).get(scenario)
        if not touch:
            continue
        steps = touch["steps"]
        by_kind = defaultdict(lambda: [0, 0.0])
        for step in steps:
            by_kind[step["k"]][0] += 1
            by_kind[step["k"]][1] += step["s"]
        total_cpu = sum(v[1] for v in by_kind.values())
        parts = [f"{kind} {count} ({duration(seconds)})" for kind, (count, seconds) in sorted(by_kind.items(), key=lambda i: -i[1][1])]
        lines.append(f"{names[scenario]}: {touch['file']}")
        lines.append(f"  {len(steps)} steps, wall {duration(touch['wall_s'])}, CPU {duration(total_cpu)}: " + ", ".join(parts))
        link_cpu = by_kind["library"][1] + by_kind["executable"][1]
        if total_cpu and link_cpu * 2 >= total_cpu:
            lines.append(f"  mostly relinking ({percent(link_cpu, total_cpu)} of the CPU): {by_kind['library'][0]} libraries and "
                         f"{by_kind['executable'][0]} executables are linked again because they depend on the rebuilt code")
        for step in steps[:min(top, 4)]:
            lines.append(f"    {duration(step['s']):>7s}  {step['k']:10s} {step['o']}")
    return lines or ["no touch scenarios in this snapshot"]


def section_links(steps, cpu, top):
    links = sorted((s for s in steps if s["k"] in LINK_KINDS), key=lambda s: (-s["s"], s["o"]))
    if not links:
        return ["no link steps"]
    total = sum(s["s"] for s in links)
    libraries = sum(1 for s in links if s["k"] == "library")
    lines = [f"{len(links)} links ({libraries} libraries, {len(links) - libraries} executables) take {duration(total)} CPU = {percent(total, cpu)} of the build"]
    lines.append(table([["time", "kind", "output"]] + [[duration(s["s"]), s["k"], s["o"]] for s in links[:min(top, 6)]], left=(1,)))
    return lines


def render(detail, snapshot, top=12):
    """The detail report of a snapshot as text."""
    steps = detail["steps"]
    if not steps:
        return "this snapshot has no build steps (was the build already up to date?)"
    wall = max(s["e"] for s in steps) if detail["source"] == "cold" else 0.0
    cpu = sum(s["s"] for s in steps)
    source = {"cold": "from the cold build of this snapshot", "log": "from the last build of every file (no cold build in this snapshot)"}[detail["source"]]
    git = snapshot.get("git", {})
    out = [
        f"snapshot {snapshot['id']}  {git.get('commit', '?')}{'*' if git.get('dirty') else ''}"
        + (f'  "{snapshot["note"]}"' if snapshot.get("note") else ""),
        f"fingerprint {snapshot['fingerprint_id']}: {bt_fingerprint.describe(snapshot['fingerprint'])}",
        f"per-file times {source}",
        f"(with {detail['jobs']} jobs they include contention for cores and caches: read them as relative cost, not as the time of the file alone)",
        "",
        "== Build at a glance ==", *section_glance(detail, steps, wall, cpu),
        "", "== Where the CPU goes ==", *section_groups(steps, cpu, top),
        "", "== Slowest compile steps ==", *section_slowest(detail, steps, wall, cpu, top),
        "", "== Headers: what a change rebuilds ==", *section_headers(detail, steps, top),
        "", "== What small changes rebuild ==", *section_touches(detail, top),
        "", "== Linking ==", *section_links(steps, cpu, top),
    ]
    return "\n".join(out)


# --- what changed between two snapshots ---------------------------------------------------------------------------

# Calibrated on three cold builds of identical code: after removing the drift of the whole build, no module moved by more
# than 5 s or 8% (the small ones by up to 38% but under 1 s, mid-size ones by up to 3.5 s, the big ones by up to 5%).
MODULE_NOISE_S = 5.0
MODULE_NOISE_RELATIVE = 0.08


def module_cpu(steps):
    """{module: (compile steps, compile CPU)} of a snapshot's steps."""
    totals = defaultdict(lambda: [0, 0.0])
    for step in steps:
        if step["k"] == "compile":
            module = module_of(step.get("src"))
            totals[module][0] += 1
            totals[module][1] += step["s"]
    return {module: (count, seconds) for module, (count, seconds) in totals.items()}


def drift(modules_before, modules_after):
    """How much slower (>1) or faster (<1) the typical module ran: the ratio at which half of the compile CPU lies.

    Weighted by CPU so that the big modules decide, and robust against one module that really changed.
    """
    pairs = sorted((modules_after[m][1] / modules_before[m][1], modules_before[m][1])
                   for m in modules_before if m in modules_after and modules_before[m][1] > 0)
    half, reached = sum(weight for _, weight in pairs) / 2, 0.0
    for ratio, weight in pairs:
        reached += weight
        if reached >= half:
            return ratio
    return 1.0


def render_changes(before, after, top=8):
    """What changed between the details of two snapshots: exact figures first, timings only where they exceed noise."""
    lines = []
    files_a = {s["src"]: s for s in before["steps"] if s["k"] == "compile" and s.get("src")}
    files_b = {s["src"]: s for s in after["steps"] if s["k"] == "compile" and s.get("src")}

    moved = sorted(((name, files_a[name]["p"], files_b[name]["p"]) for name in files_a.keys() & files_b.keys()
                    if "p" in files_a[name] and "p" in files_b[name] and files_a[name]["p"] != files_b[name]["p"]),
                   key=lambda item: (-abs(item[2] - item[1]), item[0]))
    new, gone = sorted(files_b.keys() - files_a.keys()), sorted(files_a.keys() - files_b.keys())
    lines.append("Translation units whose number of included repository files changed (exact)")
    lines.append(table([["before", "after", "change", "file"]] + [[a, b, f"{b - a:+d}", name] for name, a, b in moved[:top]])
                 if moved else "none")
    if len(moved) > top:
        lines.append(f"... and {len(moved) - top} more")
    if new or gone:
        lines.append(f"{len(new)} translation units added, {len(gone)} removed")

    exact = bool(before.get("fan_in") and after.get("fan_in"))
    if exact:
        fan_a, fan_b = before["fan_in"], after["fan_in"]
    else:   # a detail without the full map: compare what both have, the top lists
        fan_a, fan_b = ({h["f"]: h["fan_in"] for h in side["headers"]} for side in (before, after))
    cpu_a, cpu_b = {h["f"]: h["cpu_s"] for h in before["headers"]}, {h["f"]: h["cpu_s"] for h in after["headers"]}
    fan = sorted(((name, fan_a[name], fan_b[name]) for name in fan_a.keys() & fan_b.keys() if fan_a[name] != fan_b[name]),
                 key=lambda item: (-abs(item[2] - item[1]), item[0]))
    lines += ["", "Headers whose fan-in (number of files that include them) changed (exact; the rebuild CPU is the timing of the two cold builds)"]
    lines.append(table([["fan-in", "rebuild CPU", "header"]] + [
        [f"{a} -> {b}", f"{duration(cpu_a[name])} -> {duration(cpu_b[name])}" if name in cpu_a and name in cpu_b else "-", name]
        for name, a, b in fan[:top]]) if fan else "none")
    if len(fan) > top:
        lines.append(f"... and {len(fan) - top} more")
    entered, left = sorted(fan_b.keys() - fan_a.keys()), sorted(fan_a.keys() - fan_b.keys())
    if entered or left:
        lines.append(f"{len(entered)} headers added to the build, {len(left)} no longer included by anything" if exact else
                     f"{len(entered)} headers entered and {len(left)} left the top {TOP_HEADERS_STORED} by rebuild cost (one snapshot has no full header list)")

    mods_a, mods_b = module_cpu(before["steps"]), module_cpu(after["steps"])
    scale = drift(mods_a, mods_b)
    total_a, total_b = sum(c for _, c in mods_a.values()), sum(c for _, c in mods_b.values())
    rows = []
    for module in sorted(mods_a.keys() | mods_b.keys()):
        (count_a, cpu_a), (count_b, cpu_b) = mods_a.get(module, (0, 0.0)), mods_b.get(module, (0, 0.0))
        expected = cpu_a * scale
        beyond = cpu_b - expected
        if abs(beyond) >= max(MODULE_NOISE_S, MODULE_NOISE_RELATIVE * expected):
            rows.append((-abs(beyond), module, count_a, count_b, cpu_a, cpu_b, beyond, expected))
    lines += ["", f"Compile CPU per module: whole build {duration(total_a)} -> {duration(total_b)}, the typical module ran x{scale:.2f}",
              f"(a module is listed when it moved by more than {MODULE_NOISE_S:.0f} s and {MODULE_NOISE_RELATIVE:.0%} beyond that drift; measured on three "
              f"cold builds of identical code, where nothing did; each cold build is one run)"]
    lines.append(table([["files", "CPU", "change", "beyond drift", "module"]] + [
        [f"{ca} -> {cb}", f"{duration(a)} -> {duration(b)}", f"{(b - a) / a * 100:+.0f}%" if a else "new",
         f"{beyond:+.0f}s ({beyond / expected * 100:+.0f}%)" if expected else f"{beyond:+.0f}s", module]
        for _, module, ca, cb, a, b, beyond, expected in sorted(rows)[:top]]) if rows else "no module changed beyond the noise")
    return "\n".join(lines)
