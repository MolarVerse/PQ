"""The report parts built from clang `-ftime-trace` data (see bt_trace.py): compiler time by phase, module, header
and template, and what changed between two snapshots.

Imported lazily by bt_detail, which provides the shared formatting helpers.
"""

import bt_detail as base

HINT = ("no clang trace data in this snapshot: build with clang to get the split of compiler time into parsing, "
        "template instantiation and code generation per file, header and template "
        "(pqbt.py snapshot --compiler clang++-20, see the README)")


def frontend_note(totals):
    frontend = totals["frontend_s"]
    return [
        f"inside the frontend: time in headers (their own, without what they include) {base.duration(totals['header_self_s'])} "
        f"= {base.percent(totals['header_self_s'], frontend)}, template instantiation (own) {base.duration(totals['template_self_s'])} "
        f"= {base.percent(totals['template_self_s'], frontend)}",
        "  these overlap: an instantiation that happens while a header is parsed is counted in both",
        f"events: {totals['inclusions']:,} inclusions of {totals['distinct_headers']:,} distinct headers, "
        f"{totals['instantiations']:,} instantiations of {totals['distinct_templates']:,} distinct templates "
        f"(events under clang's 0.5 ms trace granularity are not recorded)",
    ]


def by_module(files, top):
    modules = {}
    for item in files:
        module = base.module_of(item.get("src"))
        entry = modules.setdefault(module, [0, 0.0, 0.0])
        entry[0] += 1
        entry[1] += item["frontend_s"]
        entry[2] += item["backend_s"]
    ordered = sorted(modules.items(), key=lambda kv: (-(kv[1][1] + kv[1][2]), kv[0]))
    rows = [["files", "frontend", "backend", "backend share", "module"]]
    for module, (count, frontend, backend) in ordered[:top]:
        rows.append([count, base.duration(frontend), base.duration(backend), base.percent(backend, frontend + backend), module])
    rest = ordered[top:]
    if rest:
        count = sum(v[0] for _, v in rest)
        frontend, backend = sum(v[1] for _, v in rest), sum(v[2] for _, v in rest)
        rows.append([count, base.duration(frontend), base.duration(backend), base.percent(backend, frontend + backend),
                     f"... {len(rest)} more modules"])
    return base.table(rows)


def headers_table(trace, top):
    frontend = trace["totals"]["frontend_s"]
    has_churn = any("ch" in h for h in trace["headers"])
    rows = [["own time", "% frontend", "with includes", "files", "per file"] + (["PRs"] if has_churn else []) + ["header"]]
    for header in trace["headers"][:top]:
        row = [base.duration(header["self_s"]), base.percent(header["self_s"], frontend), base.duration(header["incl_s"]),
               header["tus"], f"{header['self_s'] / header['tus'] * 1000:.0f} ms"]
        if has_churn:
            row.append(header.get("ch", "-"))
        rows.append(row + [header["f"]])
    return base.table(rows)


def templates_table(trace, top):
    frontend = trace["totals"]["frontend_s"]
    rows = [["own time", "% frontend", "with nested", "instantiations", "files", "template"]]
    for template in trace["templates"][:top]:
        rows.append([base.duration(template["self_s"]), base.percent(template["self_s"], frontend), base.duration(template["incl_s"]),
                     f"{template['n']:,}", template["tus"], template["f"]])
    return base.table(rows)


def backend_heavy(files, top):
    ranked = sorted(files, key=lambda f: (-f["backend_s"], f["src"] or ""))[:min(top, 6)]
    rows = [["backend", "frontend", "backend share", "file"]]
    for item in ranked:
        rows.append([base.duration(item["backend_s"]), base.duration(item["frontend_s"]),
                     base.percent(item["backend_s"], item["frontend_s"] + item["backend_s"]), item["src"] or "?"])
    return base.table(rows)


def render_section(detail, top=12):
    """The "Compiler time" section as a list of lines."""
    trace = detail.get("trace")
    if not trace:
        return [HINT]
    totals = trace["totals"]
    total = totals["total_s"]
    lines = [f"compiler CPU {base.duration(total)}: frontend (parsing, semantic analysis, template instantiation) "
             f"{base.duration(totals['frontend_s'])} = {base.percent(totals['frontend_s'], total)}, backend (optimisation, code generation) "
             f"{base.duration(totals['backend_s'])} = {base.percent(totals['backend_s'], total)}"]
    lines += frontend_note(totals)
    if trace["missing"] or trace["unreadable"]:
        lines.append(f"({trace['missing']} files without a trace, {trace['unreadable']} unreadable)")
    lines += ["", "By module", by_module(trace["files"], top),
              "", "Headers by their own parse time, summed over all files that include them "
                  "('with includes' counts what they include, 'per file' is the own time of one inclusion)",
              headers_table(trace, top),
              "", "Template instantiations by their own time", templates_table(trace, top),
              "", "Files where the backend (optimisation and code generation) takes the most", backend_heavy(trace["files"], top)]
    return lines


# --- what changed between two snapshots ---------------------------------------------------------------------------

# Calibrated on three clang cold builds of identical code (see the README): after removing the drift of the whole
# build (up to 6% in total CPU) no header, template, module frontend or module backend time moved by more than
# these amounts (0 of 866 headers, 581 templates and 105 + 105 module phases). Event counts are not exact between
# identical builds (about 5% of the headers differ), so only times are compared.
ITEM_NOISE_S = 1.0
ITEM_NOISE_RELATIVE = 0.15
PHASE_NOISE_S = 5.0
PHASE_NOISE_RELATIVE = 0.08


def drift_of(before, after):
    """How much slower (>1) or faster (<1) the typical entry ran: the ratio at which half of the time lies."""
    pairs = sorted((after[name] / before[name], before[name]) for name in before if name in after and before[name] > 0)
    half, reached = sum(weight for _, weight in pairs) / 2, 0.0
    for ratio, weight in pairs:
        reached += weight
        if reached >= half:
            return ratio
    return 1.0


def movers(before, after, absolute, relative):
    """([(beyond seconds, name, before, after, expected)], drift) of the entries that moved beyond the noise."""
    scale = drift_of(before, after)
    found = []
    for name in before.keys() & after.keys():
        expected = before[name] * scale
        beyond = after[name] - expected
        if abs(beyond) >= max(absolute, relative * expected):
            found.append((beyond, name, before[name], after[name], expected))
    return sorted(found, key=lambda item: (-abs(item[0]), item[1])), scale


def change_rows(found):
    return [[f"{base.duration(a)} -> {base.duration(b)}", f"{(b - a) / a * 100:+.0f}%" if a else "new",
             f"{beyond:+.1f}s ({beyond / expected * 100:+.0f}%)" if expected else f"{beyond:+.1f}s", name]
            for beyond, name, a, b, expected in found]


def phase_by_module(files, phase):
    modules = {}
    for item in files:
        module = base.module_of(item.get("src"))
        modules[module] = modules.get(module, 0.0) + item[phase]
    return modules


def render_changes(before, after, top=8):
    """The changes of the clang trace data between two snapshots as text lines (times only, beyond measured noise)."""
    ta, tb = before["totals"], after["totals"]

    def move(a, b):
        return f"{base.duration(a)} -> {base.duration(b)} ({(b - a) / a * 100:+.1f}%)" if a else f"{base.duration(b)}"

    lines = [f"compiler CPU {move(ta['total_s'], tb['total_s'])}: frontend {move(ta['frontend_s'], tb['frontend_s'])}, "
             f"backend {move(ta['backend_s'], tb['backend_s'])}"]
    notes = []
    for phase, label in (("frontend_s", "frontend"), ("backend_s", "backend")):
        found, scale = movers(phase_by_module(before["files"], phase), phase_by_module(after["files"], phase), PHASE_NOISE_S, PHASE_NOISE_RELATIVE)
        notes.append((label, found, scale))
    rows = [["phase", "CPU", "change", "beyond drift", "module"]]
    for label, found, _ in notes:
        rows += [[label] + row for row in change_rows(found)[:top]]
    lines += ["", "Modules whose frontend or backend time moved (beyond the drift of that phase; "
                  f"more than {PHASE_NOISE_S:.0f} s and {PHASE_NOISE_RELATIVE:.0%})",
              base.table(rows, left=(0,)) if len(rows) > 1 else "none beyond the noise"]
    lines.append("drift of the typical module: " + ", ".join(f"{label} x{scale:.2f}" for label, _, scale in notes))
    for key, title in (("headers", "Headers whose own parse time moved"), ("templates", "Templates whose own time moved")):
        a = {item["f"]: item["self_s"] for item in before[key]}
        b = {item["f"]: item["self_s"] for item in after[key]}
        found, scale = movers(a, b, ITEM_NOISE_S, ITEM_NOISE_RELATIVE)
        lines += ["", f"{title} (the typical one ran x{scale:.2f}; listed when it moved by more than {ITEM_NOISE_S:.0f} s and "
                      f"{ITEM_NOISE_RELATIVE:.0%} beyond that; only the top {len(a)} of the first snapshot are compared)"]
        rows = [["own time", "change", "beyond drift", "name"]] + change_rows(found)[:top]
        lines.append(base.table(rows) if len(rows) > 1 else "none beyond the noise")
        if len(found) > top:
            lines.append(f"... and {len(found) - top} more")
    return lines
