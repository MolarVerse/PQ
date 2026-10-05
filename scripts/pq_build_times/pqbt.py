#!/usr/bin/env python3
"""Local build-time tracking for PQ: snapshots, a local-only history, a baseline and an over-time graph.

  pqbt.py snapshot --note "what changed"   run the scenarios and store a snapshot
  pqbt.py report                           write the HTML graph and print the table
  pqbt.py compare [BEFORE] [AFTER]         compare two snapshots (default: baseline or previous vs latest)
  pqbt.py baseline set [ID]                pin a snapshot as the baseline of its fingerprint
  pqbt.py list                             show the stored fingerprints

Data lives outside the repository (see --data-dir) and is never shared with CI.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bt_compare  # noqa: E402
import bt_fingerprint  # noqa: E402
import bt_report  # noqa: E402
import bt_snapshot  # noqa: E402
import bt_store  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", help=f"where snapshots live (default: ${bt_store.ENV_VAR} or ~/.local/share/pq-build-times)")
    commands = parser.add_subparsers(dest="command", required=True)

    snap = commands.add_parser("snapshot", help="run the scenarios and store a snapshot")
    snap.add_argument("--build-dir", default=str(REPO_ROOT / "build-times"), help="directory this tool owns and may delete (default: build-times/ in the repository, which git ignores)")
    snap.add_argument("--source-root", default=str(REPO_ROOT))
    snap.add_argument("--build-type", default="Release")
    snap.add_argument("--target", default="all", help="ninja target to build (default: all)")
    snap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    snap.add_argument("--repeat", type=int, default=3, help="repetitions of the no-op and touch scenarios (median is stored)")
    snap.add_argument("--cold-runs", type=int, default=1, help="repetitions of the cold build")
    snap.add_argument("--scenarios", default=",".join(bt_snapshot.SCENARIOS), help="comma separated subset")
    snap.add_argument("--cmake-arg", action="append", default=[], help="extra CMake argument (repeatable), recorded in the fingerprint")
    snap.add_argument("--ccache", action="store_true", help="allow ccache; off by default because cache hits hide compile time")
    snap.add_argument("--note", default="", help="what changed since the last snapshot; shown as a marker in the graph")
    snap.add_argument("--leaf", help="source file to touch (default: a median source file, then pinned)")
    snap.add_argument("--header-top", help="header to touch (default: the most widely included one, then pinned)")
    snap.add_argument("--header-median", help="header to touch (default: a median one, then pinned)")
    snap.add_argument("--force", action="store_true", help="run although the machine looks busy")

    report = commands.add_parser("report", help="write the HTML graph and print the table")
    report.add_argument("--fingerprint", help="fingerprint id (default: the one of the newest snapshot)")
    report.add_argument("--out", help="HTML file (default: <data dir>/report-<fingerprint>.html)")
    report.add_argument("--no-html", action="store_true", help="only print the table")

    compare = commands.add_parser("compare", help="compare two snapshots of one fingerprint")
    compare.add_argument("before", nargs="?", help="snapshot id (or prefix), 'baseline', 'previous' or 'latest'")
    compare.add_argument("after", nargs="?", help="snapshot id (or prefix) or 'latest' (default)")
    compare.add_argument("--fingerprint", help="fingerprint id (default: the one of the newest snapshot)")

    baseline = commands.add_parser("baseline", help="pin or show the baseline")
    baseline.add_argument("action", choices=["set", "show"])
    baseline.add_argument("snapshot", nargs="?", default="latest", help="snapshot id or 'latest' (default)")
    baseline.add_argument("--fingerprint", help="fingerprint id (default: the one of the newest snapshot)")

    commands.add_parser("list", help="show the stored fingerprints")
    return parser.parse_args(argv)


def resolve_fingerprint(root, requested):
    grouped = bt_store.fingerprints(root)
    if not grouped:
        raise SystemExit(f"no snapshots in {root}; run: pqbt.py snapshot")
    if requested:
        if requested not in grouped:
            raise SystemExit(f"unknown fingerprint {requested}; known: {', '.join(sorted(grouped))}")
        return requested
    newest = max(grouped.items(), key=lambda item: item[1][-1]["id"])
    return newest[0]


def command_snapshot(args, root):
    scenarios = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    load = os.getloadavg()[0]
    cores = os.cpu_count() or 1
    if load > cores * 0.25 and not args.force:
        raise SystemExit(f"the machine looks busy (load {load:.1f} on {cores} threads); timings would be noise. Retry when idle or pass --force.")
    builder = bt_snapshot.Builder(
        args.source_root, args.build_dir, root / "deps", args.build_type, args.cmake_arg, args.target, args.jobs,
    )
    if args.ccache:
        builder.cmake_args = [a for a in builder.cmake_args if "COMPILER_LAUNCHER" not in a]
    try:
        snapshot = bt_snapshot.take_snapshot(
            builder, root, scenarios, args.repeat, args.cold_runs, args.note,
            {"leaf": args.leaf, "header_top": args.header_top, "header_median": args.header_median},
            args.ccache,
        )
    except bt_snapshot.SnapshotError as error:
        raise SystemExit(f"error: {error}")
    path = bt_store.save_snapshot(root, snapshot)
    print(f"saved {path}")
    print(f"fingerprint {snapshot['fingerprint_id']}: {bt_fingerprint.describe(snapshot['fingerprint'])}")
    if bt_store.baseline_snapshot(root, snapshot["fingerprint_id"]) is None:
        print("no baseline for this fingerprint yet; pin this one with: pqbt.py baseline set")
    return 0


def command_report(args, root):
    fingerprint = resolve_fingerprint(root, args.fingerprint)
    snapshots = bt_store.load_snapshots(root, fingerprint)
    baseline = bt_store.baseline_snapshot(root, fingerprint, snapshots)
    print(f"fingerprint {fingerprint}: {bt_fingerprint.describe(snapshots[-1]['fingerprint'])}\n")
    print(bt_report.render_table(snapshots, baseline))
    if not args.no_html:
        out = Path(args.out) if args.out else root / f"report-{fingerprint}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(bt_report.render_html(snapshots, baseline, fingerprint), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


def command_compare(args, root):
    all_snapshots = bt_store.load_snapshots(root)
    by_id = {snapshot["id"]: snapshot for snapshot in all_snapshots}
    fingerprint = resolve_fingerprint(root, args.fingerprint)
    snapshots = bt_store.load_snapshots(root, fingerprint)
    baseline = bt_store.baseline_snapshot(root, fingerprint, snapshots)

    def pick(reference):
        # an id of another series is looked up as well, so that comparing across fingerprints is refused loudly
        if reference in by_id:
            return by_id[reference]
        return bt_compare.resolve(snapshots, baseline, reference)

    try:
        after = pick(args.after or "latest")
        if args.before:
            before = pick(args.before)
        else:
            before = baseline if baseline and baseline["id"] != after["id"] else bt_compare.resolve(snapshots, None, "previous")
        if before["id"] == after["id"]:
            raise bt_compare.CompareError("both references are the same snapshot")
        print(bt_compare.render(bt_compare.compare(before, after)))
    except bt_compare.CompareError as error:
        raise SystemExit(f"error: {error}")
    return 0


def command_baseline(args, root):
    fingerprint = resolve_fingerprint(root, args.fingerprint)
    snapshots = bt_store.load_snapshots(root, fingerprint)
    if args.action == "show":
        baseline = bt_store.baseline_snapshot(root, fingerprint, snapshots)
        print(baseline["id"] if baseline else "no baseline")
        return 0
    chosen = snapshots[-1] if args.snapshot == "latest" else next((s for s in snapshots if s["id"] == args.snapshot), None)
    if chosen is None:
        raise SystemExit(f"no snapshot {args.snapshot} for fingerprint {fingerprint}")
    bt_store.set_baseline(root, fingerprint, chosen["id"])
    print(f"baseline of {fingerprint} is now {chosen['id']}")
    return 0


def command_list(args, root):
    grouped = bt_store.fingerprints(root)
    if not grouped:
        print(f"no snapshots in {root}")
        return 0
    baselines = bt_store.load_baselines(root)
    for fingerprint, snapshots in sorted(grouped.items()):
        print(f"{fingerprint}  {len(snapshots):3d} snapshots, latest {snapshots[-1]['id']}, baseline {baselines.get(fingerprint, '-')}")
        print(f"    {bt_fingerprint.describe(snapshots[-1]['fingerprint'])}")
    return 0


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = bt_store.data_dir(args.data_dir)
    handlers = {
        "snapshot": command_snapshot, "report": command_report, "compare": command_compare,
        "baseline": command_baseline, "list": command_list,
    }
    return handlers[args.command](args, root)


if __name__ == "__main__":
    sys.exit(main())
