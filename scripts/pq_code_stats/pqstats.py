#!/usr/bin/env python3
"""Code statistics of PQ's dev branch over time: lines and files per merged PR, split by area, group and kind.

  pqstats.py collect [--ci]     read the history, write the CSV files and the HTML report, verify the totals
  pqstats.py report             rebuild the HTML report from the CSV files
  pqstats.py show [--last N]    the latest changes in the terminal

The output directory (default $PQ_CODE_STATS_DIR, else $XDG_DATA_HOME/pq-code-stats, else
~/.local/share/pq-code-stats) is never part of the repository; everything in it can be recreated from git.
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cs_ci  # noqa: E402
import cs_classify  # noqa: E402
import cs_history  # noqa: E402
import cs_output  # noqa: E402
import cs_report  # noqa: E402

SCHEMA_VERSION = 1
ENV_VAR = "PQ_CODE_STATS_DIR"
REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir(override=None, environ=None):
    environ = os.environ if environ is None else environ
    if override:
        return Path(override).expanduser()
    if environ.get(ENV_VAR):
        return Path(environ[ENV_VAR]).expanduser()
    base = environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "pq-code-stats"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", help=f"output directory (default: ${ENV_VAR} or ~/.local/share/pq-code-stats)")
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="read the history and write the CSV files and the report")
    collect.add_argument("--repo-root", default=str(REPO_ROOT))
    collect.add_argument("--ref", help="branch or commit to read (default: origin/dev, else dev, else HEAD)")
    collect.add_argument("--since", help="only write changes from this date on (YYYY-MM-DD); the totals still count everything before")
    collect.add_argument("--exclude-pr", type=int, action="append", default=[], help="leave a PR out of the written files, for example a huge data import (repeatable)")
    collect.add_argument("--no-verify", action="store_true", help="skip the check of the rolled totals against a full count of the tree")
    collect.add_argument("--no-report", action="store_true", help="only write the CSV files")
    collect.add_argument("--ci", action="store_true", help="also fetch GitHub Actions runs (needs gh); only as far back as --ci-days")
    collect.add_argument("--ci-days", type=int, default=60)
    collect.add_argument("--github-repo", default="MolarVerse/PQ", help="owner/name for --ci")

    commands.add_parser("report", help="rebuild the HTML report from the CSV files")

    show = commands.add_parser("show", help="print the latest changes")
    show.add_argument("--last", type=int, default=15)
    return parser.parse_args(argv)


def iso_utc(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def filter_records(records, since, excluded):
    cutoff = date.fromisoformat(since) if since else None
    return [r for r in records
            if (cutoff is None or r.change.date.date() >= cutoff) and r.change.pr not in excluded]


def build_report(out):
    out = Path(out)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8")) if (out / "meta.json").exists() else {}
    ci_daily = cs_output.read_csv(out / "ci_daily.csv") if (out / "ci_daily.csv").exists() else None
    page = cs_report.render(cs_output.read_csv(out / "metrics.csv"), cs_output.read_csv(out / "weekly.csv"),
                            cs_output.read_csv(out / "changes.csv"), ci_daily, meta)
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path


def command_collect(args, out):
    ref = cs_history.resolve_ref(args.repo_root, args.ref)
    head = cs_history.git(args.repo_root, "rev-parse", ref + "^{commit}").decode().strip()
    records = cs_history.read_history(args.repo_root, ref)
    print(f"{len(records)} changes on {ref} ({sum(1 for r in records if r.change.kind == 'pr')} pull requests)")

    mismatches = None
    if not args.no_verify:
        mismatches = cs_history.verify(records, cs_history.count_tree(args.repo_root, ref))
        if mismatches:
            for key, rolled, counted in mismatches[:10]:
                print(f"MISMATCH {key}: rolled forward {rolled}, counted {counted}", file=sys.stderr)
            raise SystemExit("error: the rolled-forward totals differ from a full count of the tree; not writing anything")
        print("verified: the rolled-forward totals equal a full count of the tree")

    ci_daily = ci_by_pr = ci_window = None
    if args.ci:
        first, last = cs_ci.window(args.ci_days)
        try:
            branches = {}
            for record in records:
                if record.change.branch and record.change.pr:
                    branches.setdefault(record.change.branch, []).append((iso_utc(record.change.date), record.change.pr))
            ci_daily, ci_by_pr = cs_ci.aggregate(cs_ci.fetch_runs(args.github_repo, first, last), branches)
            ci_window = (first, last)
        except cs_ci.CiError as error:
            print(f"warning: no CI runs: {error}", file=sys.stderr)

    shown = filter_records(records, args.since, set(args.exclude_pr))
    paths = cs_output.write_all(out, shown, ci_daily, ci_by_pr, ci_window)
    meta = {
        "schema_version": SCHEMA_VERSION, "ref": ref, "head": head, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "changes_total": len(records), "changes_written": len(shown), "since": args.since, "excluded_prs": sorted(args.exclude_pr),
        "verified": None if mismatches is None else True, "ci_window": [day.isoformat() for day in ci_window] if ci_window else None,
    }
    (Path(out) / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for name, path in paths.items():
        print(f"wrote {path}")
    if not args.no_report and shown:
        print(f"wrote {build_report(out)}")
    if shown:
        last = cs_output.metrics_row(shown[-1])
        print(f"now: production C++ {last['prod_header_like_lines'] + last['prod_source_lines']:,} lines "
              f"({last['prod_source_share']:.0%} source), tests {last['tests_to_prod']:.2f} and perf {last['perf_to_prod']:.3f} per production line, "
              f"CI {last['ci_files']} files / {last['ci_lines']:,} lines")
    return 0


def command_report(args, out):
    if not (Path(out) / "metrics.csv").exists():
        raise SystemExit(f"no data in {out}; run: pqstats.py collect")
    print(f"wrote {build_report(out)}")
    return 0


def command_show(args, out):
    if not (Path(out) / "changes.csv").exists():
        raise SystemExit(f"no data in {out}; run: pqstats.py collect")
    rows = cs_output.read_csv(Path(out) / "changes.csv")[-args.last:]
    header = ["date", "change", "title", "+lines", "-lines", "prod hdr", "prod src", "tests", "perf", "ci", "ci runs"]
    table = [header]
    for row in rows:
        def net(*names):
            return sum(int(row[f"{n}_added"]) - int(row[f"{n}_deleted"]) for n in names)
        table.append([
            row["date"][:10], f"#{row['pr']}" if row["pr"] else row["kind"], row["title"][:44], row["lines_added"], row["lines_deleted"],
            f"{net('prod_header_like'):+d}", f"{net('prod_source'):+d}", f"{net('tests_header_like', 'tests_source'):+d}",
            f"{net('perf_header_like', 'perf_source'):+d}", f"{net('ci'):+d}", row["ci_runs"] or "-",
        ])
    widths = [max(len(str(r[i])) for r in table) for i in range(len(header))]
    for index, row in enumerate(table):
        print("  ".join(str(c).ljust(widths[i]) if i in (1, 2) else str(c).rjust(widths[i]) for i, c in enumerate(row)).rstrip())
        if index == 0:
            print("  ".join("-" * w for w in widths))
    return 0


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out = data_dir(args.out)
    return {"collect": command_collect, "report": command_report, "show": command_show}[args.command](args, out)


if __name__ == "__main__":
    sys.exit(main())
