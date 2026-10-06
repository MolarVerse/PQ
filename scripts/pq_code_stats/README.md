# Code statistics of dev over time

Reads the **first-parent history of `dev`** with one `git log` call and writes tidy CSV files (and a self-contained
HTML report) with the lines and files **per merged PR**, split by area, group and kind, plus the **state over time**
(files and lines after every change). Standard library only, no network (except the optional CI runs). Everything
can be recreated from git, so the output lives outside the repository.

```text
python3 scripts/pq_code_stats/pqstats.py collect                 # history -> CSV files + report.html (verifies the totals)
python3 scripts/pq_code_stats/pqstats.py collect --ci            # also GitHub Actions runs (needs gh), 60 days by default
python3 scripts/pq_code_stats/pqstats.py show --last 20          # the latest changes in the terminal
python3 scripts/pq_code_stats/pqstats.py report                  # rebuild report.html from the CSV files
```

Output directory: `--out`, else `$PQ_CODE_STATS_DIR`, else `~/.local/share/pq-code-stats`.
Useful options: `--since 2026-01-01` (only write newer changes; totals still count everything before),
`--exclude-pr N` (leave huge data imports out of the flows, repeatable), `--ref` (default `origin/dev`).

## The splits

Edit `AREA_RULES` in `cs_classify.py` to change them; everything else follows.

| Dimension | Values |
| --- | --- |
| **area** | `src`, `include`, `apps`, `tests`, `integration_tests`, `perf` (benchmarks/perf), `benchmarks`, `ci_workflows`, `ci_other`, `ci_data` (generated CI metrics), `scripts`, `docs`, `changelog`, `cmake`, `other` |
| **group** | `production` (src + include + apps), `tests` (tests + integration_tests), `perf` (perf + benchmarks), `ci`, `docs`, `build`, `data`, `other` |
| **kind** | `header`, `tpp`, `source`, `cmake`, `python`, `shell`, `config`, `doc`, `other` |
| **class** | `header_like` (header + tpp: compiled into every includer), `source`, `other` |

The headline columns use **C++ files only** (header, tpp, source) in three scopes: `prod` (src + include + apps),
`tests` (the unit tests; the 960k lines of integration reference data are *not* in it) and `perf` (perf + benchmarks).
That is what the two questions need: *tests + perf per production line* (a coverage proxy) and *source vs header-like
lines in production* (a compile time proxy: more in `.cpp`, less in headers).

Submodules (`external/`) are not counted. Lines are physical lines as `git` counts them (blank lines and comments
included). Renames count as delete + add. Binary files count as files with 0 lines.

## Files

| File | One row per | Columns |
| --- | --- | --- |
| `changes.csv` | change (merged PR, other merge, direct commit) | `sha, date, week, kind (pr/merge/commit), pr, title, author`, totals `files_added/deleted/modified, lines_added/deleted/net`, per group `<group>_added/_deleted`, per C++ scope `<prod\|tests\|perf>_<header_like\|source>_added/_deleted`, and `ci_runs, ci_attempts, ci_failed_runs` (with `--ci`) |
| `splits.csv` | change x area x kind (long form) | `sha, date, pr, change_kind, area, group, file_kind, code_class, files_added, files_deleted, files_modified, lines_added, lines_deleted` |
| `state.csv` | change x area x kind (long form) | `sha, date, pr, area, group, file_kind, code_class, files, lines` (the totals **after** the change) |
| `metrics.csv` | change | the state as wide columns: `<scope>_<class>_files/_lines`, `prod_source_share`, `tests_to_prod`, `perf_to_prod`, `tests_perf_to_prod`, `ci_workflow_*`, `ci_other_*`, `ci_files`, `ci_lines`, `integration_*`, `<group>_files/_lines` |
| `weekly.csv` | ISO week (weeks without changes are zero rows) | `week, week_start, changes, prs`, `<group>_added/_deleted/_net`, `<scope>_<class>_added/_deleted/_net`, `ci_runs, ci_failed_runs, ci_attempts` |
| `ci_daily.csv` | day x workflow x event x conclusion (`--ci`) | `date, workflow, event, conclusion, runs, attempts` |
| `meta.json` | - | ref, head, when, counts, filters, verification result, CI window |

Dates are UTC (`2026-10-05T20:09:54Z`). A merged PR's numbers are its **net change against the first parent**, i.e.
what it changed on `dev`, not the sum of its commits.

## Plotting

`report.html` has the standard charts (production C++ lines, source share, tests and perf per production line, files,
CI files and lines, weekly flows, PRs per week, CI runs). For your own plots load the CSVs:

```python
import pandas as pd
m = pd.read_csv("metrics.csv", parse_dates=["date"]).set_index("date")
m[["prod_source_share"]].plot(drawstyle="steps-post")          # lower = more in headers
m[["tests_to_prod", "perf_to_prod"]].plot(drawstyle="steps-post")

w = pd.read_csv("weekly.csv", parse_dates=["week_start"]).set_index("week_start")
w[["prod_header_like_net", "prod_source_net"]].rolling(4).sum().plot()   # 4-week net growth

s = pd.read_csv("state.csv", parse_dates=["date"])                        # any custom split
s[s.group == "production"].groupby(["date", "file_kind"]).lines.sum().unstack().ffill().plot.area()
```

```text
gnuplot: set datafile separator ","; set xdata time; set timefmt "%Y-%m-%dT%H:%M:%SZ"
         plot "metrics.csv" using 2:(column("prod_source_share")) with steps
```

## Verification and caveats

- `collect` rolls the totals forward change by change and then **compares them with a full count of the tree** at the
  ref (every file counted from its contents). It refuses to write anything if they differ; `--no-verify` skips it.
- History reaches back to the first commit; early imports of data files and the reverted `qmmm` merge are huge. Use the
  C++ columns, or `--exclude-pr` for the flows (the totals stay correct).
- CI runs (`--ci`) are queried one day at a time (the API returns at most 1,000 runs per query) and reach back as far as
  GitHub still has them. GitHub drops the pull request link of a run once its branch is deleted, so a `pull_request`
  run is assigned to the PR through its head branch name (taken from "Merge pull request #N from owner/branch"). PRs
  whose runs are older than the window show `0`. Weeks outside the window are empty, not zero.
