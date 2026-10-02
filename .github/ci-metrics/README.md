# CI metrics

Durable record of how long our CI jobs take, so CI speedup work can be measured
instead of guessed. Tracked in #720.

| Part | Status |
| --- | --- |
| Storage layout and record schema (this directory) | defined, see [`SCHEMA.md`](SCHEMA.md) |
| Collector script (GitHub API to JSONL), see [Running the collector](#running-the-collector) | implemented, #715 |
| Collector workflow, see [The workflow](#the-workflow) | implemented, #716 (inert until it reaches `main`) |
| Overview report for `dev`, see [`data/CI_TIMINGS.md`](data/CI_TIMINGS.md) and [Overview report](#overview-report) | implemented, #717 (build analysis section: #718) |
| Build timings: jobs upload a `.ninja_log` / ccache summary artifact, see [Build timings](#build-timings-718) | implemented (#718): job side, collector ingest and the overview section |
| clang `-ftime-trace` analysis | in progress, #719: clang job (#747), trace summariser (#748) and PR comment (#750) done; ingest and noise study planned |

## Layout

```text
.github/ci-metrics/
  README.md      this file
  SCHEMA.md      field definitions and derivation rules
  schema.json    JSON Schema for one record
  schema-build-analysis.json  JSON Schema for a build-analysis record
  config.json    repository, recorded workflows, workflows with build analysis, infrastructure-failure signatures
  collect.py     the collector (Python standard library only, uses the gh CLI)
  publish.sh     commits and pushes new data (used by the workflow)
  report.py      generates data/CI_TIMINGS.md from the shards (standard library only)
  summarise_build.py   runs in a CI job: summarises .ninja_log and ccache statistics
  summarise_traces.py  runs in the clang job: summarises the -ftime-trace files
  clang_report.py      renders the clang build time comment of a pull request
  data/          weekly JSONL shards and the generated CI_TIMINGS.md, written only by the tools above
```

The workflow is `.github/workflows/ci_metrics.yml`. Tests:
`scripts/tests/test_ci_metrics_{collect,publish,report,build_summary,traces,clang_report}.py`
(offline, the publish tests use real local git repositories; they run with the
other script tests in CI).

Nothing under `data/` is edited by hand.

## Decisions (recorded for #714)

1. **The data is committed to `dev`, under `.github/ci-metrics/data/`.**
2. **The collector writes with `GITHUB_TOKEN`** (`contents: write`), not a
   GitHub App token.
3. **All CI workflows are in scope from the start:** BUILD, LINT, Performance
   Gate, DevOps C++ Check, Clang Format Check, License Header Check, Changelog
   Fragment Check, Check PR Has Linked Issue, Dismiss Stale Approvals and Docs.
   Release, bot and Pages workflows are not recorded.

## Consequences of committing data to `dev`

- **This is an explicit exception to the "never push directly to `dev`" rule**
  in `AGENTS.md`. It applies only to the collector and only to
  `.github/ci-metrics/data/**`. People and coding assistants still never push
  to `dev`.
- **No CI feedback loop.** GitHub does not start new workflow runs for pushes
  made with `GITHUB_TOKEN` (documented behaviour; `workflow_dispatch` and
  `repository_dispatch` are the exceptions). In addition, the `changes`
  filters of BUILD and LINT (the only workflows that run on pushes to `dev`)
  do not match `.github/ci-metrics/**`, so even a push made with another token
  would start them but skip every real job.
- **Commit rate.** To keep `dev` history and every contributor's "merge
  `dev` again" churn small, the collector should batch: a daily schedule plus
  manual dispatch, so at most about one data commit per day. Per-run
  collection would add a commit to `dev` for every CI run. The trigger is
  settled in #716.
- **It relies on `dev` accepting that push.** Today `dev` has no branch
  protection and its only ruleset is disabled. If a rule that blocks direct
  pushes is enabled later, the collector will fail until `github-actions` is
  allowed to bypass it for this path.
- **The data travels.** It is merged into `NEXT` by the daily sync and into
  `main` by release PRs, so release diffs will include the data files.
- **Size.** About 4-16 MB per month uncompressed, roughly 1 MB per month in git
  (measurements in [`SCHEMA.md`](SCHEMA.md)). If repository size becomes a
  concern, move older shards to an archive location or a separate branch.
- **The changelog is unaffected.** `DEV-CHANGELOG.md` is built from
  `changes/` fragments, not from commit messages.

## Running the collector

```bash
python3 .github/ci-metrics/collect.py                     # last 3 days (incremental)
python3 .github/ci-metrics/collect.py --since-days 90     # backfill
python3 .github/ci-metrics/collect.py --run-id 36693119228
python3 .github/ci-metrics/collect.py --dry-run           # write nothing
python3 .github/ci-metrics/collect.py --data-dir /tmp/x   # try it without touching data/
python3 .github/ci-metrics/collect.py --since-days 90 --rate-limit-wait 90   # backfill, waiting out rate limits
python3 .github/ci-metrics/collect.py --fix-pr-numbers --dry-run   # preview the one-off pr_number repair
```

It needs the `gh` CLI, logged in (or `GH_TOKEN`/`GITHUB_TOKEN` set). Exit status
is 0 on success, 1 if some runs could not be collected, 2 if it stopped early
on the API rate limit; whatever was collected before stopping is still written.
When it stops on a rate limit, the summary shows GitHub's own message once.

- **Idempotent and resumable.** Jobs already present (by `job_id`) are never
  written again, and runs whose latest attempt is already recorded are not
  fetched again, so overlapping windows and re-runs are cheap and safe. Runs
  are processed newest first.
- **Cost.** One API call per run for its jobs, plus one per pull-request head
  commit and one per *failed* job (for the log). Measured on one busy day
  (95 runs): 165 calls (1 listing, 95 jobs, 48 pull-request lookups, 21 logs),
  taking 18-43 seconds with the default 8 workers. When the commit lookup does
  not find the PR, one more call lists the PRs of that branch (once per
  branch).
- **`--fix-pr-numbers`** is a one-off repair, and the only thing that ever
  changes existing data: it fills in `pr_number` where it is `null` on
  pull-request records (the first backfill wrote 450 of those, see
  [`SCHEMA.md`](SCHEMA.md)), touching nothing else on those lines. It makes
  about one API call per distinct commit or branch involved.
- **Rate limits.** A workflow's `GITHUB_TOKEN` is limited to about 1,000
  requests per hour per repository; a personal login gets 5,000. The daily
  incremental run needs a few hundred calls. The one real 90-day backfill
  (4,488 runs, 2026-07-03 to 2026-09-30, run locally with a personal login)
  stopped on the rate limit after about 45 minutes and about 4,200 runs, and
  calls worked again roughly 20 minutes later; the last ~270 runs then
  finished in about 6 minutes.
- **Waiting out a rate limit.** With `--rate-limit-wait MINUTES` a rate-limited
  call is retried after 1, 2, 4, 8, 10, 10 ... minutes for up to that long,
  instead of the run stopping (the default, `0`, stops). It deliberately does
  not try to compute when the limit resets: during that backfill
  `gh api rate_limit` kept reporting 5,000 remaining and a response header
  reported 4,173 remaining while calls were refused with HTTP 403, and a retry
  at the reset time given by that header was refused again. This was seen once,
  so the cause is not established; treat those readings as unreliable for
  scheduling. Runs are still written as they complete, so an interrupted
  backfill loses nothing and can simply be re-run.
- **Late changes.** A run is only recorded once it has completed. The default
  3-day lookback picks up runs that finished late and reruns of recent runs
  (a rerun of a run older than the lookback is not picked up).

## The workflow

`.github/workflows/ci_metrics.yml` ("CI Metrics") runs `collect.py` and commits
the new records to `dev` with `publish.sh`.

- **When.** Daily at 04:30 UTC (after the nightly BUILD at 02:00 and the
  dev to NEXT sync at 03:00), looking back 3 days, plus `workflow_dispatch`
  with `since_days` (1-100) and `dry_run` inputs. At most about one data commit
  per day, and none when there is nothing new.
- **It stays inert until it reaches `main`.** `schedule` only fires from the
  workflow file on the default branch (`main`). `workflow_dispatch` does not
  work for a workflow that exists only on another branch either: GitHub answers
  "workflow ci_metrics.yml not found on the default branch" (checked). So the
  first real run can only happen after a release brings the file to `main`.
- **Which ref it works on.** A scheduled run always uses `dev`. A manual run
  uses the ref it was started on and **only pushes when that ref is `dev`**;
  anywhere else it is a dry run that reports what it would commit.
- **Token and permissions.** The job's `GITHUB_TOKEN` with `contents: write`
  (to push to `dev`) and `actions: read`; nothing else. A push made with it does
  not start other workflows.
- **One at a time.** A `ci-metrics` concurrency group without cancellation, so
  two runs never write at once and a run that is writing is never killed.
- **Failure handling.** `publish.sh` only commits files under `data/`. It
  rebases onto the current `dev` first and refuses to push if that would change
  anything else; a rejected push (because `dev` moved) is fetched, rebased and
  retried up to 5 times; a conflicting rebase is aborted and nothing is pushed.
  If the collector hits a partial failure or a rate limit (`--rate-limit-wait
  30`), what it collected is still committed and the run then ends red. It is not
  a pull-request workflow, so it can never block a merge.
- **Output.** The collector's summary appears on the run's summary page.
- **Verified before merging.** The workflow was run for real from a throwaway
  branch (with a temporary `push` trigger that is not part of this repository)
  as a dry run: the token had `Actions: read`, `Contents: write`, `Metadata:
  read`; 455 runs were listed, 437 skipped as already collected, 18 collected
  and nothing pushed. The push itself is covered by `publish.sh`'s tests
  against local git repositories, not by a run against `dev`.

## Overview report

[`data/CI_TIMINGS.md`](data/CI_TIMINGS.md) is rendered by GitHub and regenerated
by the workflow after every collector run (`python3 .github/ci-metrics/report.py`;
options `--window-days`, `--regression-percent`, `--min-samples`, `--out`). It
lives in `data/` so the publish step commits it together with the new records.
It is generated only from the records and uses the newest record as "now", so
unchanged data gives identical bytes and never a commit.

What it shows, per recorded workflow:

- **Jobs:** median and p90 of the job duration over the current window (14 days
  by default), split into `push (dev)` and `pull_request`, next to the median of
  the 14 days before, the change, and the queue time apart from the duration.
- **Regressions:** a median up by more than 20% *and* at least 30 seconds, with
  at least 5 samples in both windows, is flagged and listed at the top.
- **Wall-clock per workflow:** first job created to last job finished, per run,
  and the job that usually finishes last (what a pull request waits for). Runs
  with any non-successful job, second attempts, and runs where the paths filter
  skipped every real job (only `changes` and `*-gate` ran) are left out, because
  they would pull the medians down.
- **Trend:** weekly median wall-clock of BUILD (Mermaid `xychart-beta`, no binary
  files), and the weekly Eigen cache hit rate (the effect of #703).
- **Build analysis** (from the `build-analysis` records; says so if there are none yet):
  per job the median ccache hit rate of cacheable calls, the calls blocked by the
  precompiled header (#732) and whether the cache is full; for complete ninja builds
  the wall and CPU time, parallelism, the link tail after the last compile and the
  link share of CPU time (does a bigger runner help?); and the slowest build steps
  of pushes to `dev`. Builds with `complete` false (`lint`) are left out of the ninja
  table and counted; each build keeps only its 20 slowest steps.
- **Exclusions:** what was left out and how many, so the numbers can be
  reproduced: cancelled, other events, pushes to branches other than `dev`,
  reruns, infrastructure failures and failed jobs.

Windows are measured in days, not in runs, so the current and the previous window
are comparable. The Eigen table starts in the week the `Cache Eigen source` step
was introduced; before that the flag is `null`.

## Build timings (#718)

Job and step durations say how long a build took, not why. The jobs that build
the project (`build` matrix, `build-static-lto`, `mpi-build`, `lint`) therefore
run the composite action `.github/actions/upload-build-timings` as a last step
(`if: always()`, never fails the job). It runs `summarise_build.py`, which writes:

- `build-analysis.json`: total build time, CPU time and parallelism, the split
  between compile, archive and link steps, the time between the last compile
  finishing and the build ending (`tail_after_compile_s`, what a link-bound
  build such as `build-static-lto` spends alone on one core), the 20 slowest
  steps, and the ccache counters with the hit rate of cacheable calls. The
  ccache counters include every "uncacheable" reason, see #732.
- the raw `.ninja_log` and `ccache --print-stats` output.

All of it is uploaded as the artifact `build-timings-<job>-a<run attempt>` with a
retention of 14 days, so the collector has time to ingest it; raw logs are never
committed. Only `build-static-lto` and `lint` use Ninja today, so the matrix and
MPI jobs have ccache data only until #705 lands.

A build that did not finish is flagged, not treated as a full build. The job
passes the outcome of its build step to the action (`build-status`), and
`complete` is true only if that step succeeded. This matters for `lint`, which
builds with `-k 0` and tolerates errors: its step records the real `ninja` exit
code for this purpose, and in practice its builds are partial (a GCC/Eigen
`-Werror` false positive in `mShake.cpp` and a few test objects fail). `ninja -n`
is deliberately not used to decide this: with LTO, GCC lists its temporary
`/tmp/cc*.ltrans*.o` files as dependencies of every link step, they are gone after
the link, and so `ninja -n` reports a finished `build-static-lto` build as having
165 steps left.

Reading the file fields: see "Build analysis summary" in [`SCHEMA.md`](SCHEMA.md).

**Ingestion.** For every new run of the workflows in `build_analysis_workflows`
(`BUILD`, `LINT`), the collector lists the run's artifacts, downloads the
`build-timings-*` ones and writes one `kind: "build-analysis"` record per job
next to the job records, in the same shard. Cost: one more API call per `BUILD`
or `LINT` run plus one per artifact (about 7 per `BUILD` run), which is the
largest part of the daily budget against the 1,000 requests per hour of
`GITHUB_TOKEN`. An artifact that expired or does not pass validation is counted
in the summary line (`dropped jobs: ...`) and skipped; any other API error makes
the whole run be retried on the next collection, so a transient failure does not
lose its analysis. Artifacts only exist from the day the jobs started uploading
them (#735), so nothing can be backfilled. Each record is 0.5 KB (ccache only) to
3.5 KB (with `.ninja_log` figures); expect about 6 MB a month on top of the job
records.

The overview shows the data in its "Build analysis" section (below).

## Clang build time comment (#750)

Every pull request that runs the `Clang Build` workflow (C++-relevant paths) gets one
comment with the clang build times, updated in place (marker
`<!-- clang-build-times-comment -->`), in the style of the performance gate. It is
informational and never a failing check.

- **Baseline.** A push to `dev` stores `clang-traces-detail.json` in the Actions cache
  (`clang-times-<dev sha>`, about 100 KB); a pull request restores the newest one and
  compares against it. Without one the comment says "No baseline yet" and shows only the
  pull request's numbers (the baseline is not built in the pull request). `ninja-includes.json`
  is stored next to it.
- **What it shows.** The totals (compiler, frontend, backend CPU time, header inclusions,
  template instantiation events) against `dev`, the **exact include graph** (include
  pairs and the project headers whose fan-in changed, from `ninja -t deps`: deterministic,
  so every change is real and there is no threshold), the headers that got heavier, and
  collapsed: lighter headers, the files and template instantiations that changed most
  and this pull request's slowest files, heaviest headers and templates.
- **Speed-adjusted.** The same code took between 882 s and 1,707 s of compiler time in five runs on shared
  runners, so the change of a single header, file or template is computed after scaling
  the baseline by the ratio of the two total compile times (a uniformly slower runner
  shows no changes). A change is listed only if it is at least 0.5 s (files 1 s) and
  15% (files and templates 20%). **This is a first-order correction only:** clang leaves
  events shorter than 0.5 ms out of the trace, so a faster run also records fewer events.
  The counts (shown without a percentage), the files-including-it column and the self time
  of headers made of many small events therefore depend on runner speed, and the
  "lighter" lists can show false changes (seen on identical code). The thresholds are
  provisional until the noise study (#751).
- **Limits.** Only the longest lists (100 files, 300 headers, 150 templates) are
  compared; a header that is not in the baseline's list is compared with the cut-off of
  that list and marked "new in the list".
- **Who runs what.** `clang-build` runs the pull request's code and has no write token:
  it renders the Markdown (`clang_report.py`), appends it to the job summary (the only
  output for forks) and uploads it. `clang-comment` has `pull-requests: write`, mints
  the `pq-perf-bot` token and only posts the downloaded file; it runs nothing from the
  pull request and is skipped for forks and `pq-bot/` branches, like the performance
  gate.

## Reading the data

```bash
# durations of every BUILD job on dev pushes in one week
jq -c 'select(.workflow=="BUILD" and .event=="push") | [.job, .duration_s]' \
  .github/ci-metrics/data/2026-W40.jsonl
```

Compare `push` runs and `pull_request` runs separately: they read different
cache scopes. Ignore `cancelled` jobs and `is_rerun` / `infra_failure`
records when computing typical durations.
