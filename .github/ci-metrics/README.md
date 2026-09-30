# CI metrics

Durable record of how long our CI jobs take, so CI speedup work can be measured
instead of guessed. Tracked in #720.

| Part | Status |
| --- | --- |
| Storage layout and record schema (this directory) | defined, see [`SCHEMA.md`](SCHEMA.md) |
| Collector script (GitHub API to JSONL), see [Running the collector](#running-the-collector) | implemented, #715 |
| Collector workflow, see [The workflow](#the-workflow) | implemented, #716 (inert until it reaches `main`) |
| Overview report for `dev` | planned, #717 |
| Build timings (`.ninja_log`, ccache, clang `-ftime-trace`) | planned, #718 and #719 |

## Layout

```text
.github/ci-metrics/
  README.md      this file
  SCHEMA.md      field definitions and derivation rules
  schema.json    JSON Schema for one record
  config.json    repository, recorded workflows, infrastructure-failure signatures
  collect.py     the collector (Python standard library only, uses the gh CLI)
  publish.sh     commits and pushes new data (used by the workflow)
  data/          weekly JSONL shards, written only by the collector
```

The workflow is `.github/workflows/ci_metrics.yml`. Tests:
`scripts/tests/test_ci_metrics_publish.py` (real local git repositories).

Tests: `scripts/tests/test_ci_metrics_collect.py` (offline; runs with the other
script tests in CI).

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
  taking 18-43 seconds with the default 8 workers.
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

## Reading the data

```bash
# durations of every BUILD job on dev pushes in one week
jq -c 'select(.workflow=="BUILD" and .event=="push") | [.job, .duration_s]' \
  .github/ci-metrics/data/2026-W40.jsonl
```

Compare `push` runs and `pull_request` runs separately: they read different
cache scopes. Ignore `cancelled` jobs and `is_rerun` / `infra_failure`
records when computing typical durations.
