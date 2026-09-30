# CI metrics

Durable record of how long our CI jobs take, so CI speedup work can be measured
instead of guessed. Tracked in #720.

| Part | Status |
| --- | --- |
| Storage layout and record schema (this directory) | defined, see [`SCHEMA.md`](SCHEMA.md) |
| Collector script (GitHub API to JSONL) | planned, #715 |
| Collector workflow | planned, #716 |
| Overview report for `dev` | planned, #717 |
| Build timings (`.ninja_log`, ccache, clang `-ftime-trace`) | planned, #718 and #719 |

## Layout

```text
.github/ci-metrics/
  README.md      this file
  SCHEMA.md      field definitions and derivation rules
  schema.json    JSON Schema for one record
  data/          weekly JSONL shards, written only by the collector
```

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

## Reading the data

```bash
# durations of every BUILD job on dev pushes in one week
jq -c 'select(.workflow=="BUILD" and .event=="push") | [.job, .duration_s]' \
  .github/ci-metrics/data/2026-W40.jsonl
```

Compare `push` runs and `pull_request` runs separately: they read different
cache scopes. Ignore `cancelled` jobs and `is_rerun` / `infra_failure`
records when computing typical durations.
