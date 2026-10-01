# CI timing record schema (version 1)

Machine-readable definition: [`schema.json`](schema.json) (JSON Schema draft-07).
This page explains what each field means and how the collector derives it.

## Files and identity

- One JSON object per line ("JSONL"), one line per **finished job**.
- Files are sharded by the ISO week (UTC) of the *run's* `created_at`, so every
  job of a run lands in the same file: `data/2026-W40.jsonl`.
- `job_id` is globally unique (a rerun creates new job ids), so it is the
  idempotency key: the collector skips a job whose `job_id` is already present
  in its shard. It keeps no other state.
- `schema_version` is bumped on any incompatible change. Readers must ignore
  records whose version they do not know.
- `kind` is `"job"` (one per finished job, `schema.json`) or `"build-analysis"`
  (one per build job that uploaded a summary, see below,
  `schema-build-analysis.json`). Clang `-ftime-trace` data will get its own kind.
  Readers that only want job timings must skip records whose `kind` is not `"job"`.

## Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | int | Always `1` for this document. |
| `kind` | string | Always `"job"` here. |
| `workflow` | string | Workflow `name:` (for example `BUILD`, `LINT`). |
| `run_id`, `run_number`, `run_attempt` | int | GitHub run identifiers; `run_attempt` is 1 for the first attempt. |
| `event` | enum | `pull_request`, `push`, `schedule` or `workflow_dispatch`. Timings must be compared per event: `dev` pushes read `dev`'s cache scope, PR runs can read `dev`'s cache. |
| `branch` | string | The run's `head_branch` (the PR branch for `pull_request`). |
| `head_sha` | string | Commit the run built. For `pull_request` this is the PR head commit, not the merge ref. |
| `pr_number` | int or null | Only set for `pull_request` events. The run's own `pull_requests` list is emptied by GitHub once the PR is merged and the branch deleted, so the collector resolves it in two steps. (1) `GET /repos/{repo}/commits/{head_sha}/pulls`, choosing the PR whose `head.ref` equals `branch`. That endpoint returns the PR that *merged* a commit, not the PR whose head it is, so it misses some runs (for example a stacked PR whose head is the merge commit of another PR). (2) Fallback: `GET /repos/{repo}/pulls?state=all&head={owner}:{branch}`, choosing the most recently created PR that was open when the run was created (a branch name can be reused by several PRs over time). It is `null` only if neither finds a PR, for instance for a PR from a fork, whose head is `{fork-owner}:{branch}`. The first backfill wrote 450 of its 10,019 `pull_request` records with `null` because step (2) did not exist yet; all 450 were repaired with `collect.py --fix-pr-numbers`. |
| `job_id`, `job` | int, string | Job id and its full display name, including matrix values, for example `build (ubuntu-24.04-arm, Debug)`. |
| `runner_labels` | string[] | The job's runner labels. |
| `arch` | enum | `arm64` if any label contains `arm`, otherwise `x86_64`. |
| `created_at`, `started_at`, `completed_at` | timestamp | UTC, `YYYY-MM-DDTHH:MM:SSZ` (the API has 1-second resolution). |
| `queue_s` | int | `started_at - created_at`: time waiting for a runner. Kept apart from `duration_s`. |
| `duration_s` | int | `completed_at - started_at`. |
| `conclusion` | enum | `success`, `failure`, `cancelled`, `timed_out`, `neutral`, `action_required`. |
| `steps` | array | `{name, seconds, conclusion}` per executed step (`success`, `failure` or `cancelled`), in order. |
| `flags` | object | Derived facts, see below. |

### Flags

| Flag | Value |
| --- | --- |
| `eigen_cache_hit` | `true` if the step `Clone Eigen (cache miss)` was skipped (the Eigen cache was restored), `false` if it ran, `null` if the job has no such step **or the preceding `Cache Eigen source` step did not succeed** (a job that failed earlier also shows the clone step as skipped). Derived from step conclusions, no log parsing. Depends on those two step names. |
| `is_rerun` | `true` if `run_attempt > 1`. |
| `infra_failure` | For `failure` jobs only: `true` if the job log contains a known infrastructure-outage signature (currently `GitLab is currently unable to handle this request`), `false` if it failed without one, `null` if the job did not fail or its log was unavailable (logs expire after 90 days). |

## Build analysis summary (job side, version 1)

Written by `summarise_build.py` inside a CI job and uploaded as an artifact
(`build-timings-<job>-a<attempt>`). The collector turns it into a `kind:
"build-analysis"` record (`schema-build-analysis.json`) next to the job record,
for the workflows listed in `config.json` under `build_analysis_workflows`
(`BUILD` and `LINT`). Every part may be `null`.

**The record.** `workflow`, `run_id`, `run_attempt`, `event`, `branch`,
`head_sha`, `job_id`, `job`, `created_at` and `conclusion` are copied from the
job record and so are never taken from the artifact; `ninja` and `ccache` are the
fields below. Join to the job record on `job_id`. The collector accepts an
artifact only if its `run_id`, `run_attempt` and `job_id` match a job of that run.

**Trust.** An artifact of a pull request run is produced by code from that pull
request, so the collector never copies it: `sanitise_summary()` rebuilds every
field from a whitelist with type and range checks (finite numbers up to fixed
limits, at most 20 slow steps, at most 200 ccache counters with names matching
`[a-z0-9_]{1,64}`, printable targets of at most 300 characters, artifacts of at
most 1 MB) and drops an artifact that does not pass, counting it in the summary
line. Values inside those limits can still be wrong if a pull request wants them
to be, so treat these records as measurements, not as proof.

**Fields of the artifact.** (`run_id`, `run_attempt` and `job_id` are only used
for the join; `job_key` and `artifact` are not stored.)

| Field | Meaning |
| --- | --- |
| `run_id`, `run_attempt` | From the job's environment. |
| `job_id` | REST API id of the job (`job.check_run_id`), the join key to the `job` record. `null` if the runner did not provide it. |
| `job_key` | `GITHUB_JOB`, the job's key in the workflow file. |
| `artifact` | Name the summary was uploaded under. |
| `ninja` | `null` if the build directory has no `.ninja_log`. |
| `ninja.log_version` | `.ninja_log` format version; 5, 6 and 7 are read. |
| `ninja.complete` | `true` if the job's build step succeeded, `false` if it failed (for example `lint`'s `-k 0` build with errors), `null` if the job did not say. A `false` build must not be compared with full builds. It is not derived from `ninja -n`, which is wrong for LTO links. |
| `ninja.steps`, `wall_s`, `cpu_s` | Distinct commands run, time from the first start to the last end, sum of step times. A rebuilt output keeps its last entry; outputs of one command count once. |
| `ninja.parallelism` | `cpu_s / wall_s`. |
| `ninja.tail_after_compile_s` | Time from the last compile step ending to the end of the build (mostly linking). |
| `ninja.by_kind` | `steps` and `cpu_s` per `compile` (`.o`, `.gch`), `archive` (`.a`), `link` (executables and shared libraries) and `other`, classified by output file name. |
| `ninja.slowest` | The 20 slowest steps: `target`, `kind`, `seconds`. |
| `ninja.error` | Present instead of the figures if the log had an unsupported version or no steps. |
| `ccache` | `null` if ccache statistics were unavailable. |
| `ccache.hits`, `misses`, `hit_rate` | Direct plus preprocessed hits, misses, and `hits / (hits + misses)` (of *cacheable* calls; `null` if there were none). |
| `ccache.counters` | Every non-zero counter of `ccache --print-stats`, including the reasons calls were uncacheable. |

The ccache counters cover the job only, because the setup action zeroes them at
the start.

## What is and is not recorded

- Runs are collected only once `completed`.
- Jobs with `conclusion: skipped`, or without a start or end time, are omitted.
- `cancelled` jobs that did start **are** recorded (with their conclusion) so
  that analysis can see them; reports exclude them by default.
- Steps that were `skipped` are omitted (they have no timing). Flags are
  computed from the full step list *before* omission.
- All attempts of a run are recorded. `GET .../runs/{id}/jobs` returns only the
  latest attempt by default, so the collector must request
  `.../runs/{id}/jobs?filter=all` (one call, all attempts; checked on a real
  rerun). `.../runs/{id}/attempts/{n}/jobs` returns a single attempt if needed.

## Example

A real record (BUILD, arm64 Debug job of a `pull_request` run):

```json
{
  "schema_version": 1,
  "kind": "job",
  "workflow": "BUILD",
  "run_id": 36693119228,
  "run_number": 1329,
  "run_attempt": 1,
  "event": "pull_request",
  "branch": "ci/cancel-only-on-pull-requests",
  "head_sha": "8f61c01c5fb279060dd6192d54409bbf938b274b",
  "pr_number": 721,
  "job_id": 109814753706,
  "job": "build (ubuntu-24.04-arm, Debug)",
  "runner_labels": [
    "ubuntu-24.04-arm"
  ],
  "arch": "arm64",
  "created_at": "2026-09-30T08:58:58Z",
  "started_at": "2026-09-30T08:59:02Z",
  "completed_at": "2026-09-30T09:06:57Z",
  "queue_s": 4,
  "duration_s": 475,
  "conclusion": "success",
  "steps": [
    {
      "name": "Set up job",
      "seconds": 2,
      "conclusion": "success"
    },
    {
      "name": "Run actions/checkout@v4",
      "seconds": 2,
      "conclusion": "success"
    },
    {
      "name": "install gcc13",
      "seconds": 15,
      "conclusion": "success"
    },
    {
      "name": "setup ccache",
      "seconds": 8,
      "conclusion": "success"
    },
    {
      "name": "Cache Eigen source",
      "seconds": 1,
      "conclusion": "success"
    },
    {
      "name": "setup python ubuntu",
      "seconds": 0,
      "conclusion": "success"
    },
    {
      "name": "install python dependencies",
      "seconds": 3,
      "conclusion": "success"
    },
    {
      "name": "Build and Test Project",
      "seconds": 439,
      "conclusion": "success"
    },
    {
      "name": "Post setup python ubuntu",
      "seconds": 0,
      "conclusion": "success"
    },
    {
      "name": "Post Cache Eigen source",
      "seconds": 1,
      "conclusion": "success"
    },
    {
      "name": "Post setup ccache",
      "seconds": 2,
      "conclusion": "success"
    },
    {
      "name": "Post Run actions/checkout@v4",
      "seconds": 0,
      "conclusion": "success"
    },
    {
      "name": "Complete job",
      "seconds": 0,
      "conclusion": "success"
    }
  ],
  "flags": {
    "eigen_cache_hit": true,
    "is_rerun": false,
    "infra_failure": null
  }
}
```

## Volume

Measured on the full set of runs of one PR push (all recorded workflows): 23
records, about 26 KB uncompressed (about 1.2 KB per record). The repository saw
between 6 and 27 PR-triggered BUILD runs per day, so expect roughly 4-16 MB per
month uncompressed and about 1 MB per month in git. Weekly shards keep single
files at a few MB.
