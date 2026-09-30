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
- `kind` is `"job"` today. `"build-analysis"` is reserved for per-target build
  timings (`.ninja_log`, ccache, clang `-ftime-trace`) and will get its own
  definition when those are added.

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
| `pr_number` | int or null | Only set for `pull_request` events. The run's own `pull_requests` list is emptied by GitHub once the PR is merged and the branch deleted, so the collector resolves it from `GET /repos/{repo}/commits/{head_sha}/pulls`, choosing the PR whose `head.ref` equals `branch`. |
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
| `eigen_cache_hit` | `true` if the step `Clone Eigen (cache miss)` was skipped (the Eigen cache was restored), `false` if it ran, `null` if the job has no such step. Derived from step conclusions, no log parsing. Depends on that step name. |
| `is_rerun` | `true` if `run_attempt > 1`. |
| `infra_failure` | For `failure` jobs only: `true` if the job log contains a known infrastructure-outage signature (currently `GitLab is currently unable to handle this request`), `false` if it failed without one, `null` if the job did not fail or its log was unavailable (logs expire after 90 days). |

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
