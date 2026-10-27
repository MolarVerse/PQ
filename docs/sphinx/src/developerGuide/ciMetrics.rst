.. _ciMetrics:

##########
CI Metrics
##########

The CI metrics are a durable record of how long the CI jobs take and why, so that speed-ups are measured and not
guessed. They consist of a collector that stores one record per CI job, an overview page that is generated from the
records, and a comment on every pull request with the clang build times. This page tells you how to use them and what
to be careful about. The complete reference (record fields, derivations, decisions) is in the repository:
``.github/ci-metrics/README.md`` and ``.github/ci-metrics/SCHEMA.md``.

For build times on *your* machine, which are independent of all of this, see :doc:`localBuildTimes`.

****************
What is recorded
****************

    - **Job records.** For every finished job of the recorded workflows (BUILD, LINT, Performance Gate, DevOps C++
      Check, Clang Format Check, License Header Check, Changelog Fragment Check, Check PR Has Linked Issue, Dismiss Stale
      Approvals, Docs): duration, queue time, conclusion, event, branch, the pull request, whether it was a re-run and
      whether it failed because of the infrastructure.
    - **Build analysis records.** The jobs that build the project run the action
      ``.github/actions/upload-build-timings`` as their last step. It summarises the Ninja log (wall and CPU time,
      parallelism, the slowest steps, the time spent linking at the end), the ccache statistics (hit rate and the reasons
      why calls were not cacheable), the exact include graph of the build (from ``ninja -t deps``, deterministic), and
      the CPU model of the runner. The Clang Build workflow adds a summary of the ``-ftime-trace`` files on pushes to
      ``dev``.

The data is stored in weekly files ``.github/ci-metrics/data/<year>-W<week>.jsonl`` that are **committed to** ``dev``.
That is the one documented exception to the rule that nobody pushes to ``dev``: only the collector writes there, only
below ``.github/ci-metrics/data/``, and nothing in that directory is edited by hand.

*************
The collector
*************

``.github/workflows/ci_metrics.yml`` runs ``collect.py`` daily at 04:30 UTC, looks back three days and commits the new
records. You can run the same script yourself (it needs the ``gh`` command line tool to be logged in):

.. code:: bash

    $ python3 .github/ci-metrics/collect.py                       # the last 3 days, incremental
    $ python3 .github/ci-metrics/collect.py --since-days 90       # backfill
    $ python3 .github/ci-metrics/collect.py --dry-run             # write nothing
    $ python3 .github/ci-metrics/collect.py --data-dir /tmp/x     # try it without touching data/

It is idempotent: jobs that are already recorded are never written again. A workflow's ``GITHUB_TOKEN`` may make about
1,000 API requests per hour; a backfill of three months needs more than that and stops on the rate limit (use
``--rate-limit-wait``).

.. note::

    **The scheduled collection only starts when** ``ci_metrics.yml`` **is on** ``main``. GitHub runs a ``schedule`` (and
    a ``workflow_dispatch``) only for the workflow file of the default branch. Until a release brings the file to
    ``main`` the data stops at the last manual collection. A manual run on another ref never pushes; it is a dry run.

************
The overview
************

``.github/ci-metrics/data/CI_TIMINGS.md`` is generated from the records (``report.py``) after every collection and
rendered by GitHub. It shows, per workflow:

    - the median and the 90th percentile of the job durations in the current window (14 days) next to the previous
      window, split into pushes to ``dev`` and pull requests, with the queue time apart from the duration;
    - **regressions**: a median that rose by more than 20% *and* 30 seconds, with at least five samples in both windows;
    - the time a pull request really waits for a workflow (first job created to last job finished) and the job that
      usually finishes last;
    - the weekly trend of the BUILD duration;
    - the **build analysis**: the ccache hit rate and what blocks it, the Ninja figures of complete builds, the slowest
      steps on ``dev``, and a table of the build time per runner CPU model;
    - the heaviest headers and slowest files of the clang build, as a share of the total compile time;
    - the **include graph**: how many include pairs and project header pairs the latest builds have, and the headers
      with the highest fan-in.

Read it with these cautions in mind:

    - Compare pushes and pull requests separately: they read different cache scopes.
    - A job that is slower is first a question of the **ccache hit rate**: the same job takes about 2.5 minutes with a
      99% hit rate and about 10 minutes with less than 50%.
    - The runner CPU changes the time of identical code by up to 1.8 times. Look at the CPU table before you call
      something a regression.
    - The include graph numbers are exact. Everything measured in seconds is noisy.

.. _ciMetricsClangComment:

***********************
The clang build comment
***********************

Every pull request that changes C++ or build files gets one comment (updated in place) from the ``Clang Build``
workflow. The workflow builds the pull request with clang 20 and ``-ftime-trace``, compares the result with the
newest ``dev`` build that is stored in the Actions cache, and posts:

    - the totals (compiler, frontend and backend CPU time, header inclusions, template instantiation events) against
      ``dev``;
    - the **exact include graph change**: include pairs and the project headers whose fan-in changed. This part is
      deterministic, so every change in it is real;
    - the headers that got heavier, and under a collapsed section the lighter ones, the files and the templates that
      changed most;
    - the CPU model of this run and of the baseline.

The comment is informational and never a failing check. **Do not read the timings as exact.** The baseline is the
newest ``dev`` build, so commits that were merged to ``dev`` since then also show up. Changes of single headers, files and
templates are computed after scaling the baseline by the ratio of the two total compile times, which removes a
uniformly slower runner, but clang leaves events shorter than 0.5 ms out of the trace, so a faster run records fewer
events and counts and small times depend on the speed of the runner. Use the include graph section for conclusions
about what the code does and the timing sections as hints.

If there is no baseline yet, the comment says so and shows only the numbers of the pull request.

***************************
Adding a job to the metrics
***************************

    - A new workflow is recorded when its name is in ``.github/ci-metrics/config.json`` (``workflows``).
    - A job that builds the project and should have a build analysis runs ``./.github/actions/upload-build-timings`` as a
      step with ``if: always()`` and passes the outcome of its build step as ``build-status``, so that a partial build is
      not mistaken for a full one. The workflow must be listed in ``build_analysis_workflows``.
    - The artifacts of the build analysis are kept for 14 days. They are what the collector ingests, so a job that
      uploads them can be recorded the next day at the latest.
    - The scripts have offline tests in ``scripts/tests/test_ci_metrics_*.py``. They run together with the other script
      tests in the Changelog Fragment Check.
