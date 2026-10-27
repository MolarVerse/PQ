.. _continuousIntegration:

######################
Continuous Integration
######################

Every pull request and every push to the long-lived branches is checked by GitHub Actions. The workflow files are in
``.github/workflows/``. This page explains what runs when, how to run the same checks on your own machine, why a job
can be slow or red for reasons that have nothing to do with your change, and what to keep in mind when you edit a
workflow. The numbers behind the timing statements come from the :doc:`CI metrics <ciMetrics>`.

***********************
How the CI is organised
***********************

Three conventions apply to almost every workflow:

    - **A job named** ``changes`` **decides whether the heavy jobs run.** It compares the files of the pull request with a path
      filter, so a pull request that only touches documentation does not start the C++ builds.
    - **A gate job reports the result.** ``ci-build-gate``, ``lint-gate``, ``cpp-check-gate``, ``clang-format-gate`` and
      ``license-check-gate`` fail if one of their jobs failed and pass if the jobs passed *or were skipped*. A skipped
      job is therefore not a missing check.
    - **Only superseded pull request runs are cancelled.** Pushing to a pull request branch cancels the older run of
      the same workflow. Runs for pushes to ``dev`` and ``main``, the nightly run and manual runs each get their own
      concurrency group and are never cancelled, because they give the verdict for a commit and fill the cache of the
      branch.

*********
Workflows
*********

.. list-table::
    :header-rows: 1
    :widths: 18 18 24 40

    * - Workflow
      - File
      - Runs on
      - What it checks
    * - BUILD
      - ``ci_build.yml``
      - pull requests (any base branch), pushes to ``dev``, ``main`` and ``NEXT``, nightly at 02:00 UTC, manual
      - Builds and tests the project. See :ref:`the BUILD jobs <ciBuildJobs>` below.
    * - LINT
      - ``lint.yml``
      - pull requests (any base branch), pushes to ``dev``, ``main`` and ``NEXT``, manual
      - ``clangd-tidy`` on the changed C++ files (``scripts/clang_tidy.sh``). The job builds the project first,
        tolerating compile errors, only to get a compile database.
    * - DevOps C++ Check
      - ``devops_cpp_check.yml``
      - pull requests to ``main``, ``dev`` and ``NEXT``, manual
      - ``cpp_checks`` from ``external/devops``: style and AST rules, see :doc:`staticChecks`.
    * - Clang Format Check
      - ``clang_format.yml``
      - pull requests to ``main``, ``dev`` and ``NEXT``, manual
      - Every changed C++ file must be clean under clang-format 20.1.3, **as a whole file**, not only the changed lines.
    * - License Header Check
      - ``license_check.yml``
      - pull requests (any base branch), pushes to ``main``, manual
      - The exact GPL header of ``config/licenseHeader.txt`` in every C++ file.
    * - Changelog Fragment Check
      - ``changelog.yml``
      - pull requests to ``dev`` and ``NEXT``
      - Runs the tests of the scripts in ``scripts/tests`` and requires a valid changelog fragment,
        see :ref:`changelogFragments`.
    * - Check PR Has Linked Issue
      - ``gh_checks.yml``
      - pull request events, manual
      - The description or title contains ``Closes #N``, ``Fixes #N``, ``Resolves #N`` or ``Related to #N`` (or the
        pull request has an issue linked in GitHub). A phrase such as "Part of #N" is *not* recognised.
    * - Performance Gate
      - ``perf.yml``
      - pull requests to ``main``, ``dev`` and ``NEXT`` that touch C++ or build files
      - Builds ``benchmarks/perf`` for the pull request and its base, runs both under ``callgrind`` and fails if an
        instruction count grew beyond the threshold. Instruction counts are deterministic, so this check is not
        flaky.
    * - Clang Build
      - ``clang_build.yml``
      - pull requests that touch C++ or build files
      - Not a gate. Builds with clang and ``-ftime-trace`` and posts a comment with the build times, see
        :doc:`ciMetrics`.
    * - Docs
      - ``jekyll-gh-pages.yml``
      - pull requests to ``main`` and ``dev`` that touch ``docs``, ``include`` or ``src``; pushes to ``main``
      - Builds the Sphinx and Doxygen documentation; a push to ``main`` also deploys the website.
    * - CI Metrics
      - ``ci_metrics.yml``
      - daily at 04:30 UTC, manual
      - Collects the timing of all CI runs, see :doc:`ciMetrics`.
    * - Sync dev → NEXT
      - ``sync-dev-into-next.yml``
      - pushes to ``dev``, daily at 03:00 UTC, manual
      - Merges ``dev`` into ``NEXT``. On a conflict it opens a pull request from a ``merge/dev-into-NEXT-<time>``
        branch instead.
    * - Sync main → dev
      - ``sync-main-into-dev.yml``
      - pushes to ``main``, manual
      - Merges ``main`` back into ``dev`` after a release or a hotfix; on a conflict it opens a pull request from a
        ``merge/main-into-dev…`` branch.
    * - Create Release PR, Release Lifecycle
      - ``create-release-pr.yml``, ``release.yml``
      - manual; pull requests to ``main`` from ``release/*`` and ``hotfix/*``
      - The release process, see :doc:`branchesAndReleases`.
    * - Dismiss Stale Approvals
      - ``dismiss_stale_approvals.yml``
      - new commits on pull requests to ``main``, ``dev`` and ``NEXT``
      - Dismisses approvals when the new commits contain more than a merge of ``dev``.
    * - PQ Bot Coworker, PQ Bot Review
      - ``pq-bot.yml``, ``pq-bot-review.yml``
      - comments on issues and pull requests
      - The :ref:`PQ Bot <pqbot>`.

.. _ciBuildJobs:

==============
The BUILD jobs
==============

    - ``build``: a matrix of x86_64 and arm, each as Release and as Debug, built with Ninja. The Debug build on
      x86_64 also measures the test coverage (``gcovr``). The Release builds run the integration tests (see below).
      The Debug builds use the strict warning set with ``-Werror``, and the arm Debug job also builds the
      ``benchmarks/perf`` and Google Benchmark sources, so that they are held to the same warnings.
    - ``benchmark``: runs the Google Benchmark suite as a smoke test.
    - ``build-static-lto``: a static build with link-time optimisation. Its tests only run in the nightly run, which
      builds ``main``.
    - ``static-singularity-build``: only in the nightly and manual runs.
    - ``mpi-build``: builds with MPI and runs the tests.
    - ``python-bindings``: builds ``PQ`` with ``BUILD_WITH_PYTHON_BINDINGS=ON`` and runs the ``testPQCli`` test, which
      compares the capabilities that ``PQ`` reports with the configuration.

The integration tests run on both the x86_64 and the arm Release build **on purpose** (decided in issue #706):
``dftbplus``, ``xtb`` and ``pyscf`` come from architecture-specific conda-forge packages, so the arm run exercises
different binaries. It costs no wall-clock time, because the arm Release job is not the slowest job.

**************************
Running the checks locally
**************************

The checks that fail most often can be run before you push:

.. list-table::
    :header-rows: 1
    :widths: 22 78

    * - Check
      - How to run it
    * - Build and unit tests
      - .. code:: bash

            $ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_WITH_NATIVE=OFF
            $ cmake --build build --parallel
            $ ctest --test-dir build --output-on-failure

        Use the compiler version of ``.github/actions/setup-gcc`` (the default of its ``version`` input) if you want
        the same warnings as CI.
    * - clang-format
      - ``pip install clang-format==20.1.3``, then ``clang-format -i <file>`` for every C++ file you changed. A
        different version can format differently.
    * - License header
      - ``bash scripts/addLicense.sh``
    * - DevOps C++ checks
      - See :ref:`the setup <runningCppChecks>` on the static checks page.
    * - clang-tidy
      - ``scripts/clang_tidy.sh -p <build dir with compile_commands.json>``, see :doc:`staticChecks`.
    * - Changelog fragment
      - .. code:: bash

            $ python3 scripts/check_changelog_fragment.py origin/dev HEAD

        The check looks at *committed* changes, so commit the fragment first.
    * - Tests of the scripts
      - .. code:: bash

            $ python3 -m unittest discover -s scripts/tests -p 'test_*.py'
    * - Documentation
      - .. code:: bash

            $ cmake -S . -B build -DBUILD_WITH_DOCS=ON -DBUILD_WITH_ASE=OFF -DBUILD_WITH_TESTS=OFF -DPQ_DOXYGEN_GRAPHS=OFF
            $ cmake --build build --target sphinx-docs

        This needs ``doxygen``, ``graphviz`` and the Python packages listed in :ref:`the Sphinx section <sphinxBuild>`.
        ``PQ_DOXYGEN_GRAPHS=OFF`` skips the dot graphs, which is what pull request builds do.

****************
Caches and speed
****************

A job that takes twice as long as usual is rarely a problem in your change. These are the effects to look at first:

    - **ccache.** Every build job keeps a ccache per job and branch scope. ``CCACHE_SLOPPINESS=pch_defines,time_macros``
      is set, because without it ccache cannot cache compiles that use a precompiled header (about two thirds of all
      compiler calls were uncacheable before, issue #732). The Debug and the lint caches are allowed 1.5 GB, the others
      500 MB; a cache that is at its limit evicts what the next build needs.
    - **A header change is a cold cache.** Touching a header that most files include recompiles most files. The x86_64
      Release job went from about 2.5 to 9 minutes for a pull request that changed ``exceptionTypes.hpp``. The same is
      true for your local build, see :doc:`localBuildTimes`.
    - **The runner CPU.** GitHub gives the same job runners of different CPU generations. The same fixed compile took
      4.8 to 5.1 s on an AMD EPYC 9V45 and 7.3 to 8.7 s on an EPYC 7763. The CI metrics record the CPU model of the
      runner, and the clang comment of a pull request names the CPU of the run and of its baseline.
    - **The Eigen source** is cached under a key that hashes ``.cmake/eigen.cmake`` and passed to CMake with
      ``-DFETCHCONTENT_SOURCE_DIR_EIGEN``, so that no job clones it again.
    - **The conda environment** of the integration tests is cached under a key that hashes ``ci_build.yml``. **Every
      edit of that file, even a comment, makes the next run install the environment from scratch on both
      architectures.** Keep that in mind before you touch it, and prefer to document a decision elsewhere.
    - **The cache limit.** The repository's Actions cache is limited to 10 GB and the oldest entries are evicted. Each
      pull request saves its own ccache entries, which only that pull request can read.

*********************************
When a job fails without a reason
*********************************

A job that sits in the queue and is cancelled after about 15 minutes, with no runner assigned and no steps, was never
started: the runner pool or GitHub Actions had a problem (check https://www.githubstatus.com). Re-running does not
help during such an outage and does not need a code change; re-run the failed jobs once Actions has recovered. A job
that did start and fails is a real result, even if it looks unrelated: read its log first.

*****************************
Adding or changing a workflow
*****************************

    - Follow the pattern above: a ``changes`` job, the heavy jobs depending on it, and a gate job with ``if: always()``.
    - ``scripts/tests/test_workflow_path_filters.py`` fails if a ``dorny/paths-filter`` filter contains a negated
      pattern: the patterns of a filter are ORed, and a negated pattern on its own matches every other file.
    - Workflows that need a write token mint one from a GitHub App (``actions/create-github-app-token``, pinned by
      commit). Check which apps are installed on the organisation before you rely on a secret: the sync workflow used an
      app that was not installed and its token step failed with "Not Found" (issue #804). Today the release bot and the
      performance bot are installed.
    - A push made with the default ``GITHUB_TOKEN`` does not start other workflows. A branch that has to trigger CI
      needs a token of a GitHub App.
    - Prefer the shared actions in ``.github/actions``: ``setup-gcc`` installs the pinned GCC once for all workflows and
      ``upload-build-timings`` records the build analysis for the :doc:`CI metrics <ciMetrics>`.
