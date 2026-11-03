.. _codeStatistics:

###############
Code Statistics
###############

``scripts/pq_code_stats`` turns the history of ``dev`` into tables: how many lines and files every merged pull request
added or removed, split by area and kind, and what the repository looked like after every change. It reads git only
(one ``git log`` call), needs no packages and writes plain CSV files plus a self-contained HTML report outside the
repository. The README (``scripts/pq_code_stats/README.md``) lists every column.

********
Commands
********

.. code:: bash

    $ python3 scripts/pq_code_stats/pqstats.py collect          # history -> CSV files and report.html
    $ python3 scripts/pq_code_stats/pqstats.py collect --ci     # also the GitHub Actions runs (needs gh)
    $ python3 scripts/pq_code_stats/pqstats.py show --last 20   # the latest changes in the terminal
    $ python3 scripts/pq_code_stats/pqstats.py report           # rebuild report.html from the CSV files
    $ python3 scripts/pq_code_stats/pqstats.py branch           # what the checked out branch changes

``--out DIR`` (before the command) selects the output directory, otherwise ``$PQ_CODE_STATS_DIR`` or
``~/.local/share/pq-code-stats`` is used. ``collect`` also takes ``--since <date>``, ``--exclude-pr N`` for huge data
imports and ``--ref`` (default ``origin/dev``).

``show`` prints one line per change:

.. code:: text

          date  change  title                                         +lines  -lines  prod hdr  prod src  tests  perf  ci
    ----------  ------  --------------------------------------------  ------  ------  --------  --------  -----  ----  --
    2026-10-26  #884    518 Inject OptimizerSettings replace static      520     491        +2       +33     -7    +0  +0
    2026-10-27  #886    618 Inject ConvSettings replace static setti     483     469        +0        +8     +5    +0  +0
    2026-10-27  #876    fix: count intramolecular energy once in exa      75       4        +4        +2    +63    +0  +0

The four right-hand columns are the **net lines of C++ files**: header-like files (headers and ``.tpp``) and sources
of the production code (``src``, ``include``, ``apps``), the unit tests, and the performance code.

*********************
Check your own branch
*********************

``branch`` uses the same areas, kinds and counting rules for the checked out branch, so you see the numbers your pull
request will get. It reads git directly and needs no collected data.

.. code:: bash

    $ python3 scripts/pq_code_stats/pqstats.py branch              # committed changes since the merge base with origin/dev
    $ python3 scripts/pq_code_stats/pqstats.py branch -u           # plus uncommitted work
    $ python3 scripts/pq_code_stats/pqstats.py branch --files      # also every changed file

The first line tells you how many commits the branch is ahead of and behind ``dev``. *Behind* means that ``dev`` moved
on: merge it before you open the pull request (the ``pr_request`` check fails otherwise).

**********
The splits
**********

.. list-table::
    :header-rows: 1
    :widths: 18 82

    * - Dimension
      - Values
    * - area
      - ``src``, ``include``, ``apps``, ``tests``, ``integration_tests``, ``perf``, ``benchmarks``, ``ci_workflows``,
        ``ci_other``, ``ci_data``, ``scripts``, ``docs``, ... The rules are ``AREA_RULES`` in ``cs_classify.py``.
    * - group
      - ``production`` (src, include, apps), ``tests``, ``perf``, ``ci``, ``docs``, ``build``, ``data``, ``other``
    * - kind
      - ``header``, ``tpp``, ``source``, ``cmake``, ``python``, ``shell``, ``config``, ``doc``, ``other``
    * - class
      - ``header_like`` (a header or ``.tpp``: compiled into every file that includes it), ``source``, ``other``

The headline figures count **C++ files only**, because they answer two questions: how much test and performance code
there is per production line (a proxy for coverage), and how much of the production code is in sources and how much in
headers (a proxy for compile time: a lower share in sources means more is compiled again and again).
Submodules are not counted, lines are physical lines as git counts them, and a rename counts as a deletion plus an
addition. The numbers of a merged pull request are its net change against the first parent of ``dev``, which is what it
changed there, not the sum of its commits.

*********
The files
*********

.. list-table::
    :header-rows: 1
    :widths: 22 78

    * - File
      - One row per
    * - ``changes.csv``
      - change (merged pull request, other merge, direct commit): date, week, number, title, author, lines and files by
        group
    * - ``splits.csv``
      - change by area by kind: what was added, deleted, modified
    * - ``state.csv``
      - change by area by kind: the files and lines **after** the change
    * - ``metrics.csv``
      - change, as wide columns: C++ lines per scope and class, ``prod_source_share``, ``tests_to_prod``,
        ``perf_to_prod``, ...
    * - ``weekly.csv``
      - ISO week, including the empty ones: flows per group and scope, pull requests, CI runs
    * - ``ci_daily.csv``
      - day, workflow, event and conclusion (only with ``--ci``)
    * - ``meta.json``
      - the ref, the head, the filters and the result of the verification

The tidy long form (``splits.csv``, ``state.csv``) makes any custom plot a one-liner:

.. code:: python

    import pandas as pd
    m = pd.read_csv("metrics.csv", parse_dates=["date"]).set_index("date")
    m[["prod_source_share"]].plot(drawstyle="steps-post")        # lower = more code in headers
    m[["tests_to_prod", "perf_to_prod"]].plot(drawstyle="steps-post")

    s = pd.read_csv("state.csv", parse_dates=["date"])
    s[s.group == "production"].groupby(["date", "file_kind"]).lines.sum().unstack().ffill().plot.area()

********************
Trusting the numbers
********************

    - ``collect`` rolls the totals forward change by change and then **compares them with a full count of the tree**
      at the ref. If they differ it writes nothing (``--no-verify`` skips the check).
    - The early history contains huge data imports and a reverted merge. Use the C++ columns, or ``--exclude-pr`` for the
      flows; the totals stay correct.
    - The CI runs reach back as far as GitHub keeps them. GitHub forgets the pull request of a run when its branch is
      deleted, so a run is assigned to a pull request through its branch name. Older pull requests show ``0`` runs,
      and weeks outside the window are empty, not zero.
