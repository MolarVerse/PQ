.. _buildTimes:

#################
Local Build Times
#################

``scripts/pq_build_times`` measures and explains how long **your** machine needs to build PQ, so that a change that is
meant to speed up the build can be judged by numbers. It uses only the Python standard library, keeps its data outside
the repository and has no connection to the :doc:`CI metrics <ciMetrics>`. The README next to the scripts
(``scripts/pq_build_times/README.md``) is the complete reference; this page shows the workflow.

********************
The improvement loop
********************

.. code:: bash

    $ python3 scripts/pq_build_times/pqbt.py snapshot --note "baseline"     # measure and store
    $ python3 scripts/pq_build_times/pqbt.py baseline set                   # pin it as the reference
    # ... change something ...
    $ python3 scripts/pq_build_times/pqbt.py snapshot --note "forward-declare X in a.hpp"
    $ python3 scripts/pq_build_times/pqbt.py compare                        # baseline against the latest
    $ python3 scripts/pq_build_times/pqbt.py report                         # table and HTML graph
    $ python3 scripts/pq_build_times/pqbt.py detail                         # where does the time go?

A snapshot configures and builds in a directory the tool owns (``build-times/``, ignored by git) and runs five
scenarios:

.. list-table::
    :header-rows: 1
    :widths: 24 76

    * - Scenario
      - What it answers
    * - ``cold``
      - Configure and build everything from nothing. The dependencies come from a cache, so the network is not timed.
    * - ``noop``
      - What does "nothing to do" cost?
    * - ``touch_leaf``
      - A median source file changed: the everyday edit.
    * - ``touch_header_top``
      - The most widely included project header changed: the worst case.
    * - ``touch_header_median``
      - A median shared header changed.

Besides the wall time it stores the **steps rebuilt**, which is deterministic and does not depend on the speed of the
machine, the CPU time, the parallelism, the link time, the run-to-run spread and the exact include graph. ccache is
off unless you pass ``--ccache``, because cache hits hide the compile time.

Useful options of ``snapshot``:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Option
      - Meaning
    * - ``--target``
      - Build only one ninja target. A different series (the target is part of the fingerprint); good for trying an
        idea quickly.
    * - ``--scenarios``
      - A comma separated subset, for example ``cold,touch_leaf``.
    * - ``--repeat``, ``--cold-runs``
      - Repetitions of the cheap scenarios (default 3, the median is stored) and of the cold build.
    * - ``--compiler clang++-20``
      - Build with clang and ``-ftime-trace``; see :ref:`below <buildTimesClang>`. A separate series.
    * - ``--cmake-arg=-DX=Y``
      - Extra CMake argument, recorded in the fingerprint. Write it **with an equals sign**: argparse takes a value
        that starts with a dash for an option.
    * - ``--note``
      - What changed since the last snapshot. It becomes a marker in the graph.
    * - ``--force``
      - Measure although the machine looks busy or a submodule is not at its recorded commit.

*******************
Series and baseline
*******************

Every snapshot carries a fingerprint: architecture, CPU model, number of threads, compiler, build type, ninja target,
CMake arguments, ``-j`` and ccache. **Only snapshots with the same fingerprint are compared or plotted together.**
Change the compiler or a flag and you start a new series with its own baseline (``pqbt.py list`` shows them, and
``compare`` names the keys that differ if you try to mix two). The touched files are chosen from the include graph and
pinned to the first snapshot of a series.

The data is stored in ``$PQ_BUILD_TIMES_DIR``, else ``$XDG_DATA_HOME/pq-build-times``, else
``~/.local/share/pq-build-times`` (``--data-dir`` overrides all of them). The baseline is pinned explicitly and never
overwritten by itself.

A snapshot is refused while a submodule that the build uses (for example ``external/mstd``) is not at the commit that
the repository records, because the build would fail or measure something else:

.. code:: bash

    $ git submodule update --init external/mstd

*******************
Reading ``compare``
*******************

.. code:: text

    Deterministic figures (independent of machine speed)
                           before        after         change
    ---------------------  ------------  ------------  ---------
    objects                59            59            +0.0%
    include pairs          18,028        18,028        +0.0%
    project include pairs  1,010         1,010         +0.0%
    include graph          0b5e44c6bb5f  0b5e44c6bb5f  identical

    Scenarios (steps are deterministic, times are judged against the noise)
    scenario             steps     wall            change  cpu                   change  noise  verdict
    -------------------  --------  --------------  ------  --------------------  ------  -----  ---------
    touch source file    12 -> 12  0.50s -> 0.46s  -7.9%   0.54s -> 0.50s        -7.9%   +-8%   unchanged
    touch top header     43 -> 43  6.31s -> 5.82s  -7.7%   1m 23.2s -> 1m 17.2s  -7.3%   +-91%  unchanged
    touch median header  15 -> 15  4.08s -> 4.15s  +1.8%   18.5s -> 18.6s        +0.6%   +-6%   unchanged

The first block is **fact**: the sizes of the include graph and a digest of it (``identical`` means no include was
added or removed). The second block combines two kinds of evidence:

    - ``improved`` or ``regressed`` needs **fewer or more steps and** a timing change beyond the noise threshold.
    - A timing change with the same steps, or a change in the other direction, is ``check noise`` and is *not*
      reported as an effect of your change.
    - The noise threshold is twice the larger run-to-run spread of the two snapshots, at least 5% (10% for a single
      run), and a change below 0.05 s is never a change.

In the example nothing changed, and the tool says so although the numbers moved by up to 8%.

If you have two snapshots of a cold build with details, ``compare`` also lists **what changed in the detail**: first
the exact changes (translation units that now include more or fewer repository files, headers whose fan-in changed),
then the compile CPU per module after removing the drift of the whole build.

Keep the machine quiet
======================

The tool refuses to start while the machine is busy (a load of more than 25% of the threads) and measures **how much CPU
other processes used while the build ran**. On an idle machine that is about 2% of the build's CPU. Above 5% and 10 s
``compare`` and ``detail`` print a warning like this one:

.. code:: text

    warning: other processes used about 76 s of CPU during the touch top header of 20261027T191639Z (99% of the
    build's CPU); its timings are inflated, repeat it on an idle machine

Take that warning seriously. In a test, four busy loops during a cold build moved the timing by only +6.6%, which is
inside the band of a single run, so without the warning nobody could have told. The example above is why the
``touch top header`` row says ``+-91%`` noise and ``unchanged``: another job on the same host was running.

******************
Reading ``detail``
******************

``detail`` explains one snapshot. It is made from what Ninja recorded (when every step ran, which files every object
includes) and from git (how often a file changed in the last 180 days):

.. code:: text

    == Build at a glance ==
    cold build: wall 9.5s, CPU 2m 04s, 78 steps (59 compiles), 32 parallel jobs
    parallelism: 13.1 running steps on average = 40.8% of the 32 job slots
    the wall time is bound by idle cores: a serial phase or a few long steps limit the wall time
    precompiled header pq_test_pch: 3.3s, ready after 4.3s (45.6% of the wall time)
    tail: steps still running in the last 10% of the build: gtest-all.cc.o (9.4s); testThermostat (0.04s)

    == Where the CPU goes ==
                          steps     CPU  share
    --------------------  -----  ------  -----
    src (project)            54  1m 36s  78.1%
      src/molsys              7   27.7s  22.4%
      src/settings           19   22.9s  18.4%
      src/thermostat          5   15.3s  12.4%

    == Slowest compile steps ==
    time  % CPU    ran  repo files  PRs  file
    9.4s   7.6%  0-9s*          33    -  external/googletest/googletest/src/gtest-all.cc
    6.9s   5.5%  0-7s          164   16  src/molsys/simulationBox.cpp
    concentration: 13 files (22.0%) make up half of the compile CPU, 25 files (42.4%) four fifths

(This is a build of the target ``testThermostat`` only, to keep the example short.) How to read the sections:

    - **Build at a glance** tells you whether *fewer includes* help (the cores are busy: the build is CPU-bound) or
      *a long step* does (the cores idle: a serial phase such as the precompiled header, or a long tail).
    - **Where the CPU goes** splits the compile CPU into project modules, third party code, precompiled headers, tests
      and linking.
    - **Slowest compile steps** shows the time, the share, the number of repository files the file pulls in and the number
      of merged pull requests that touched it (``PRs`` over the history window). A slow file that is also edited often is
      worth more than a slow file nobody touches.
    - The sections after these rank headers by their **burden** (rebuild cost times how often they change) and by their
      **rebuild cost** (what a single edit makes you wait for), and list the slowest links. A header with a high cost that
      never changes costs nothing; work through the burden list.

Per-file times come from one parallel build, so they contain contention for cores and caches: read them as a relative
cost, not as the time of the file alone.

.. _buildTimesClang:

****************
Clang trace data
****************

.. code:: bash

    $ sudo apt install clang-20
    $ python3 scripts/pq_build_times/pqbt.py snapshot --compiler clang++-20 --note "clang baseline"
    $ python3 scripts/pq_build_times/pqbt.py baseline set

The clang series builds with ``-ftime-trace``. After the cold build the tool reads the trace of every object and
``detail`` gains a section **Compiler time**: the compiler CPU split into frontend (parsing, semantic analysis, template
instantiation) and backend (optimisation, code generation), per module, and the headers and templates that cost the most.

    - The **own time** of a header is the time clang spent in it without what it includes; "with includes" contains the
      nested includes. The own time of a template is its instantiation without nested instantiations.
    - Header and template time overlap: an instantiation that happens while a header is parsed counts in both.
    - Clang leaves events shorter than 0.5 ms out of the trace, so **event counts are not exact** (about 5% of the
      headers differ between identical builds). ``compare`` therefore compares only times, after removing the drift of
      the whole build, and lists a header or template only if it moved by more than 1 s and 15% beyond that drift. These
      thresholds come from three builds of identical code, where nothing was listed.

This is the same view that the Clang Build comment on every pull request shows, see :ref:`the clang build comment
<ciMetricsClangComment>`, but on your own machine and for the state of your working tree.

****************
Habits that work
****************

    - One change per snapshot, each with a ``--note``.
    - Trust the include numbers (exact) before the seconds (noisy), and the steps rebuilt before the wall time.
    - Do not compare a Release number with a Debug number or a GCC number with a clang number: they are different series
      on purpose.
    - When a result matters, take the snapshot twice on an idle machine.
