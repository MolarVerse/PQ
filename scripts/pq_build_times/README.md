# Local build-time tracking

A small toolkit for analysing and improving PQ's build times step by step, on **your** machine.
Standard library only. It is deliberately **uncoupled from the CI metrics** (`.github/ci-metrics`):
it has its own parsers and schema, and its data never leaves your machine.

```text
python3 scripts/pq_build_times/pqbt.py snapshot --note "what changed"   # measure and store
python3 scripts/pq_build_times/pqbt.py baseline set                     # pin the newest snapshot as the baseline
python3 scripts/pq_build_times/pqbt.py report                           # HTML graph + terminal table
python3 scripts/pq_build_times/pqbt.py compare [BEFORE] [AFTER]         # what did my change do?
python3 scripts/pq_build_times/pqbt.py detail [ID]                      # where does the build time go?
python3 scripts/pq_build_times/pqbt.py list                             # stored fingerprints
```

## What a snapshot measures

It configures and builds in a directory **it owns** (default `build-times/` in the repository, ignored by git; it
is deleted and recreated for the cold build, and the tool refuses to touch a directory it did not create) and runs
these scenarios:

| Scenario | What it answers |
| --- | --- |
| `cold` | configure + full build from nothing (dependencies come from a cache, so the network is not timed) |
| `noop` | cost of "nothing to do" |
| `touch_leaf` | a median source file changed: the everyday edit |
| `touch_header_top` | the most widely included project header changed: the worst case |
| `touch_header_median` | a median shared header changed |

Besides wall time, each scenario stores the **steps rebuilt** (deterministic: it does not depend on machine
speed), CPU time, parallelism, link time and the run-to-run spread. The snapshot also stores the exact include graph
(objects, files, include pairs and a digest). Cheap scenarios are repeated (`--repeat`, default 3, median stored).
ccache is off unless you pass `--ccache`, because cache hits hide compile time.

The touched files are chosen deterministically from the include graph and then **pinned** to the first snapshot
(or the baseline) of the series, so the numbers stay comparable. Override with `--leaf`, `--header-top`,
`--header-median`.

## Fingerprint, baseline, storage

Every snapshot carries a fingerprint: architecture, CPU model, logical cores, compiler, build type, ninja target,
CMake arguments, `-j`, ccache. Its id is a hash of exactly those, and **only snapshots with the same id are compared
or plotted together**. Change the compiler or a flag and you start a new series (and pin a new baseline); kernel, RAM,
tool versions and the git commit (with a dirty flag and the submodule commits) are recorded for reading only.

Data lives in `$PQ_BUILD_TIMES_DIR`, else `$XDG_DATA_HOME/pq-build-times`, else `~/.local/share/pq-build-times`
(`--data-dir` overrides all). The baseline is pinned explicitly and never overwritten by itself.

## The report

`report` prints a table (the change against the baseline in brackets) and writes one self-contained HTML file with
SVG charts: wall time per scenario over the snapshots, the rebuild scope (steps) and include-graph size, the
baseline as a dashed line and every `--note` as a marker. It needs no network and no packages.

## Comparing two snapshots

`compare` (no arguments: the baseline, or the previous snapshot if there is none, against the latest; one argument:
that snapshot against the latest; or `baseline`, `previous`, `latest`, an id or a unique id prefix) prints, in this
order:

1. the **deterministic figures**: include graph size and digest (`identical` means the include graph did not change),
2. per scenario the **steps rebuilt** and the wall and CPU time with their change, the **noise threshold** and a verdict.

The noise threshold is twice the larger run-to-run spread of the two snapshots, at least 5% (10% if a snapshot has
only one run, e.g. the cold build, marked `*`), and a change below 0.05 s is never a change. The verdicts combine both
kinds of evidence, as in the improvement loop: `improved` / `regressed` need fewer or more steps **and** a timing
change beyond the noise. A timing change with unchanged steps (or the opposite direction) is reported as `check noise`
and not as an effect of your change. Touch scenarios that touched different files are not compared.

Snapshots with different fingerprints are refused, naming the differing keys.

## Detail: where the build time goes

`detail [ID]` (default: the latest snapshot) explains one snapshot. Every snapshot stores a detail file
(`details/<id>.json` next to the snapshots) made from what Ninja recorded (`.ninja_log`: when every step ran,
`.ninja_deps`: which files every object includes) and from git. Sections:

| Section | What it tells you |
| --- | --- |
| **Build at a glance** | wall time, CPU time, how many of the job slots were busy, **what limits the wall time** (CPU throughput, or idle cores because of a serial phase or a few long steps), when each precompiled header is ready, what is still running at the end (long poles), running steps per tenth of the build |
| **Where the CPU goes** | CPU per group (project, tests, third-party, linking, ...) and per `src/<module>`: where the compile work is |
| **Slowest compile steps** | the slowest files with their share of the CPU, when they ran, how many repository files they include and in how many PRs they changed lately; how concentrated the work is (how many files make up half of it); the translation units with the biggest parse load |
| **Headers: what a change rebuilds** | per header the **rebuild cost** (compile CPU of every translation unit that includes it) and the **burden** (rebuild cost x the number of PRs that changed it in the last 180 days, `--churn-days`), which is what you actually pay for. Headers included by exactly the same files are one row. For a header of a submodule (`external/...`) the changes are bumps of the submodule |
| **What small changes rebuild** | for the touch scenarios the steps that ran, by kind. A source file change that rebuilds one object but links 170 executables is visible here ("mostly relinking") |
| **Linking** | link CPU and the slowest links |

How to read it: **burden** is the list to work through (a header with a high rebuild cost that never changes costs
nothing), **rebuild cost** tells what a single change to a header makes you wait for, and the **CPU-bound** line tells
whether fewer includes help the wall time (it does when the cores are busy) or a long step does (it does when they idle).

`compare` adds, when both snapshots have a cold-build detail, a section **What changed in the detail**: first the
exact changes (translation units that include more or fewer repository files, headers whose fan-in changed), then
the compile CPU per module. The timing part removes the **drift of the whole build** (the ratio at which half of the
compile CPU lies, so a machine that was 4% slower in the second run does not make every module look changed) and lists a
module only if it moved by more than 5 s **and** 8% beyond that drift. These thresholds come from measurement: three
cold builds of identical code on one machine gave 0 listed modules out of 105 comparisons, while without the drift
removal and with a 10% threshold seven modules were listed. Each cold build is one run, so take the timings as hints and
the include numbers as facts. Limit: a module that holds more than half of the compile CPU decides the drift itself, so
only the whole-build line shows its change (the largest module, the tests, is 43% today).

Limits: per-file times come from one parallel build, so they include contention for cores and caches and are a
relative cost, not the time of the file alone. Ninja stores times per build, so the wall time, the parallelism and the
"ran" column exist only for a snapshot that includes the `cold` scenario; without it the per-file times come from
the last build of each file. Parse versus code generation time per header and per template needs clang's
`-ftime-trace` and is not part of this (yet). `snapshot` refuses to start while a submodule that the build uses
(for example `external/mstd`) is not at the commit the repository records, because that build would fail or measure
something else; `--force` skips the check.

## Habits that keep the numbers honest

- Measure on an idle machine; the tool refuses above a load of 25% of the threads unless you pass `--force`.
- Use `--target` for a part of the project while developing the tool or testing an idea, but know that it is a
  different series (the target is part of the fingerprint).
- To try clang: `--cmake-arg -DCMAKE_CXX_COMPILER=clang++ --cmake-arg -DCMAKE_C_COMPILER=clang`. It becomes its
  own series; `-ftime-trace` data is a planned next step.
- One change per snapshot, with a `--note`; that is what makes the markers in the graph useful.
