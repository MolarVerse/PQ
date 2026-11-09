# Developer Changelog

Build, CI, test, refactor, and implementation changes are documented here.
User-visible release notes are documented in [CHANGELOG.md](CHANGELOG.md).
This split starts with the next release; older release entries remain unchanged
in `CHANGELOG.md`.

## Next Release

<!-- insertion marker -->
## [v1.0.2](https://github.com/MolarVerse/PQ/releases/tag/v1.0.2) - 2026-11-08

### CI

- Install GCC and the other CI packages through the cached apt action (awalsh128/cache-apt-pkgs-action) in the shared setup-gcc action and the clang build, as the DevOps check already does, so a stalled or slow Azure mirror only matters on a cache miss.
- The Changelog Fragment Check honours the existing skip-changelog label: a pull request with the label needs no fragment (the script tests still run), and adding or removing the label re-runs the check. The failure message names the label.
- Use ccache in the clang build job on pull requests (with -Xclang -fno-pch-timestamp so the precompiled headers can be cached); pushes to dev still compile everything and refresh the cache.
- Refresh the apt index before installing the docs packages, so a stale runner image no longer fails the Docs job.
- Hotfix PRs now take their version from a `Hotfix-X.Y.Z` title prefix (must be the next patch); `.github/release.json` is removed, so hotfix PRs no longer show a bot edit of it.
- Skip the "Check PR Has Linked Issue" job for release pull requests (release/* branches): the release workflow creates them with the body "Release vX.Y.Z." and they close no issue, so the check failed on every release.
- Skip the lint job for release pull requests and pushes to main: it linted every file of the release (812 files, serially) and was killed after half an hour.
- Remove the CI metrics (daily collector workflow, build-analysis artifacts and steps, report, tests, page and the 21 MB of old data under .github/ci-metrics): they did not tell us anything we use, and the collector pushed data commits to dev. The history stays in git.
- Skip the lint, DevOps C++ check, performance gate and clang build jobs for release pull requests (release/* branches): they contain every change since the last release, which these checks already passed on the dev pull requests, and the checks failed or timed out on the 1.0.0 release pull request.

### Tests

- Add a FeNNol integration test (malondialdehyde, ANI-1x model, 2 MD steps, CPU) and install `fennol` and `jax` for the integration tests in CI.
- Add MACE-OFF and MACE-MP integration tests (malondialdehyde, 2 MD steps); they run only where torch, MACE and a CUDA device are available and is skipped otherwise.
- Add a timing comparison to the performance gate: CPU time of the measured loop, base against PR on the same runner (copies of the binaries, interleaved pinned runs, bootstrap interval, confirmation). A confirmed slowdown above 10% fails the check. New `perf_timer` preload library and `scripts/perf_timing.py` with tests.
- Timing comparison: the loop length is the median of three runs and the minimum is 2 ms (was one run and 5 ms), so benchmarks near the limit are judged in every run instead of in some.
- Add unit tests for the binary operators of `linalg::Vector3D` holding `mstd` quantities.

### Internal

- Include the pybind11 header in the ASE xTB, MACE and FeNNol runner sources instead of relying on the precompiled header, so lint no longer reports errors when a pull request changes them.
- Replace `Box::toOrthoSpace` and `Box::toSimSpace` with `getBoxMatrix()`, `getInverseBoxMatrix()` and `isOrthorhombic()`; `Molecule::scale` and `Molecule::scaleVelocity` now expect a scaling tensor in simulation space, converted once by `Manostat::_toSimSpaceScaling`.
- Add `StaticMatrix3x3` operators for unit/quantity scalars, mixed element-type matrix multiplication and a converting constructor.
- Introduce tracking of by whom a `Setting` was actually set (user vs. system vs. default), replacing the former value-based "is this still the default" checks.
- Start removing static access from `QMSettings` - `QMMethod`
- introduce new class `MaceSettings` for less coupling to `QMSettings`
- introduce new class `FennolSettings` for less coupling to `QMSettings`
- ASE-DFTB+ input validation now rejects third order DFTB with non-3ob slakos if `hubbard_derivs` is missing, and `DftbSettings`/`XtbSettings` were extracted from `QMSettings`.
- remove static access of `ThermostatType` from `ThermostatSettings`
- make timestep unit aware
- make compressiblity unit aware
- add a devops check to not include `units.hpp` in a header file to not pollute namings 
- add relaxtime of thermostat as a unitful quantity
- introduce units for pressure
- start introducing temperature unit in thermostat
- finish all places of temperature units
- cleanup manostat by introducing some mstd math functions for quantities like `mstd::exp`, `mstd::pow` and `mstd::sqrt`
- Refactor the binary `+`, `-`, `*` and `/` operators of `linalg::Vector3D`: the five overloads per operator are replaced by one constrained template each, built on new component-wise helpers in `linalg::detail` (`vector3dDetail.hpp`).
- `linalg::Vector3D` of `mstd` quantities now supports the binary `+`, `-`, `*` and `/` operators.

## [v1.0.0](https://github.com/MolarVerse/PQ/releases/tag/v1.0.0) - 2026-11-03

### Enhancements

- Add clang -ftime-trace data to the local build-time tooling (pqbt.py snapshot --compiler clang++-20): frontend/backend split, own parse time of headers and templates, calibrated compare rules, and a warning when other processes used CPU.
- Add "pqbt.py compare" to the local build-time tooling: compares two snapshots of one fingerprint, deterministic figures first, with timing verdicts judged against the measured run-to-run noise.
- Add "pqbt.py detail" to the local build-time tooling: per snapshot where the CPU goes, the slowest files, the rebuild cost and burden of every header, what small changes rebuild, plus a detail section in compare.
- Add scripts/pq_code_stats: lines and files per merged PR of dev, split by area, group and kind (header-like vs source, tests, perf), the state over time, optional CI runs, tidy CSV output and an HTML report.
- add new exception handling approach via tempalted NTTP for color and type, which makes handling of new exception types much easier
- Add scripts/pq_build_times: local-only build-time tracking with snapshots (cold, no-op and touch scenarios plus the include graph), a hardware and toolchain fingerprint, a pinned baseline and an over-time HTML graph.
- Add qm_force_distribution to handle distribution of the deficit QM forces of smoothing molecules in an upcoming smoothing method in hybrid type calculations (options: equal, random, distance_weighted)
- add strong type for `AtomNumber`
- add strong type for `BondId`
- add strong type for `AngleId`
- add strong type for `DihedralId`
- add strong types for `ExtVdwType` and `VdwType`
- add strong type for morse params aka `MorseParams`
- add strong type for lennard jones params aka `LJParams`
- add strong type for buckingham params aka `BuckinghamParams`
- make guff pair used `std::array` instead of `std::vector`
- introduce `AtomIndex` instead of `size_t` for indexing into molecule 
- implement `AtomType` and `ExtAtomType` instead of `size_t`
- implement `MolType` instead of `size_t`
- add guarded RAII approach for starting and automatically stopping a timings section with its destructor
- use a dedicated enum class for timers to make it less error prone
- removed now all timer dependencies from header files -- global timer is stateless and more robust
- Add VS Code tasks "devops cpp_checks (changed)" and "devops cpp_checks (all)" that run the CI C++ style checks locally and show the findings in the Problems panel.

### Bug Fixes

- Explain the allowed characters when a changelog fragment filename is invalid.
- fix clangd-tidy false positives on `vector3dConcepts.hpp` by adding a same-named translation unit (`src/linearAlgebra/vector3dConcepts.cpp`) so clangd can infer its real, absolute-path compile command instead of an unrelated fallback one
- skip the shared PCH for that new file so devops_cpp_check's libclang-based parser isn't tripped up by Eigen's SSE intrinsics headers in a configure-only (unbuilt) tree
- Make the commit-msg hook work with bash 3.2, the default `/bin/bash` on macOS.
- Make the commit-msg hook read the message file git passes to it, so it also works in git worktrees.
- Keep the changes/user and changes/developer folders across releases: a release consumed every fragment and git does not track empty folders, so both disappeared from main. A .gitkeep file in each now keeps them.
- Fix compilation with libc++ by including <charconv> in inputConverter.tpp.
- Repair the 450 CI timing records of the first backfill that were written with a null pr_number (run with collect.py --fix-pr-numbers); only that value changes on those lines.
- CI metrics collector: find the pull request of runs whose head is a merge commit (stacked PRs) through the branch's PRs, and add --fix-pr-numbers to repair records that were written with a null pr_number.
- Correct the sign in the virial calculation of the intramolecular mTR water model routine
- Stop the DevOps C++ check and the clang-format check from running on pull requests that change no C++ files: a negated path filter made every change count as relevant.
- Keep PQ Bot draft publication successful when GitHub cannot add the invoking user as reviewer.
- Restore GitHub Pages deployment from its fresh publish runner.
- Fix the Debug build with Python bindings (`-Werror=conversion` in `selection.cpp`).
- Fix the per-configuration QM charge reset in exact smoothing, which reset a copy and had no effect.

### CI

- Build the four-way CI build matrix with Ninja instead of Make (including the coverage build), so these jobs also record .ninja_log and include graph summaries for the CI metrics.
- Only cancel superseded runs for pull_request events; push (dev/main), schedule and manual runs of BUILD/LINT and the other gates now get their own concurrency group and are never cancelled.
- Give the ccache of the two Debug build jobs and of the lint job 1.5 GB instead of 500 MB: their caches were at the limit and evicting what the next build needed, so warm rebuilds hit only 56% to 78%.
- Let ccache cache compiles that use the precompiled headers in the BUILD workflow (`CCACHE_SLOPPINESS=pch_defines,time_macros`); about 65% of the compiler calls were uncacheable before.
- Add a non-gating Clang Build workflow that builds and tests the project with clang-20 on pull requests and dev pushes, as the basis for clang build-time analysis.
- Post an informational clang build time comment on pull requests, comparing the clang trace summary with the latest dev run (headers that got heavier, speed-adjusted), through a separate job that runs no pull request code.
- Check clang-format compliance on the whole changed file instead of only the changed lines, so context-dependent formatting (e.g. operator or argument alignment) can no longer produce false positives or slip through (issue #590).
- Summarise the clang `-ftime-trace` output of the clang job (totals, slowest files, heaviest headers and templates by self time) into the build-timings artifact, together with the raw traces of the slowest files.
- The commit-msg hook accepts the merge messages Git proposes itself (Merge branch ..., Merge remote-tracking branch ..., Merge pull request ...) without the merge: prefix; added tests for the hook.
- Cache the clang-20/libclang-rt-20-dev/g++-14 apt install in the DevOps C++ Check workflow via `awalsh128/cache-apt-pkgs-action`, instead of a plain `apt-get install`, to avoid re-downloading and unpacking the full clang/llvm toolchain on every run.
- Cache and shallow-clone Eigen in the Docs validate job, like other CI workflows already do, instead of letting CMake's FetchContent hit gitlab.com fresh on every run; fixes frequent clone failures in that job.
- Build the documentation without Doxygen dot graphs on pull requests (new CMake option PQ_DOXYGEN_GRAPHS) and without LaTeX output, which nothing uses. The PR docs check took about 5 minutes, almost all of it Doxygen.
- Run the Docs workflow on pull requests to `dev` and configure it without ASE and tests.
- Record the exact include graph of Ninja builds (`ninja -t deps`: include pairs, fan-in of project headers, a digest) in the build summary and show the headers whose fan-in changed in the clang build time comment.
- Restrict CI clangd-tidy to files changed in the PR instead of every file under src/apps/include, and build with tests and both benchmark suites enabled so a changed test or benchmark file is compiled and linted too (issue #620).
- LINT workflow no longer runs a full build to generate `compile_commands.json`; it configures with `CMAKE_DISABLE_PRECOMPILE_HEADERS=ON` instead, so `clangd-tidy` can re-parse every file from real headers without anything being compiled first.
- Drops the ccache/mold setup and the best-effort build step (including its `-Wmaybe-uninitialized`/`mShake.cpp` workaround) from the LINT job, since it no longer builds anything.
- Add the first CI timing data (.github/ci-metrics/data/): a 90-day backfill of 11,822 job records (2026-W27 to W40) collected with the CI metrics collector.
- Upload a per-job build summary (`.ninja_log` timings with a partial-build flag, ccache counters) as a short-lived artifact from the build and lint jobs via a new `build-timings` composite action.
- CI metrics collector: show GitHub's rate-limit message once when it stops, and add --rate-limit-wait to retry with backoff instead of stopping; document what the first backfill showed.
- Add the CI metrics collector (.github/ci-metrics/collect.py) that turns finished GitHub Actions jobs into weekly JSONL timing records, idempotently and with rate-limit handling, plus offline tests.
- Make the CI metrics collector ingest the build-timings artifacts into `build-analysis` records (validated field by field, joined to the job records by `job_id`) and add their schema.
- Ingest the clang trace summary (dev pushes) and the exact include graph into the build-analysis records and show them in the CI timings overview; the non-gating clang job stays out of the critical-path numbers.
- Add a build analysis section to the CI timings overview: ccache hit rate and precompiled-header blocked calls per job, ninja wall/CPU/link-tail figures for complete builds, and the slowest build steps on dev.
- Add a CI timings overview page (`.github/ci-metrics/data/CI_TIMINGS.md`) generated from the collected data by `report.py` and refreshed by the CI Metrics workflow; add a test for it.
- Add .github/ci-metrics/ with the CI timing record schema, storage layout and decisions (data committed to dev via GITHUB_TOKEN, all CI workflows in scope); no collector yet.
- Add the CI Metrics workflow (daily schedule + manual dispatch) that runs the collector and commits new timing data to dev via publish.sh; it stays inert until it reaches main.
- Add bounded PQ Bot reviews and test-driven draft coworker PRs with isolated model edits and scoped write credentials.
- Fix the PQ coworker bot agent so it runs headless (selectable mode, no prompts, review tier restored).
- Add the PQ coworker bot: tagged tasks, reviews, and upkeep via OpenCode on collaborator mentions.
- Build PQ with BUILD_WITH_PYTHON_BINDINGS=ON in CI (new python-bindings job) and check the reported capability; define WITH_WITH_PYTHON_BINDINGS before the subdirectories so the precompiled header is no longer rejected.
- Require successful BUILD and LINT runs for the release commit before publication.
- Keep hotfix versions fixed and allow preparation and publication to be retried after failures.
- Generate standard release notes and request release contributors on generated release and sync pull requests.
- Remove the clang build time comment on pull requests and the -ftime-trace analysis behind it (compile times on shared runners were too noisy): the report and baseline steps, the comment job, summarise_traces.py, clang_report.py, the clang part of the CI metrics and the automatic -ftime-trace of clang builds. The clang build and test job stays.
- Record the runner CPU model and core count (x86 only) in the build analysis, add a per-CPU table to the CI timings overview and name the CPUs of the run and the baseline in the clang build time comment.
- Add a `setup-gcc` composite action that installs the pinned GCC/G++ version and exports `CC`/`CXX`, replacing the per-job `apt install gcc-14 g++-14` step and per-step `CC`/`CXX` env blocks repeated across `ci_build.yml`, `devops_cpp_check.yml`, `lint.yml`, `perf.yml`, `jekyll-gh-pages.yml`, and `clang_build.yml`.
- Add concurrency cancellation to BUILD/LINT/Performance Gate/DevOps/clang-format workflows, ccache to build-static-lto, a shared Eigen source cache, and pip caching to the DevOps check.
- Build the static LTO application without the test executables on pull requests and pushes (their 165 LTO links took 78% of the job's CPU time) and build and run the tests in the nightly run, which builds main.
- Build the perf and Google Benchmark sources in the arm Debug job, so they are held to the -Werror strict warning set (before, only lint built them, with errors tolerated). The benchmark smoke tests stay in the benchmark job.
- Make the dev-to-NEXT sync workflow run on every push to dev (the cron only fires from main) and mint its token from the installed pq-release-bot app, since the pq-bot app is not installed on the repository.
- Merge every push to main back into dev automatically, opening a PR on conflicts or a rejected push; the dev push then syncs NEXT, so hotfix and exceptional main commits reach both branches.
- The release workflow no longer opens its own main-to-dev PR, and sync PRs from `merge/` branches skip the changelog fragment check.
- Add tests for HybridSettings and HybridInputParser

### Tests

- Add unit tests for the `concepts` module (`base.hpp`, `innerType.hpp`), which previously had no `tests/src/concepts` target; this also gives clangd-tidy a real compile command for these headers so it stops inferring flags from an unrelated external source file.
- Add unit tests for previously-untested degenerate-geometry edge cases in BondConstraint::applyShake, DistanceConstraint::applyDistanceConstraint, and CellList::determineCellSize.
- Add the first dedicated DistanceConstraint unit test file.
- Add the first tests/src/engine/ unit test suite, covering Engine::calculateTotalSimulationTime and the isXxxActivated getters via the concrete MMMDEngine.
- Add unit tests for QMRunnerManager (runner per method, missing qm_prog, ASE disabled) and for OptEngine::throwOnFailure, which HessianEngine now shares.
- Add an integration test that runs the oh_mm_hessian example (MM geometry optimization plus Hessian, no QM program needed) with and without max-learning-rate and checks the error for a minimum above the maximum.
- Add a `testAllKeywordsAcceptValidValue` test that processes every registered keyword with a representative value from `keywordList.txt`, catching renamed/typo'd keywords (like #864) by exercising each key's parse function instead of only comparing keyword-map sizes.
- Fix ctest failures under parallel execution (`ctest -j`): tests shared one working directory and raced on the same relative output files (e.g. the default log/restart/trajectory files); each test now runs in its own directory.
- Perf gate: count the heap allocations of the measured loop of every benchmark (exact, independent of the CPU) with a preloaded library and fail when they rise; the PR comment shows base and PR counts.
- Perf gate: report simulated cache misses (L1 data, last level) and branch mispredictions of every benchmark next to the instruction counts (callgrind cache and branch simulation, report only), measured through a path of fixed length and an empty environment so that base and PR are comparable.
- Perf gate: add an end-to-end benchmark that runs a short MM-MD simulation (219 rigid TIP3P waters, M-Shake, cell list, 100 steps) through the real driver, including setup and output.
- Perf gate: a benchmark that crashes or reports no instruction count now fails the check instead of counting as an improvement, and benchmarks removed in a PR are listed. The SHAKE/RATTLE benchmark (now perf_shakeRattle) corrects a violated bond in every iteration instead of only checking convergence.
- Perf gate: add fixed-work benchmarks for the GUFF pair kernel (dense and sparse coefficients), the Coulomb reaction field, the Maxwell-Boltzmann velocity initialisation and ResetKinetics.
- Add a QM/MM zone bookkeeping benchmark (inner region center, zone assignment, smoothing factors) to the performance gate.
- Add input reader benchmarks (moldescriptor, restart file, topology bonds and angles) in three sizes to the performance gate, with a scaling exponent.
- Perf gate: an improvement of 30% or more (PERF_LARGE_DROP_PERCENT) is flagged with a warning to check that the benchmark still does its work; it never fails the check.
- Perf gate: add a fixed-work benchmark for the M-SHAKE / M-RATTLE solver on 100 rigid water molecules.
- Skip the simulated cache and branch counts in the pull request performance gate (`PERF_CACHE_SIM=0`) and report them in a nightly run of `dev`.
- Add benchmarks of the steepest descent and Adam update steps, the optimizer history and convergence check and the learning rate strategies to the performance gate.
- Add output writer benchmarks (xyz, velocities, forces, charges, restart file, energy) in three sizes to the performance gate, with a scaling exponent.
- Report the peak heap in use of every performance benchmark (exact, from the allocation counter; bytes per molecule for the size series) in the performance gate comment. Report only.
- Add benchmarks of the DFTB+, Turbomole and PySCF runner file handling (coordinates, point charges, forces, charges, stress tensor; no QM program is started) to the performance gate.
- Perf gate: add scaling benchmarks (cell list update and inter-water forces for 216, 512 and 1728 waters) and show the scaling exponent of such size series in the performance comment.
- Perf gate: add fixed-work benchmarks for the Berendsen, Langevin, Nose-Hoover and velocity rescaling thermostats and the Berendsen and stochastic rescaling manostats (the Monte Carlo manostat is an empty stub).
- Add tests for ExternalQMRunner periodicity rejection and the DFTB+/PySCF/Turbomole runners' missing-script-file rejection; both guards already existed, only the tests were missing.
- Add unit tests for the file formats the DFTB+, Turbomole and PySCF runners write (coordinates, cell, point charges), and give QMRunner::_periodicity a default value instead of leaving it uninitialised.
- Add unit tests for ReferenceSettings, WaterModelSettings::finalize, and ConstraintSettings::finalize, and extend ManostatSettings, QM/thermostat/integrator input parser tests to assert on the actual reference files added rather than only on compilation.
- Seed `testRandomNumberGenerator`, so its statistical checks no longer fail at random in CI.
- Add a unit test verifying RingPolymerEngine::writeOutput only writes the ring-polymer restart file on the MPI root rank.
- Add a settings builder for tests (tests/include/testUtils/testSettings.hpp) that sets individual settings without the input parser or the static settings, with tests; the resetKinetics and constraints tests use it; documented in the developer guide.
- Add a regression test that runs the same water box with SHAKE and M-SHAKE and compares the final positions and velocities. The regression runner can now run several inputs per case and compare the last frames of two output files.
- Add thermostat regression tests for extreme parameters (relaxation time below the time step, zero target temperature, zero degrees of freedom, near-zero temperature, tiny Langevin time step).
- Add the first tests/src/timings/ unit test suite (TimingsSection/Timer) and add calculateQMVirial coverage to the existing virial test suite.
- Add regression tests for degenerate geometries of the intramolecular water models, and a WaterModelException type with a shared guard.

### Internal

- preparation of main app for architectural rework
- move `commandLineArgs.{hpp,cpp}` completely into `apps/` as nothing else depends on it
- fix a lot of more small clang-tidy warnings (larger one will be handled in separate PRs)
- address all cppcoreguideline clang-tidy checks and all darwin checks
- add most important fuchsia clang-tidy warnings (especially no default arguments)
- add most important google clang-tidy warnings
- add most important hicpp clang-tidy warnings
- add all important llvm clang-tidy warnings
- add most important misc* clang-tidy warnings
- add all important modernize clang-tidy warnings
- fix all bugprone-easily-swappable-parameters in `apps/` directory
- fix all straight forward fixable issues with clang-tidy warnings
- remove all `using` keywords from global namespace (even more hardcore then the google clang-tidy check)
- fix `cppcoreguidelines-non-private-member-variables-in-classes`
- remove a lot of transitive includes of `box.hpp`
- keep the per-pair `Potential::calculateSingle*` and `InterWaterStrategy::calculateSingle*` kernels inlinable in the pair loops (private kernel headers, `Box` passed directly) so dropping `box.hpp` from public headers does not add a call per atom pair
- introduce new devops cpp check rule for keeping a consistent naming convention for `SimulationBox` and `PhysicalData` parameters in source files
- run the devops C++ check in CI only on files changed in the pull request via `--base-ref`
- move `ShakeType` from settings to enum
- move `ConvStrategy` from settings to enum
- move `HessianBuilderType` from settings to enum
- move `SmoothingMethod` and `QMForceDist` from settings to enum
- move `ManostatType`, `Isotropy` and `FixedAxis` from settings to enum
- move `OptimizerType` and `LearningRate` from settings to enum
- move `ForceFieldType`, `NonCoulombType` and `CoulombLongRangeType` from settings to enum
- use new enum aliases strategy from mstd to make life easier for QM enums later
- move all `QM` related enums from settings to enum
- move all enums from `settings.hpp` to enum
- move `InitVelocities` and `ThermostatType` from settings to enum
- move `WaterIntraModel` and `WaterInterModel` from settings to enum
- make all function parameters in `output` lib const correct
- make `Optimizer` cleaner by using PIMPL
- disallow `throw(...)` -- instead use `throw ...`
- disallow the use of `EXPECT_THROW` and `ASSERT_THROW` -- they should always be used via their alternatives `EXPECT_THROW_MSG` and `ASSERT_THROW_MSG`
- disallow private members to NOT start with a leading underscore
- disallow private member functions to NOT start with a leading underscore
- remove `cell.hpp` 50x transitive includes by removing `molsys::CellList` public include in `engine.hpp`
- fix ordering of members and methods in classes and structs + disallow public members starting with an underscore
- do not use raw pointers in tests with new and delete (enforced by devops)
- remove useless comments next to includes for less PR review garbage
- diasllow the use of `clang-format` `on/off` in source code
- disallow the use of `final` key for dependency injection (mocking in tests)
- disallow the use of the `class` specifier for template parameters -- use `typename` instead
- enforce using `mstd` lib as system header in PQ 
- enforce consistent header guard naming
- prepare cell list input file parser to remove engine dependency
- prepare constraints input file parser to remove engine dependency
- prepare files input file parser to remove engine dependency
- prepare mm input file parser to remove engine dependency
- remove engine member of all input file parser and the input file reader
- start refactoring input file parser approach with a clean generalized interface using `InputKey<T>` and `InputRegistry` (no changes of existing code yet)
- implement `input key adapter` for a smooth transitioning to the new approach
- add `KeyValidator` strategy and implement new strategy for input file parser
- migrate `RingPolymerInputParser` and `CellListInputParser`
- migrate `TimingsInputParser`
- migrate `ConstraintsInputParser`
- migrate `ResetKineticsInputParser`
- migrate `FilesInputParser`
- rework error messages when `tryParse` fails
- migrate `OutputInputParser`
- migrate `VirialInputParser`
- migrate `IntegratorInputParser`
- migrate `CoulombLongRangeInputParser`
- migrate `MMInputParser`
- migrate `SimulationBoxInputParser`
- migrate `ManostatInputParser`
- migrate `HessianInputParser`
- migrate `OptInputParser`
- migrate `ThermostatInputParser`
- migrate `ConvergenceInputParser`
- migrate `QMInputParser`
- migrate `HybridInputParser`
- migrate `FPType` key of `GeneralInputParser`
- construct engine type outside of parser
- remove `Engine` completely from `InputFileReader`
- remove some member variables of `Molecule` that are very cheap to recompute
- introduce exceptions if some variables were forgot to set in the code to not get UB
- remove unused simulationbox member functions
- use mp-units strong types for box angles instead of raw doubles
- rework `ResetKinetics` internal `temperature`, `momentum` and `angularMomentum` state handling
- introduce new base type `Setting<T>` for settings
- rename `Settings` class to `GeneralSettings` 
- introduce `ResetKineticsSettings` as non-static class and `Settings` POD
- inject `SimulationBoxSettings` into `SimulationBox`/`SimulationBoxSetup` instead of reading static settings, pass `Settings` by const reference into the engine factory and `MDEngine`
- Replace static `HessianSettings` with an instance-based `Setting<T>` member injected via `Settings`, removing global/static state.
- Replace static `RingPolymerSettings`
- Replace static `OptimizerSettings`
- Replace static `ConvSettings`
- Replace static `WaterModelSettings`
- Replace static `ConstraintSettings`
- Replace static `ManostatSettings`
- Replace static `ForceFieldSettings`
- Add the possibility to add names for `Setting` as an `mstd::Tag`
- Replace static `OutputFileSettings`
- Introduce non-static `ReferenceSettings`
- file tracking issues (#906-#912) for the free-form `// TODO` comments flagged by clangd-tidy's `google-readability-todo` check, and update each comment to the `TODO(97gamjak): <issue-url>` convention already used elsewhere in the codebase
- delete `typeAliases.hpp` file completely
- shorten `simulationBox` namespace to `molsys`
- short `potential` namespace to `pot`
- shorten `output` namespace to `out`
- shorten `forceField` namespace to `ff`
- shorten `linearAlgebra` namespace to `linalg`
- remove `constants` namespace
- make virial a pure static class, as the entire member handling was already partially deprecated with the qmmm 
- add virial timers back as they are now not dependent anymore on an inhertance approach
- remove color settings when using VS Code IDE
- add explicit path filtering via filters in CI to make them checkable in rulesets
- add CI to check if a pull request has a linked issue
- add mentioned issues in PR title or description as a linked issue
- add 0.5% threshold for test coverage
- fix CI issue if special characters like backticks are present in PR title
- make all compiler warnings as errors
- add a new `NEXT` branch for future feature development that should not be in released in the next version (feature branch). Additionally, add a CI to automatically merge dev into NEXT daily to keep it in sync
- install `tqdm` for lint CI for clangd-tidy
- Store large integration-test reference data sets as zip archives instead of raw files to shrink the repository; `test_with_data_dir` lazily extracts them into a gitignored cache.

### Documentation

- Correct AGENTS.md (CI workflow table, clang-format scope, duplicate entry) and state the enforced 360-character changelog bullet limit in the docs.
- Document in AGENTS.md why the integration tests run on both x86 and arm Release builds (decision for #706: they cost no wall-clock time, and no failure in about 380 runs was architecture specific).
- Record the outcome of the clang build-time noise study (#751) in the CI metrics README: the thresholds are final, timings stay informational, no include-graph gate and no nightly Release clang build.
- Extend the developer guide with pages on the CI workflows, the static checks, the CI metrics, the local build-time and code-statistics tools, the CMake options and the branch and release process; correct the clang-format version note.
- enforce clean doxygen documentation for source code
- add source code documentaion to sphinx github pages documentation
- enable macro expansion for mstd enum macros: `MSTD_ENUM` and `MSTD_ENUM_BITFLAG`
- `make docs` fails now if no doxygen executable was found
- print `make docs` error to output stream not only warning file
- Add GitHub instructions for releases and hotfixes, with CLI commands on a fallback page.
- Document the nBeads parameter of the ring polymer trajectory header so the Doxygen build passes.

## [v0.7.0](https://github.com/MolarVerse/PQ/releases/tag/v0.7.0) - 2026-08-07

### Enhancements

- Add FeNNol references and its QM runner.
- Add `mace_mode` for selecting accurate or accelerated MACE execution.
- Add the MM Hessian workflow.
- Add `mshake-iter` and `mshake-tolerance`.
- Add net-force removal for imported QM forces.
- Implement Reaction Field long-range correction.

### Bug Fixes

- Count non-adjacent atom types correctly.
- Guard the Berendsen thermostat against zero-temperature invalid values.
- Reject aliased cell-list layouts.
- Guard angle-force division for collinear configurations.
- Reject non-finite external QM energies and forces.
- Refresh the Langevin noise amplitude after friction changes.
- Correct M-SHAKE loop bounds, convergence, iteration limits, time units, and previous-position handling.
- Correct the no-PBC squared-distance calculation.
- Define `ParameterFileReader` destruction where `ParameterFileSection` is complete so AppleClang builds succeed.
- Export the potential force kernels.
- Fix release recovery when main and dev contain deleted or conflicted files.
- Guard SLAKOS tests in builds without ASE.
- Preserve molecular geometry during manostat scaling.

### Performance

- Skip zero-coefficient terms in `GuffPair::calculate`.

### Build

- Add the `.tpp` extension to `addLicense.sh`.

### CI

- Warm build caches from `dev` and `main`.
- Build portable binaries for reusable compiler caches.
- Require regular pull requests to add audience-qualified changelog entries.
- Split user and developer changelogs and require curated user notes in release pull requests.
- Replace per-PR changelog edits with release-time generation.
- Add a workflow that dismisses stale PR approvals on real code changes but keeps them when a push only merges dev's tip into the PR branch.
- Cache the integration-test environment.
- Fix integration-test environment setup.
- Cache base-branch performance instruction counts.
- Run the performance gate only for relevant changes.
- Post performance results as a persistent pull-request comment.
- Add the callgrind instruction-count regression gate.
- Automate release PR creation, changelog preparation, publishing, and sync PRs with PQ Release Bot.
- After a release is triggered, open a pull request from `main` to `dev` for approval instead of merging directly under the branch-protection rules.
- Consolidate scheduled checks into BUILD and remove the incorrect target-branch ancestry workflow.

### Tests

- Cover the renamed ASE MACE runner.
- Expand coverage for reset kinetics, output writers, optimizers, evaluators, settings, force fields, thermostats, manostats, cell lists, and setup paths.
- Add explicit coverage for enum sentinels.
- Add FeNNol parser and runner tests.
- Add brute-force and cell-list force-equivalence tests.
- Add Hessian-builder tests.
- Add integration test for atomic virial correction in integration_tests/cgo/
- Add the math and algorithm regression suite.
- Add M-SHAKE keyword, convergence, unit, loop-bound, and state tests.
- Add fixed-work performance benchmarks for force kernels, pair potentials, linear algebra, box transforms, integration, kinetics, and constraints.
- Add Reaction Field unit and long-range-correction tests.

### Internal

- Rename the ASE-based MACE runner classes consistently.
- Prefix performance benchmark filenames.
- Avoid a temporary vector during cell-list rebuilds.
- Move changelog fragments into `changes/user/` and `changes/developer/` directories with `<category>.<title>.md` naming, and allow a fragment to hold multiple bullets.
- Document the changelog fragment workflow in the Sphinx developer guide.
- Organize migrated release notes into scope-specific fragments and validate content-preserving fragment reorganizations.
- Add first version of CI for static analysis via clangd and clang-tidy (all clangd-tidy checks for now disabled apart a test check)
- fix all `bugprone-*` apart from `bugprone-easily-swappable-parameters` warnings of code base
- remove `<chrono>` transient header from timer as it was included in every single TU at the moment
- remove transient pybind headers from `AseRunners` to decrease compilation parsing time by about 5-6% (total speedup approx. 4%)
- remove more `matrix.hpp` header inclusions -- until now `mShake.hpp` exposed the header in its public API and therefore it ended up via `constraints.hpp` in `engine.hpp` and therefore almost everywhere, included `Eigen`
- remove `matrix.hpp` dependency from `ForceFieldNonCoulomb` class as it was again included transitively in many many TUs
- remove some useless public API functions from `ForceFieldNonCoulomb` class for easier maintainance as they were only used for testing
- add pre-compiled-headers (pchs) to speedup compilation time ~20%
- zero-initialize kinetic-energy accumulator tensors exposed by PCH-enabled builds
- Return cell and neighbor collections by constant reference.
- Inline Coulomb cutoff getters.
- Remove default branches from enum switches and share sentinel fallbacks.
- Make the input parser's `bind_front` approach clangd-compliant.
- Move input validation into the engine configuration path.
- Remove the unused Kokkos integrator implementation.
- Add language-server configuration.
- remove type alias for output namespace from pq namespace
- add `DefaultFile` class for default file names and use it also in `EnergyOutput` to avoid manually repeating the default file names
- remove `matrix.hpp` header from `typeAliases.hpp` as this way the `Eigen` library get included everywhere
- remove `thermostat` namespace from `typeAliases.hpp`
- remove `opt` namespace entries of `typeAliases.hpp`
- remove `virial` namespace types for `typeAliases.hpp`
- remove all `std` aliases from `typeaAliases.hpp`
- remove `resetKinetics` from `typeAliases.hpp`

### Documentation

- Add Claude Code and AI-agent instructions.
- Update the feature list and PQ reference.
- Document Reaction Field, M-SHAKE, and FeNNol.
- Refresh the reference manual, quick start, examples, and troubleshooting.
