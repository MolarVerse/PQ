# Developer Changelog

Build, CI, test, refactor, and implementation changes are documented here.
User-visible release notes are documented in [CHANGELOG.md](CHANGELOG.md).

## Next Release

### Enhancements

- Add input-file validation.
- Implement Reaction Field long-range correction.
- Add `mace_mode` for selecting accurate or accelerated MACE execution.
- Add `mshake-iter` and `mshake-tolerance`.
- Add FeNNol references and its QM runner.
- Add the MM Hessian workflow.
- Add net-force removal for imported QM forces.

### Bug Fixes

- Refresh the Langevin noise amplitude after friction changes.
- Preserve molecular geometry during manostat scaling.
- Reject aliased cell-list layouts.
- Guard SLAKOS tests in builds without ASE.
- Count non-adjacent atom types correctly.
- Export the potential force kernels.
- Correct M-SHAKE loop bounds, convergence, iteration limits, time units, and
  previous-position handling.
- Reject non-finite external QM energies and forces.
- Correct the no-PBC squared-distance calculation.
- Guard the Berendsen thermostat against zero-temperature invalid values.
- Guard angle-force division for collinear configurations.

### Performance

- Skip zero-coefficient terms in `GuffPair::calculate`.

### Build

- Enforce exhaustive enum switches.
- Guard the built-in SLAKOS path in builds without ASE.
- Add `BUILD_WITH_NATIVE` for controlling native compiler optimizations.
- Add Clang and Apple Clang support.
- Detect available compiler caches and faster linkers.
- Make link-time optimization opt-in through `BUILD_WITH_LTO`.

### CI

- Split user and developer changelogs and require curated user notes in release
  pull requests.
- Replace per-PR changelog edits with release-time generation.
- Cache base-branch performance instruction counts.
- Run the performance gate only for relevant changes.
- Post performance results as a persistent pull-request comment.
- Add the callgrind instruction-count regression gate.
- Warm build caches from `dev` and `main`.
- Cache the integration-test environment.
- Build portable binaries for reusable compiler caches.
- Fix integration-test environment setup.

### Tests

- Add explicit coverage for enum sentinels.
- Add Reaction Field unit and long-range-correction tests.
- Add the math and algorithm regression suite.
- Cover the renamed ASE MACE runner.
- Add M-SHAKE keyword, convergence, unit, loop-bound, and state tests.
- Add FeNNol parser and runner tests.
- Add Hessian-builder tests.
- Add brute-force and cell-list force-equivalence tests.
- Expand coverage for reset kinetics, output writers, optimizers, evaluators,
  settings, force fields, thermostats, manostats, cell lists, and setup paths.
- Add fixed-work performance benchmarks for force kernels, pair potentials,
  linear algebra, box transforms, integration, kinetics, and constraints.

### Internal

- Remove default branches from enum switches and share sentinel fallbacks.
- Move input validation into the engine configuration path.
- Prefix performance benchmark filenames.
- Add language-server configuration.
- Rename the ASE-based MACE runner classes consistently.
- Return cell and neighbor collections by constant reference.
- Avoid a temporary vector during cell-list rebuilds.
- Inline Coulomb cutoff getters.

### Documentation

- Update the feature list and PQ reference.
- Document Reaction Field, M-SHAKE, and FeNNol.
- Refresh the reference manual, quick start, examples, and troubleshooting.

<!-- insertion marker -->
## [v0.6.4](https://github.com/MolarVerse/PQ/releases/tag/v0.6.4) - 2026-03-31

### Build

- Add mstd 0.0.2 as a submodule for future generalizations.

## [v0.6.3](https://github.com/MolarVerse/PQ/releases/tag/v0.6.3) - 2025-11-12

### CI

- Add the daily build and test workflow.
- Add automatic tag creation for releases.

## [v0.6.2](https://github.com/MolarVerse/PQ/releases/tag/v0.6.2) - 2025-08-22

### Workflow

- Add and update commit-message hooks.
- Add the license-header check.

### Tests

- Add integration tests for QM programs.

### Build

- Suppress GoogleTest double-promotion warnings.
- Fix warnings in Sphinx documentation builds.

## [v0.6.1](https://github.com/MolarVerse/PQ/releases/tag/v0.6.1) - 2025-07-25

### Internal

- Add a helper for validating boolean input strings.

### CI

- Remove the macOS workflow.

## [v0.6.0](https://github.com/MolarVerse/PQ/releases/tag/v0.6.0) - 2025-04-02

### CI

- Combine the CI workflows.

### Tests

- Exclude `src/QM` from coverage reports.

## [v0.5.3](https://github.com/MolarVerse/PQ/releases/tag/v0.5.3) - 2025-02-03

### Build

- Add macOS arm64 support to CMake.

### CI

- Add the macOS arm64 workflow.

## [v0.5.2](https://github.com/MolarVerse/PQ/releases/tag/v0.5.2) - 2025-01-05

### CI

- Run build and test workflows only for relevant changes.
- Check that pull requests include the latest base commit.
- Install all integration-test dependencies in release builds.

## [v0.5.1](https://github.com/MolarVerse/PQ/releases/tag/v0.5.1) - 2025-01-05

### Workflow

- Add changelog checks for pull requests.

### Tests

- Add a DFTB+ integration test.

## [v0.4.2](https://github.com/MolarVerse/PQ/releases/tag/v0.4.2) - 2024-07-04

### Tests

- Add an isotropic NPT integration test using the Berendsen thermostat and
  manostat.

## [v0.4.1](https://github.com/MolarVerse/PQ/releases/tag/v0.4.1) - 2024-07-02

### CI

- Add a Kokkos build workflow.
