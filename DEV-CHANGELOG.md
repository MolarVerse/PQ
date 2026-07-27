# Developer Changelog

Build, CI, test, refactor, and implementation changes are documented here.
User-visible release notes are documented in [CHANGELOG.md](CHANGELOG.md).
This split starts with the next release; older release entries remain unchanged
in `CHANGELOG.md`.

## Next Release

### Enhancements

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
