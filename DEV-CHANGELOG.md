# Developer Changelog

Build, CI, test, refactor, and implementation changes are documented here.
User-visible release notes are documented in [CHANGELOG.md](CHANGELOG.md).
This split starts with the next release; older release entries remain unchanged
in `CHANGELOG.md`.

## Next Release

<!-- insertion marker -->
## [v0.7.0](https://github.com/MolarVerse/PQ/releases/tag/v0.7.0) - 2026-08-06

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
- After a release is triggered, open a pull request from `main` to `dev` for approval instead of merging directly under the branch-protection rules.
- Finalize release changelogs in the release pull request before tagging protected branches.
- Keep automated release tags consistent with the existing v-prefixed version format.
- Limit changed-file formatting checks to pull requests targeting dev.
- Pass release PR metadata safely into shell validation.

### Tests

- Cover the renamed ASE MACE runner.
- Expand coverage for reset kinetics, output writers, optimizers, evaluators, settings, force fields, thermostats, manostats, cell lists, and setup paths.
- Add explicit coverage for enum sentinels.
- Add FeNNol parser and runner tests.
- Add brute-force and cell-list force-equivalence tests.
- Add Hessian-builder tests.
- Add an integration test for the atomic virial correction.
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

- Add shared contributor guidance.
- Update the feature list and PQ reference.
- Document Reaction Field, M-SHAKE, and FeNNol.
- Refresh the reference manual, quick start, examples, and troubleshooting.
