# Changelog

All notable changes to this project will be documented in this file.

## Next Release

### Enhancements

- Add keyword "remove_net_force" for removing total net force when reading in 
  forces from QM calculations

### Build

- Faster builds: link-time optimization (`-flto`) is now opt-in via
  `BUILD_WITH_LTO` (default OFF) instead of always-on in Release builds;
  `ccache` is used automatically as a compiler launcher when available; and a
  faster linker (`mold`/`lld`) is used automatically on Linux when available
- Fixed compilation with `-DBUILD_WITH_ASE=Off`: the built-in SLAKOS path
  (`__SLAKOS_DIR__`) is now guarded, so non-ASE builds compile and report a
  clear error only if a built-in SLAKOS set is actually requested

### CI

- Faster CI: install and use the `mold` linker, and build with all available
  cores instead of two
- Faster CI: build with `-DBUILD_WITH_NATIVE=Off` (portable, no `-march=native`)
  so compilation can be cached across runs with `ccache`
- New `BUILD_WITH_NATIVE` option (default ON) to toggle `-march=native`
- Faster CI: cache the conda environment (integration-test dependencies),
  skipping reinstall on a cache hit
- Enabled compilation with Clang/Apple Clang (matched the `requires`-clause form
  on Vector3D compound-assignment operators and fixed a narrowing conversion in
  TriclinicBox)
- Faster CI: also trigger the build workflow on push to `dev`/`main` so the
  `ccache` and conda-env caches get populated on the base branch and
  subsequently-opened PRs start warm instead of cold
- Performance-regression gate: fixed-work benchmarks (`BUILD_WITH_PERF_BENCH`)
  run under callgrind; CI fails if a benchmark's instruction count regresses vs
  the base branch (deterministic, so not flaky)

<!-- insertion marker -->
## [v0.6.4](https://github.com/MolarVerse/PQ/releases/tag/v0.6.4) - 2026-03-31

### Bug Fixes

- Added unit conversion from fs to s in applyShake routine

### Build

- Added mstd-0.0.2 as git submodule to external directory for future generalizations

## [v0.6.3](https://github.com/MolarVerse/PQ/releases/tag/v0.6.3) - 2025-11-12

### Bug Fixes

- Fixed segfault when setting force-field to "bonded"
- Eigen version finally fixed to 5.0.0 (latest aka master broken on 28.09.25)

### Enhancements

- Atom positions of triclinic boxes are now wrapped into the simulation box
  when written to the trajectory output file
- Atom charges are now written to the .chrg output file in case of pure QM-MD jobs

### CI

- Daily CI workflow added to build and test the codebase
- Automatic git tag creation on new release via GitHub Actions

## [v0.6.2](https://github.com/MolarVerse/PQ/releases/tag/v0.6.2) - 2025-08-22

### Workflow

- added/updated git hooks for commit messages
- added license header check in CI workflow

### Bug Fixes

- NaN and Inf are recognized as invalid in .rst file input 
- VelocityRescalingThermostat is prevented from generating -nan velocities

### Tests

- added integration tests for QM programs

### Build

- Suppress googletest warnings for double promotion
- Fix warnings when building the Sphinx documentation

## [v0.6.1](https://github.com/MolarVerse/PQ/releases/tag/v0.6.1) - 2025-07-25

### Enhancements

- new random_seed keyword for reproducibility
- QM loop time limit info gets printed to the .log file
- QM loop time limit default value is set to 3600 (1 hour)
- Cleaned up example runs and added three new examples

### Bug Fixes

- Index 0 is now correctly out of bounds in topology file
- The path provided for qm_script_full_path preserves its letter casing

### Internal

- added function to check boolean strings in input file

### CI

- CI workflow for macOS architecture removed

## [v0.6.0](https://github.com/MolarVerse/PQ/releases/tag/v0.6.0) - 2025-04-02

### Enhancements

- new MACE models added
- ASE based xTB calculator added
- new keyword added to set custom MACE model *via* url
- option to overwrite existing output files added

### Bug Fixes

- Temperature setup now gets correctly printed to the .log output file

### CI

- Combined all CI workflows into a single workflow file

### Testing

- Added `src/QM` to ignore for code coverage reports

## [v0.5.3](https://github.com/MolarVerse/PQ/releases/tag/v0.5.3) - 2025-02-03

### Enhancements

- ASE interface for DFTB+ calculations added
- Add a new keyword 'freset_forces' to reset forces to zero after each step
- init_velocities keyword is ignored if non-zero velocities are present
- init_velocities can now be forced via the 'force' option

### Bug Fixes

- Volume now gets correctly printed to the .log output file

### CI

- Updated CMakeLists.txt to support macOS arm64 architecture.
- Added CI workflow for macOS arm64 architecture.

## [v0.5.2](https://github.com/MolarVerse/PQ/releases/tag/v0.5.2) - 2025-01-05

### Enhancements

- The reference output file is now decoupled from the .log output file and is given
  its own input file keyword 'reference_file'
- Citations added in the .ref output file for the available QM programs,
  the v-Verlet integrator, the RATTLE algorithm and PQ itself
- BibTeX entries are now included in the .ref output file

### CI

- CI workflows removed `on push` events
- building and testing workflows are deployed now only if relevant files change
- Added checks to PRs if latest base commit is included in changes of PR

### Bug Fixes

- CI for Release build updated to install all integration test dependencies
- Full anistrop coupling works now with stochastic cell rescaling manostat

## [v0.5.1](https://github.com/MolarVerse/PQ/releases/tag/v0.5.1) - 2025-01-05

### Enhancements

- Nose-Hoover chain restarting now including old chain parameters
- 'dftb_file' keyword added to change default input file dtfb.template
  for dftbplus QMMD
- Input keys in input file can now be given case-insensitive as well as with '-' or '_'
- Checks for `CHANGELOG.md` modifications on pull requests and pulls

### Bug Fixes

- Fixed QM atoms update for QM-MD calculations

### Testing

- Integration test added for DFTB+ calculation

## [v0.4.5](https://github.com/MolarVerse/PQ/releases/tag/v0.4.5) - 2024-07-13

### Bug Fixes

- Minimal Image Convention for triclinic cells now implemented with analytic extension

## [v0.4.4](https://github.com/MolarVerse/PQ/releases/tag/v0.4.4) - 2024-07-09

### Bug Fixes

- Anisotropic NPT calculations now working correctly

### Known Bugs

- Minimal Image Convention for triclinic cells only approximate

## [v0.4.3](https://github.com/MolarVerse/PQ/releases/tag/v0.4.3) - 2024-07-08

### Bug Fixes

- MACE NPT calculations bug fix - virial evaluation is now correct

### Known Bugs

- Anisotropic NPT calculations not working properly!
- Minimal Image Convention for triclinic cells only approximate

## [v0.4.2](https://github.com/MolarVerse/PQ/releases/tag/v0.4.2) - 2024-07-04

### Bug Fixes

- Isotropic manostats producing SEGFAULTS is now fixed
- Version number in output files is now always the latest tag

### Testing

-Integration Test added for an exemplary NPT calculation using Berendsen-Thermostat and -Manostat (isotropic)

### Known Bugs

- MACE NPT calculations not working!
- Anisotropic NPT calculations not working properly!
- Minimal Image Convention for triclinic cells only approximate

## [v0.4.1](https://github.com/MolarVerse/PQ/releases/tag/v0.4.1) - 2024-07-02

### Enhancements

- Logfile output updated to give all important information about the simulation settings

### CI

- added CI workflow for Kokkos enabled compilations

### Known Bugs

- Isotropic manostats producing SEGFAULTS
- MACE NPT calculations not working!
- Anisotropic NPT calculations not working properly!
- Minimal Image Convention for triclinic cells only approximate

## [v0.4.0](https://github.com/MolarVerse/PQ/releases/tag/v0.4.0) - 2024-07-01

### Features

- M-Shake
- MACE Neural Network Potential for QM-MD calculations
- Steepest-Descent Optimizer and ADAM optimizer

### Known Bugs

- MACE NPT calculations not working!
- Anisotropic NPT calculations not working properly!
- Minimal Image Convention for triclinic cells only approximate
