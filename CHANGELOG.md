# Changelog

User-visible changes to PQ are documented here. Build, CI, test, and
implementation changes are documented in
[DEV-CHANGELOG.md](DEV-CHANGELOG.md).

## Next Release

<!-- insertion marker -->
## [v0.7.0](https://github.com/MolarVerse/PQ/releases/tag/v0.7.0) - 2026-08-07

### New Features

- Add FeNNol as an ASE-based QM runner.
- Add `mace_mode` to choose between accurate and accelerated MACE execution; accelerated mode requires matching cuequivariance packages.
- Add the `mm-hessian` job for molecular-mechanics Hessian calculations, with optional geometry optimization.
- Add `mshake-iter` and `mshake-tolerance` for controlling M-SHAKE convergence.
- Add `remove_net_force` for removing the total net force from imported QM forces.
- Add Reaction Field as a long-range electrostatics method.
- The `PQ` executable now exposes version, help, and compiled capabilities for setup tools.
- PQ setup tools now expose bundled external-QM scripts and validate input ranges, setting constraints, resource paths, and disabled couplings.
- Trajectory comments can include the effective simulation step when `include_output_metadata` is enabled.

### Changes

- Clean include directives with IWYU.
- Rename `mace_model_size` to `mace_model`; the old keyword remains available with a deprecation warning.

### Bug Fixes

- Count non-adjacent duplicate atom types correctly.
- Prevent the Berendsen thermostat from producing invalid velocities when the kinetic energy is zero.
- Reject periodic cell-list layouts in which neighbor offsets refer to the same cell more than once.
- Prevent undefined forces for collinear angle configurations, including linear equilibrium geometries such as CO2.
- Reject non-finite energies and forces from external QM calculations instead of propagating them into a trajectory.
- J-coupling topology references now remain valid throughout a simulation, preventing incorrect molecule data.
- Fix the virial mode used after copying physical data with atomic virial enabled.
- Fix molecular virial broken with exception after previous bugfix
- Recompute Langevin noise when the friction setting changes.
- Fix M-SHAKE convergence, iteration limits, previous-position handling, and velocity corrections for constrained molecules.
- Reference output is validated before writing, preventing partial files when a referenced file cannot be read.
- Single-step simulations now complete their progress indicator without dividing by zero.
- Preserve molecular geometry and wrap positions correctly during stochastic cell rescaling.
- Display subsection percentages of total time correctly in the `.timings` output file.

### Performance

- Skip inactive terms in Guff pair-potential calculations.

### Build and Compatibility

- Allow builds without ASE even when built-in SLAKOS data is unavailable.
- Make native optimizations and link-time optimization configurable, and use available compiler caches and faster linkers automatically.
- Add support for Clang and Apple Clang.
- Add support for building Debug mode with Clang
- Installed PQ now resolves reference data, external-QM scripts, and Slakos files relative to its installation prefix.

### Documentation

- Rework the quick start, examples, troubleshooting, setup-file guidance, and reference manual.

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
