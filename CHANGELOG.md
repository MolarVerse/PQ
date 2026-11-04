# Changelog

User-visible changes to PQ are documented here. Build, CI, test, and
implementation changes are documented in
[DEV-CHANGELOG.md](DEV-CHANGELOG.md).

## Next Release

<!-- insertion marker -->
## [v1.0.1](https://github.com/MolarVerse/PQ/releases/tag/v1.0.1) - 2026-11-04

### Bug Fixes

- Fix MACE runs failing at startup with `module 'mace.calculators' has no attribute 'MACE_OFF'`; PQ again passes the MACE calculator and model names (e.g. `mace_off`, `small-0b`).

## [v1.0.0](https://github.com/MolarVerse/PQ/releases/tag/v1.0.0) - 2026-11-03

### New Features

- Add "fixed_axis" keyword to enable keeping lattice parameters constant during pressure coupling
- Add the forced_layer_list keyword to specify molecules which are forced to be LAYER molecules in the hybrid calculations
- Add potassium in water example QM/MM input to the examples directory
- Add script for Turbomole RI-DFT calculations
- Print the inner region center of hybrid type calculations (e.g. QM/MM) to a separate output file with the file ending .center.xyz
- Add QM/MM hybrid type calculations with two smoothing options (hotspot, exact)
- Add timing subsections to external QM runners
- Add intra- and intermolecular water models (SPC, SPC_E, SPC_Fw, qSPC_Fw, SPC_DC, H2O-DC, TIP3P, OPC3, SPC-mTR, TIP3P-mTR)

### Changes

- QM/MM: the reported QM energy now has the QM reference energies subtracted; the subtracted sum is printed as E(QM-REF) next to N(SM-MOL) in the `.info` and `.en` files.
- The forced_inner_list keyword has been renamed to forced_core_list

### Bug Fixes

- Fix QM timeouts of ASE calculations aborting PQ; the Python calculation is now interrupted and a regular error is raised.
- Fix stale box dimensions, angles, and volume by caching them instead of tracking ad-hoc "has changed" flags, so scaling and manostat updates are reflected consistently everywhere the box is read.
- Cell-list assignment wraps escaped atom positions correctly and rejects non-finite coordinates with a clear error instead of segfaulting.
- Invalid or overflowing cell-list dimensions are rejected before storage is allocated.
- Fix SHAKE/distance constraints and cell-list setup silently producing NaN/Inf forces or an unstable layout on degenerate geometry (e.g. colliding atoms, or the box collapsing); these now throw a descriptive exception instead.
- Fix DFTB+ QM/MM energies counting the QM–point-charge Coulomb interaction twice.
- Fix exact QM/MM smoothing counting intramolecular non-bonded energy twice in Coulomb, non-Coulomb and total energy.
- Fix exact smoothing with 32 or more smoothing molecules silently using a wrong set of configurations; too many smoothing molecules now raise an error.
- External-QM calculations now fail cleanly on process errors or incomplete result files, including when paths contain spaces.
- `forced_core_list`, `forced_layer_list` and `forced_outer_list` take molecule indices only; a selection-grammar value now gives an input error instead of being read as atom indices.
- Fix examples/h2o_mm: the Coulomb coefficients in guff.dat did not match the charges of moldescriptor.dat any more (the GUFF reader checks them), so the example stopped with an error during setup.
- Fix inner_region_center rejecting atom index 0, the documented default.
- Fix optimization and Hessian runs without max-learning-rate: the maximum learning rate is optional, but PQ stopped with 'Setting value is not set and no default is available' when it was not given (for example examples/oh_mm_hessian/run-01.in).
- Fix the sign of the M-RATTLE velocity correction. M-SHAKE runs added velocity along constrained bonds instead of removing it, so the thermostat cooled the real motion and rigid water became over-structured with far too slow diffusion.
- Optimizer and Hessian runs that are stopped by the learning rate strategy now report its error messages instead of the generic "Optimizer did not converge" message, which hid them.
- Fix the build with BUILD_WITH_PYTHON_BINDINGS=ON: the first configure no longer aborts after a successful pip install of PQAnalysis (and no longer installs a stray "update" package), and the Python-enabled selection code compiles again.
- fix broken build pipeline for python bindings
- Fix atom selections through PQAnalysis (e.g. `inner_region_center = res(LIG)`) crashing PQ, and a second selection failing after the first.
- Add the optional `qm_reference_energy` line to moltype blocks in the moldescriptor file. QM/MM subtracts it for each QM molecule, so the total energy no longer jumps when molecules enter or leave the QM region.
- Fix QM timeouts of external QM programs aborting PQ; the QM process is now stopped and a regular error is raised.
- Preserve molecular geometry during QM/MM pressure coupling and correct molecular and triclinic pressure calculations.
- Correct Berendsen relaxation rates, stochastic cell noise, and cached kinetics after velocity rescaling.
- Warn about atomic virial for pressure coupling of multi-atom molecules and reject invalid cell updates before changing the simulation state.
- `remove_net_force` now removes the net QM force over the QM atoms only. In QM/MM runs it no longer spreads part of it onto MM atoms.
- Fix ring-polymer MD under MPI: non-root ranks no longer write the trajectory, restart, energy, and reference output files, removing a filesystem race that could abort the run with "Could not open file".
- Temperature ramps validate their schedule and reach the requested target across partial final intervals.
- Berendsen thermostat: a relaxation time shorter than the time step now gives a clear error instead of NaN velocities. Nose-Hoover rejects a zero target temperature; it and velocity rescaling reject zero degrees of freedom.
- Throw a exc::QMRunnerException when an external QM runner does not finish successfully
- output of timings detail sections in `.timings` was not sorted
- Fix the flexible SPC and MTR water models silently producing NaN forces for degenerate geometries (a hydrogen on the oxygen, or both hydrogens on the same ray from it); they now stop with a descriptive error.
- Fix missing citation entry when only an intramolecular water model is configured without an intermolecular one: the reference file is now added based on whether the intramolecular model itself is set, not on whether the intermolecular model is set.
- Temperature rescaling rejects zero kinetic energy before division and handles a zero target from a finite temperature.

### Build and Compatibility

- Remove the `BUILD_WITH_KOKKOS` option and Kokkos capability; PQ now uses the CPU MM implementation for all builds.

### Documentation

- Document that QM atoms with moltype 0 always count as neutral in the total QM charge.
- The `h2o_mm-mshake` example now writes every 50th step, ships an O-O RDF input for PQAnalysis and lists the expected RDF and diffusion values.

## [v0.7.1](https://github.com/MolarVerse/PQ/releases/tag/v0.7.1) - 2026-08-18

### Documentation

- Document the units used by topology setup-file indices, constraints, and force constants.

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
