# Changelog

User-visible changes to PQ are documented here. Build, CI, test, and
implementation changes are documented in
[DEV-CHANGELOG.md](DEV-CHANGELOG.md).

## Next Release

### New Features

- Add the `mm-hessian` job for molecular-mechanics Hessian calculations, with
  optional geometry optimization.
- Add Reaction Field as a long-range electrostatics method.
- Add FeNNol as an ASE-based QM runner.
- Add `mace_mode` to choose between accurate and accelerated MACE execution;
  accelerated mode requires matching cuequivariance packages.
- Add `remove_net_force` for removing the total net force from imported QM
  forces.
- Add `mshake-iter` and `mshake-tolerance` for controlling M-SHAKE convergence.

### Changes

- Rename `mace_model_size` to `mace_model`; the old keyword remains available
  with a deprecation warning.
- Validate incompatible input settings before simulation setup.

### Bug Fixes

- Fix M-SHAKE convergence, iteration limits, previous-position handling, and
  velocity corrections for constrained molecules.
- Prevent undefined forces for collinear angle configurations, including
  linear equilibrium geometries such as CO2.
- Prevent the Berendsen thermostat from producing invalid velocities when the
  kinetic energy is zero.
- Reject non-finite energies and forces from external QM calculations instead
  of propagating them into a trajectory.
- Count non-adjacent duplicate atom types correctly.
- Reject periodic cell-list layouts in which neighbor offsets refer to the same
  cell more than once.
- Recompute Langevin noise when the friction setting changes.
- Preserve molecular geometry and wrap positions correctly during stochastic
  cell rescaling.

### Performance

- Skip inactive terms in Guff pair-potential calculations.

### Build and Compatibility

- Add support for Clang and Apple Clang.
- Allow builds without ASE even when built-in SLAKOS data is unavailable.
- Make native optimizations and link-time optimization configurable, and use
  available compiler caches and faster linkers automatically.

### Documentation

- Rework the quick start, examples, troubleshooting, setup-file guidance, and
  reference manual.

<!-- insertion marker -->
## [v0.6.4](https://github.com/MolarVerse/PQ/releases/tag/v0.6.4) - 2026-03-31

### Bug Fixes

- Correct the time-unit conversion in SHAKE velocity corrections.

## [v0.6.3](https://github.com/MolarVerse/PQ/releases/tag/v0.6.3) - 2025-11-12

### Bug Fixes

- Fix a segmentation fault when using the `bonded` force field.
- Pin Eigen to a working version.

### Enhancements

- Wrap atom positions in triclinic boxes before writing trajectory output.
- Write atom charges to `.chrg` output for pure QM-MD jobs.

## [v0.6.2](https://github.com/MolarVerse/PQ/releases/tag/v0.6.2) - 2025-08-22

### Bug Fixes

- Reject NaN and infinite values in restart-file input.
- Prevent the velocity-rescaling thermostat from producing invalid velocities.

## [v0.6.1](https://github.com/MolarVerse/PQ/releases/tag/v0.6.1) - 2025-07-25

### Enhancements

- Add `random_seed` for reproducible simulations.
- Report the QM loop time limit in the log.
- Set the default QM loop time limit to one hour.
- Refresh the example simulations and add three new examples.

### Bug Fixes

- Treat atom index zero as out of bounds in topology files.
- Preserve letter casing in `qm_script_full_path`.

## [v0.6.0](https://github.com/MolarVerse/PQ/releases/tag/v0.6.0) - 2025-04-02

### Enhancements

- Add new MACE models.
- Add the ASE-based xTB calculator.
- Add a keyword for loading a custom MACE model from a URL.
- Add an option to overwrite existing output files.

### Bug Fixes

- Print temperature setup correctly in the log output.

## [v0.5.3](https://github.com/MolarVerse/PQ/releases/tag/v0.5.3) - 2025-02-03

### Enhancements

- Add the ASE interface for DFTB+ calculations.
- Add `freset_forces` for resetting forces after each step.
- Ignore `init_velocities` when non-zero velocities are already present.
- Add a `force` option for reinitializing velocities.

### Bug Fixes

- Print volume correctly in the log output.

## [v0.5.2](https://github.com/MolarVerse/PQ/releases/tag/v0.5.2) - 2025-01-05

### Enhancements

- Give reference output its own file and `reference_file` input keyword.
- Add citations for supported QM programs, velocity Verlet, RATTLE, and PQ.
- Include BibTeX entries in `.ref` output.

### Bug Fixes

- Fix full-anisotropic coupling with the stochastic cell-rescaling manostat.

## [v0.5.1](https://github.com/MolarVerse/PQ/releases/tag/v0.5.1) - 2025-01-05

### Enhancements

- Restore old Nose-Hoover chain parameters when restarting.
- Add `dftb_file` for selecting the DFTB+ input template.
- Accept input keys case-insensitively and with either `-` or `_`.

### Bug Fixes

- Fix QM atom updates in QM-MD calculations.

## [v0.4.5](https://github.com/MolarVerse/PQ/releases/tag/v0.4.5) - 2024-07-13

### Bug Fixes

- Implement the analytic minimum-image convention for triclinic cells.

## [v0.4.4](https://github.com/MolarVerse/PQ/releases/tag/v0.4.4) - 2024-07-09

### Bug Fixes

- Fix anisotropic NPT calculations.

### Known Bugs

- The minimum-image convention for triclinic cells is approximate.

## [v0.4.3](https://github.com/MolarVerse/PQ/releases/tag/v0.4.3) - 2024-07-08

### Bug Fixes

- Correct virial evaluation in MACE NPT calculations.

### Known Bugs

- Anisotropic NPT calculations do not work correctly.
- The minimum-image convention for triclinic cells is approximate.

## [v0.4.2](https://github.com/MolarVerse/PQ/releases/tag/v0.4.2) - 2024-07-04

### Bug Fixes

- Fix segmentation faults in isotropic manostats.
- Always write the latest tag as the version number in output files.

### Known Bugs

- MACE NPT calculations do not work.
- Anisotropic NPT calculations do not work correctly.
- The minimum-image convention for triclinic cells is approximate.

## [v0.4.1](https://github.com/MolarVerse/PQ/releases/tag/v0.4.1) - 2024-07-02

### Enhancements

- Expand log output with the important simulation settings.

### Known Bugs

- Isotropic manostats can produce segmentation faults.
- MACE NPT calculations do not work.
- Anisotropic NPT calculations do not work correctly.
- The minimum-image convention for triclinic cells is approximate.

## [v0.4.0](https://github.com/MolarVerse/PQ/releases/tag/v0.4.0) - 2024-07-01

### Features

- Add M-SHAKE.
- Add the MACE neural-network potential for QM-MD calculations.
- Add the steepest-descent and ADAM optimizers.

### Known Bugs

- MACE NPT calculations do not work.
- Anisotropic NPT calculations do not work correctly.
- The minimum-image convention for triclinic cells is approximate.
