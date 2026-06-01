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
- Perf gate now triggers only on PRs that touch code, benchmarks, build
  config, or the gate itself; doc / changelog / unrelated-workflow PRs no
  longer spend ~5 min building benchmarks for a no-op diff
- Perf gate now caches the base-branch instruction counts keyed on the base
  commit SHA; on a hit the whole base checkout + build + callgrind run is
  skipped (≈ half the workflow), with identical numerics (callgrind is
  deterministic per binary)

### Bug Fixes

- `BerendsenThermostat::applyThermostat` no longer produces NaN
  velocities when called with zero kinetic energy: the `T_target / T`
  ratio would diverge and `0.0 * Inf = NaN` corrupted every atom's
  velocity. The thermostat now skips silently when `_temperature` is
  (approximately) zero, mirroring the velocity-rescaling NaN guard
  added in v0.6.2
- `kernel::distVecAndDist2(pos_i, pos_j)` (no-PBC overload) now
  correctly returns `dot(r_ij, r_ij)` for the squared distance instead
  of `dot(pos_i, pos_j)`. The only caller, `MShake::initMShakeReferences`,
  was therefore storing wrong reference squared bond lengths, so
  `applyMShake` was driving the constraint toward an incorrect target
  and could not converge

### Internal

- Added missing trailing newline at end of `src/simulationBox/simulationBox.cpp`
- `CoulombPotential::getCoulombRadiusCutOff()`, `getCoulombEnergyCutOff()`
  and `getCoulombForceCutOff()` are now `inline` in the header so the
  per-pair call in `Potential::calculateSingleInteraction` can be elided
  without LTO
- Cell-list rebuild no longer constructs a temporary `std::vector<size_t>`
  per atom: `try_emplace(cellIndexScalar, std::vector<size_t>({j}))` +
  fallback `push_back` replaced by a single `mapCellIndexToAtomIndex[cellIndexScalar].push_back(j)`
- `utilities::isZero<T>(a)` helper added to `mathUtilities.hpp`,
  centralizing the exact-zero check (`a == T(0)`). Callers that need a
  tolerance can still use `compare(a, T(0), tol)`

### Tests

- New unit test asserting that `PotentialBruteForce::calculateForces` and
  `PotentialCellList::calculateForces` produce identical per-atom forces and
  intermolecular energies for the same configuration, guarding the
  brute-force/cell-list equivalence under hot-path refactors
- `testResetKinetics` revived: the file was 500 lines of commented-out
  tests targeting an old 6-arg constructor; replaced with 6 working
  tests covering the 7-arg constructor's getters, the temperature /
  momentum / angular-momentum setters, `resetTemperature` (lambda
  rescaling, finite output), `resetMomentum` (drives total linear
  momentum to ~0), `resetAngularMomentum` (finite velocities), and
  `resetForces` (zeros per-atom forces). The previously 0%-covered
  97-line `src/resetKinetics/resetKinetics.cpp` is now exercised
- Coverage for `ManostatSettings`, `ConstraintSettings`, `FileSettings`,
  and `ConvergenceSettings` static-class setters/getters (manostat type
  + isotropy string round-trips, shake/rattle tolerances + max-iters,
  input/output file-name round-trips, optional energy/force convergence
  thresholds)
- Coverage for the `kernel::dist*` family (no-PBC `distVec` /
  `distVecAndDist2` matching analytical subtraction and squared norm;
  PBC overloads choosing the minimum-image displacement on a known
  orthorhombic box; consistency between `distSquared`, `distVec`, and
  `distVecAndDist2` under PBC) — these tests caught a real bug in the
  no-PBC `distVecAndDist2` that's also fixed here
- Coverage for `opt::LearningRateStrategy` and its three concrete
  variants (`ConstantLRStrategy`, `ConstantDecayLRStrategy`,
  `ExpDecayLR`): constructor stores the initial rate, the constant
  strategy's `updateLearningRate` is a no-op, the constant-decay
  variant decays only on frequency hits, the exponential-decay variant
  matches the analytical `initial * exp(-decay * step / nEpochs)` and
  is monotonically decreasing, and the base class's
  `checkLearningRate` clamps to the min/max bounds and appends a
  warning
- Expanded coverage for `mathUtilities` (`compare` with tolerance,
  `compare(Vec3D)` with tolerance, `kroneckerDelta`); `Thermostat`
  variants (`VelocityRescaling`: tau getter/setter, thermostat type,
  apply doesn't produce NaN; `Langevin`: sigma after construction,
  setters/getters, sigma recompute on target-temperature change,
  thermostat type; `NoseHoover`: thermostat type, coupling-frequency
  setter/getter, chi/zeta index setter); and `Manostat` variants
  (`BerendsenManostat`: tau and compressibility getters, manostat type
  and isotropy; isotropy and type for `SemiIsotropic`, `Anisotropic`,
  and `FullAnisotropic` Berendsen)
- Coverage for `JCouplingType` and `JCouplingForceField` (operator==
  contract, getter/setter coverage, default symmetry flags)
- Expanded coverage for `PhysicalData` energy/virial accumulators
  (`addCoulombEnergy`, `addNonCoulombEnergy`, `addBondEnergy`,
  `addAngleEnergy`, `addDihedralEnergy`, `addImproperEnergy`,
  `addRingPolymerEnergy`, `addVirial`); and `CellList` lifecycle
  (`activate`/`deactivate`/`isActive` toggle; `clone` preserves the
  configured cell counts, neighbour-cell count, and activation state)
- Coverage for `opt::Convergence` (all four `ConvStrategy` branches
  in `checkConvergence`, `calcEnergyConvergence` / `calcForceConvergence`
  flag flips above/below threshold, disabled-flag short-circuits,
  threshold getters)
- Coverage for `opt::Optimizer` via `SteepestDescent` (constructor stores
  `nEpochs`, `maxHistoryLength`, `clone`, history-index out-of-range
  exception, `updateHistory` populates deques and trims to the history
  cap, offset-indexed `getEnergy` / `getMaxForce` / `getRMSForce` /
  `getForces` / `getPositions`, `setConvergence` / `getConvergence`
  round-trip, `hasConverged` for flat-energy/zero-force vs. large-force)
- Coverage for `setup::OptimizerSetup` (free `setupOptimizer` no-op when
  not an opt job; `setupLearningRateStrategy` for `CONSTANT`,
  `CONSTANT_DECAY`, `EXPONENTIAL_DECAY` and exception paths for
  `LINESEARCH_WOLFE` / `NONE` / missing decay; `setupMinMaxLR`
  min ≥ max guard; `setupEmptyOptimizer` for `STEEPEST_DESCENT`,
  `ADAM`, exception for `NONE`; `setupConvergence` writes back into
  the optimizer; `setupEvaluator` for `MM_OPT` and exception for
  non-opt jobs; full `setup()` happy path)
- Coverage for `setup::HybridSetup` (free `setupHybrid` no-op when QMMM
  inactive; `parseSelectionNoPython` for single index, comma list,
  range, mixed range+list, empty input throws; `parseSelection`
  empty-string returns `{0}`, sorts and dedupes, throws on
  letters without Python bindings; `setup()` throws not-implemented)
- Coverage for `output::OptOutput::write` (step column, all four
  convergence-threshold columns, `ABSOLUTE` zeros the relative-energy
  indicator, `RELATIVE` zeros the absolute-energy indicator, disabled
  energy convergence zeros both energy indicators)
- Coverage for `output::TimingsOutput::write` (header rows present,
  `Total` row present, sub-timer registered via `Timer::startTimingsSection`
  is listed in the per-section block)
- Coverage for both `JCouplingSection` parsers (parameter-file: 7- and
  8-element lines with `+` / `-` / `0` symmetry, wrong-count throws;
  topology-file: keyword, `endedNormally`, 5-element happy path,
  wrong-count throws, duplicate-atom-index throws)
- Coverage for `opt::SteepestDescent::update` (single-step
  `pos_new = pos + lr * force`, old position stored, PBC wrap on
  out-of-box updated positions, no-op at zero learning rate)
- Coverage for `opt::Adam::update` (analytic step-1 reduction to
  `pos_new ≈ pos + lr * sign(force)` with per-component sign
  preservation, old position stored, PBC wrap, no-op on zero force;
  both constructors and `clone` / `maxHistoryLength`)
- Coverage for `opt::MMEvaluator` (`clone` produces an `MMEvaluator`
  instance; `evaluate()` walks copy-old, force-reset, cell-list update,
  brute-force inter-non-bonded, intra-non-bonded, bonded-interaction
  steps without throwing on a minimal one-molecule box; per-atom force
  buffer is zeroed when there are no inter-molecular pairs)

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
