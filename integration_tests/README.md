# Integration tests

End-to-end tests: the real `PQ` binary is run on an input file and its output is compared with
stored reference data. This page says **what they cover, what they do not cover, and what they
cannot cover**, including the combinations of features that are *not independent* of each other.
It was written by reading the inputs of all tests and the code paths behind them; where a
statement is an inference from the code and not something a test proves, it says so.

Contents: [How the tests work](#how-the-tests-work) ·
[The tests](#the-tests) · [Coverage by feature](#coverage-by-feature) ·
[Feature coupling](#feature-coupling-which-coverage-carries-over-and-which-does-not) ·
[What the checks cannot see](#what-the-checks-cannot-see) ·
[What cannot be covered here](#what-cannot-be-covered-here) · [Suggested next tests](#suggested-next-tests) ·
[Adding a test](#adding-a-test)

Legend used below: **yes** = a test runs it and compares the result; **partly** = covered only in a
setting that leaves relevant variants open (the note says which); **no** = no integration test.
"Elsewhere" names the other place that exercises it: **unit** = `tests/src` (GoogleTest),
**regression** = `tests/regression` (ctest, runs in every build job), **example** = `examples/`
(documentation inputs; only `oh_mm_hessian` is also run here).

## How the tests work

- `pytest integration_tests` from the repository root, with `build/apps/PQ` built (Release).
  CI runs them in the Release jobs on x86_64 and arm (`ci_build.yml`), in a conda environment with
  `dftbplus`, `xtb`, `pyscf==2.9.0`, `fennol`, `jax`, `ase` and `pqanalysis`.
- Test data lives in `<name>.zip` next to the test directory (extracted once into `.zip_cache/`);
  each test copies it into `tmpdir/`, runs `PQ`, and asserts that `stderr` is empty.
- `analysis.check_pq_output(prefix, folder, ref_folder="ref_data")` compares with the reference:
  the **trajectory** (`.xyz`), **velocities** (`.vel`), **forces** (`.force`), **charges**
  (`.chrg`) with `isclose`, and the **energy file** (`.en`, all columns except the last two, which
  are the momentum and the loop time as far as I can tell) plus its column description.
  The reference files are named `ref_data*` inside each zip; a test variant uses its own
  `ref_data_<variant>` directory.
- `oh_mm_hessian` does not use reference data: it checks that the Hessian file is a finite
  6 x 6 matrix, the info file, and the error message of one invalid input.
- Runs are **2 to 10 steps** long. These tests find "the output changed", not "the physics is right
  over a long run".

## The tests

| Test (directory) | Input | Job | System | What it combines |
|---|---|---|---|---|
| `cgo` `test_cgo` | `run-01.in` | `mm-md` | 1475 atoms, cubic box 27.4 Å, 10 steps | NVE, Buckingham + GUFF file, Wolf Coulomb, brute-force neighbours |
| `cgo` `test_cgo_cellList` | `run-01_celllist.in` | `mm-md` | same | same as above with `cell-list = on` |
| `cgo` `test_cgo_NPT` | `run-01_NPT.in` | `mm-md` | same | Berendsen thermostat + Berendsen manostat (isotropic), molecular virial, `output_freq = 2` |
| `cgo` `test_cgo_atomic_virial` | `run-01_atomic_virial.in` | `mm-md` | same | the NPT run with `virial = atomic` |
| `cgo` `test_cgo_temperature_ramp` | `run-01_temp_ramp.in` | `mm-md` | same | Berendsen thermostat with `start_temp`, `temp_ramp_steps`, `temp_ramp_frequency`; also checks the ramp lines of the log |
| `chig` `test_chig` | `run-01.in` | `mm-md` | chignolin in water, 7382 atoms, 10 steps of 0.5 fs | `force-field = on` (bonds, angles, dihedrals, impropers, LJ from the parameter file, topology with 191 linker `*` entries), intra-nonbonded file, SHAKE/RATTLE, Wolf |
| `dftbplus` | `run-01.in` | `qm-md` | 9 atoms, 100 Å box, 2 steps | external DFTB+ runner with the periodic-stress script and a custom `dftb_file` |
| `ase-dftbplus` | `run-01.in` | `qm-md` | 9 atoms, 2 steps | ASE DFTB+ with the `3ob` Slater-Koster set |
| `ase-xtb` | `run-01.in` | `qm-md` | 9 atoms, 2 steps | ASE xTB, GFN2 |
| `pyscf` | `run-01.in` | `qm-md` | 9 atoms, 2 steps | PySCF runner with the Hartree-Fock script |
| `fennol` | `run-01.in` | `qm-md` | 9 atoms, 2 steps | FeNNol ANI-1x model on CPU (`gpu_preprocessing = false`) |
| `mace` (2 inputs) | `run-01.in`, `run-02.in` | `qm-md` | 9 atoms, 2 steps | MACE-OFF and MACE-MP, small model. **Skipped in CI** (needs `torch`, `mace` and a CUDA device) |
| `qm-mm/*` (8 tests) | `run-01.in` in 8 directories | `qmmm-md` | one K ion + 2000 waters (6001 atoms), box 39 Å, 2 steps | DFTB+ as QM program, Berendsen thermostat; the 8 = smoothing `hotspot`/`exact` x `brute_force`/`cell_list` x (`force-field = bonded` with topology \| `force-field = off` with a water model) |
| `oh_mm_hessian` (4 tests) | `run-01.in`, `run-02-no-opt.in` | `mm-hessian` | 2 atoms | optimisation then Hessian, Hessian alone, with and without `max-learning-rate`, and the error for a minimum learning rate not below the maximum |

Each water model appears in exactly one QM/MM test, so it is always combined with one particular
smoothing method and neighbour search: `spc_fw` (exact, brute force), `spc-mtr` (exact, cell
list), `tip3p` (hotspot, brute force), `spc_e` (hotspot, cell list). In every test intra- and
intermolecular model are the same.

## Coverage by feature

### Job types

| Feature | Status |
|---|---|
| `mm-md` | yes (`cgo`, `chig`) |
| `qm-md` | yes, DFTB+, ASE DFTB+, ASE xTB, PySCF, FeNNol (MACE skipped) |
| `qmmm-md` | partly: DFTB+ only, 2 steps, no zone changes (see coupling section) |
| `mm-opt` | no. Elsewhere: unit (`tests/src/opt`) |
| `mm-hessian` | partly: 2 atoms, central differences only; values are not compared |
| `qm-rpmd` (ring polymer) | no. Elsewhere: regression (`qm-rpmd-mpi-smoke`, MPI build, one smoke step), example |

### Integrator, thermostat, ramp, manostat, kinetics

| Feature | Status |
|---|---|
| Velocity Verlet (the only integrator) | yes |
| Thermostat Berendsen | yes (`cgo` NPT/ramp, all `qm-mm`) |
| Thermostat velocity rescaling, Langevin, Nose-Hoover chain | no. Elsewhere: unit, perf benchmarks |
| Temperature ramp (`start_temp`, `temp_ramp_steps`, `temp_ramp_frequency`) | yes, with Berendsen (see coupling section for the other thermostats) |
| `end_temp` as synonym of `temp` | no. Elsewhere: unit |
| Manostat Berendsen, isotropic | yes (`cgo` NPT) |
| Manostat stochastic rescaling | no. Elsewhere: unit |
| `isotropy` semi/anisotropic/full, `fixed_axis`, `compressibility` | no. Elsewhere: unit |
| Virial `molecular` / `atomic` | yes, both, only inside the NPT run (the pressure column of `.en` is compared) |
| `init_velocities`, `random_seed`, `density`, `floating_point_type` | no. Elsewhere: unit |
| Reset kinetics (`nscale`, `fscale`, `nreset`, `freset`, angular, forces) | no. Elsewhere: unit |
| `remove_net_force`, `dispersion`, `qm_force_dist` | no. Elsewhere: unit |

### Interactions

| Feature | Status |
|---|---|
| Wolf long range | yes (`cgo`, `chig`) |
| Default Coulomb treatment (no `long_range`) | yes, in the QM/MM and QM tests |
| Reaction field, shifted long range | no. Elsewhere: unit, perf benchmark |
| Non-Coulomb Buckingham via GUFF file | yes (`cgo`) |
| Non-Coulomb LJ from the parameter file | yes (`chig`, through `force-field = on`) |
| Non-Coulomb Morse, LJ 9-12, other GUFF mixtures | no. Elsewhere: unit |
| Force field bonded terms: bonds, angles, dihedrals, impropers | yes (`chig`; bonds and angles also `qm-mm`) |
| Linker `*` entries | yes (`chig`, 191 entries; none in `qm-mm`) |
| J-coupling terms | no, and the term is not part of the force calculation yet (issue #904) |
| Intra-nonbonded file | yes (`chig`) |
| Cell list | yes in NVE (`cgo`), with the water models and the hybrid code (`qm-mm`); **not with a manostat** (see coupling) |
| Brute force neighbours | yes |
| Water models | partly: `spc_fw`, `spc-mtr`, `tip3p`, `spc_e`, only in QM/MM. No: `spc`, `qspc_fw`, `spc_dc`, `h2o_dc`, `opc3`, `tip3p_mtr`, and no pure-MM water run |

### Constraints

| Feature | Status |
|---|---|
| SHAKE / RATTLE (topology `SHAKE` section) | yes (`chig`), with tolerances `1e-3` / `1e9` that are loose on purpose |
| Rigid water through the water models (`shake-iter`, `rattle-iter`) | yes (`qm-mm` hotspot tests) |
| M-SHAKE (`mshake_file`, `shake = mshake`) | no. Elsewhere: regression (SHAKE vs M-SHAKE give the same trajectory), unit, example |
| Distance constraints (`DIST_CONSTRAINTS`) | no. Elsewhere: unit |

### QM programs and hybrid QM/MM

| Feature | Status |
|---|---|
| DFTB+ external runner | yes, with the periodic-stress script (the only DFTB+ script shipped) and a custom `dftb_file` |
| ASE DFTB+ (`3ob`) | yes. `matsci`, custom Slater-Koster sets, `third_order`, `hubbard_derivs`: no |
| ASE xTB | partly: GFN2 only (GFN1, IPEA1: no) |
| PySCF | partly: the HF script only (`pyscf_mp2.py`: no) |
| FeNNol | yes, one model, CPU only |
| MACE | skipped in CI |
| Turbomole | no (commercial, see below). Elsewhere: unit (command construction), example |
| `qm_loop_time_limit` | no. Elsewhere: unit |
| Hybrid zones `hotspot` / `exact` smoothing | partly: both, but `core_radius = 0`, 2 steps |
| `forced_core_list`, `forced_layer_list`, `forced_outer_list`, `inner_region_center` | no. Elsewhere: unit |
| `qm_charges` (QM or MM charges), point-charge shells | partly: default only |
| QM/MM with a QM program other than DFTB+ | no |

### Optimiser and Hessian

| Feature | Status |
|---|---|
| Steepest descent, constant learning rate | partly: `nstep = 1`, energy and force thresholds of `1e10`, so the first convergence check ends it |
| Adam, decaying/exponential/line-search learning rates | no. Elsewhere: unit |
| Convergence strategies and thresholds | no. Elsewhere: unit |
| `max-learning-rate` | yes, with and without, plus the error message |
| Hessian builder `central` | yes (shape and finiteness only) |
| Hessian builders forward, five-point, analytic | no. Elsewhere: unit |
| `optimize_before_hessian = false` | yes |

### Input, output and restart

| Feature | Status |
|---|---|
| Start file, moldescriptor, GUFF, topology, parameter, intra-nonbonded files | yes (by use) |
| `output_freq` 1 and 2 | yes. Larger values and 0: no |
| Output files compared: `.xyz .vel .force .chrg .en` | yes |
| Output files written but **not compared**: `.rst .box .stress .vir .mom .instant_en .info` (content), `.log`, `.timings`, `.ref` | see "What the checks cannot see" |
| `overwrite_output`, `include_output_metadata`, custom file names | partly: `overwrite_output = on` in the QM tests; the rest no |
| **Restart from a file that PQ wrote** | **no** (the start files are hand-made or from older tools) |
| Triclinic boxes | no (not supported for MM-MD yet, issue #910); all start files are orthorhombic |
| Input errors and validation messages | one (`oh_mm_hessian`). Elsewhere: unit (`inputFileParser`, `inputValidation`) |
| Command line (arguments, `--version`, missing file) | no. Elsewhere: unit (`testPQCli`) |

## Feature coupling: which coverage carries over and which does not

Features share code in some places and not in others. "A works and B works" implies "A with B
works" only where the shared code is the *only* interaction. The cases below come from reading the
code.

**Temperature ramp (carries over for the thermostats, partly for the rest).** The ramp lives in
`Thermostat::applyTemperatureRamping`, which every MD engine calls once per step. It changes one
value, the thermostat's target temperature (through the virtual `setTargetTemperature`, so the
Langevin sigma follows), and a second one, the global `ThermostatSettings::actualTargetTemperature`.
The thermostats read the first; the **stochastic rescaling manostat, reset kinetics and the
Maxwell-Boltzmann velocities read the second**. So Berendsen covers the writer and one reader of
the first value. A ramp test per thermostat is not needed for the ramp itself, but Nose-Hoover
(chain state) and Langevin (noise) have internal state that the Berendsen test cannot reveal; they
need their *own* integration test, with or without a ramp. The readers of the global value are
exercised by none.

**Neighbour search x manostat (does not carry over).** The cell list is covered in NVE, the
manostat with brute force. A manostat changes the box every step, so the cells have to be resized
or rebuilt; no test runs both.

**Neighbour search x everything that builds pair lists.** The cell list is covered for `cgo`
(Buckingham/GUFF/Wolf) and for the water models and hybrid code in `qm-mm`, but **not with a force
field, topology, linker entries or SHAKE** (`chig` is brute force only).

**QM x MM (does not carry over).** QM-only runs cover every QM program; MM-only runs cover the
force field. A QM/MM run adds code that neither exercises: zone assignment and smoothing factors,
activation/deactivation of molecules, point charges written for the QM program, the QM force
redistribution, and removal of the net force. These are covered only by 8 runs of 2 steps with
DFTB+, a core radius of 0 and no molecule changing its zone in two steps, so **zone changes are
never exercised**. The other QM programs are never run in QM/MM.

**Water model x smoothing x neighbour search (confounded).** Four of the eleven water models are
covered, each in one fixed combination, and always with identical intra/inter model. A fault that
depends on the pair (model, smoothing) or (intra model, inter model) would not show.

**Force field x shake x long range (one combination).** `chig` is the only run with the force
field and SHAKE together, and with Wolf; the bonded-only QM/MM runs use no SHAKE.

**Thermostat x manostat x virial.** One thermostat with one manostat and both virial types. The
stochastic manostat additionally reads the thermostat target (see ramp), the anisotropic variants
change the cell differently; none of that is exercised.

**Job type x output.** The trajectory writers are shared by MD and optimisation, but the optimiser
and Hessian engines write different files (`.opt`, `.hessian`); only shape and finiteness of the
Hessian are checked, so a wrong Hessian would pass.

## What the checks cannot see

- **Length.** At most 10 steps. History-dependent state (Verlet half steps, Berendsen/NH chain
  history, ramp end, neighbour-list ageing, drift of constraints) is barely exercised, and
  anything that needs hundreds of steps (equilibration, conserved quantities, thermostat
  statistics) is not.
- **Unchecked outputs.** Restart, box, stress, virial, momentum and instant-energy files and the
  log/timings files are written but not compared, so an error only in one of them passes. The
  `.rst` file is the most important: it is never read back.
- **Tolerance.** `isclose` defaults hide differences below its tolerances, and the last two energy
  columns are skipped on purpose.
- **Only the final state of a short run.** Intermediate frames are compared only when
  `output_freq` writes them.

## What cannot be covered here

- **Turbomole.** Commercial software; it cannot be installed in CI. Only the command/file
  handling is tested (unit).
- **MACE.** The test exists but needs `torch`, `mace` and a CUDA device; it is skipped in CI.
- **Ring polymer MD with several MPI ranks.** The integration tests run a non-MPI Release
  binary. This needs a `BUILD_WITH_MPI` build and `mpirun`; today only a one-step smoke test runs
  (regression, `mpi-build` job). It could become an integration test if the MPI job ran pytest.
- **Interrupts (Ctrl-C), QM time limits and process kills.** Needs signals at a controlled moment;
  the result depends on timing and would be flaky. Covered by unit tests of the runners and the
  interrupt handling.
- **Stochastic features** (Langevin, stochastic rescaling manostat, random initial velocities).
  They can be made reproducible with `random_seed`, but the reference would depend on the random
  number generator of the standard library, so it is fragile across toolchains. Better covered by
  *statistical* tests (average temperature within a tolerance over a longer run), which do not fit
  a 10-step reference test.
- **Long-run physics** (RDF, diffusion, energy conservation over thousands of steps) and
  performance. These belong in a nightly job and in the perf gate, not here.
- **Other platforms and build types** (Windows, macOS, Debug, static/LTO builds, Singularity
  containers, Python bindings): separate CI jobs build them but no pytest run uses them.

## Suggested next tests

In rough order of value for the cost; each is a **new variant** of an existing cheap test (the
`cgo` pattern below), so existing references stay untouched.

1. **Restart round trip**: run 5 steps, then start from the `.rst` PQ wrote and run 5 more; the
   result must match a 10-step run. Covers restart writing *and* reading, the step counter and
   (with Nose-Hoover) the chain state.
2. **Cell list + manostat** (`cgo` NPT with `cell-list = on`): the box changes while the cells are
   in use.
3. **Nose-Hoover, Langevin and velocity-rescaling** thermostats on `cgo` (one variant each, the
   Nose-Hoover one with a ramp since it keeps internal state).
4. **Stochastic rescaling manostat** and one **anisotropic** variant, with the ramp (they read the
   global target).
5. **QM/MM that moves a molecule across a zone**: a start file with a molecule next to a boundary,
   or more steps, plus a non-zero `core_radius` and a `forced_*_list`.
6. **M-SHAKE** on a rigid-water system, and **cell list with the force field and SHAKE** on `chig`.
7. **Water models** in a pure MM run, and the remaining models with different intra/inter pairs.
8. **Comparison of the other outputs** (`.rst`, `.box`, `.stress`, `.vir`, `.mom`) in
   `check_pq_output`, which costs nothing but reference size.
9. **A value check for the Hessian** (analytic result for a harmonic OH bond) instead of shape
   only, and the other Hessian builders.
10. **`mm-opt` and `Adam`/decaying learning rates** with a real number of optimisation steps.

## Adding a test

Follow the pattern of `cgo`: add an input `run-01_<feature>.in` and a directory
`ref_data_<feature>/` to the zip (`python3 -c "import zipfile; ..."` in append mode is enough),
write a `test_<name>_<feature>` that runs the input, asserts an empty `stderr`, and calls
`check_pq_output("<prefix>", "<folder>", "ref_data_<feature>")`. Generate the reference with a
Release build with `-DBUILD_WITH_NATIVE=Off` (as CI does), run it twice to check that the files
are identical apart from the timing columns, and store only the files the comparison reads
(`.xyz .vel .force .chrg .en .info`). Prove that the test can fail: change the feature in the
extracted input and see the test go red. Keep the system small; the CI budget for these tests is
the time of the slowest Release job.
