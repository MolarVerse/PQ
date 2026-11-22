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
  Set `PQ_BUILD_DIR` (e.g. `PQ_BUILD_DIR=.build pytest integration_tests`) to use a different build
  directory than `build`; `os.path.join` makes an absolute path work too.
  CI runs them in the Release jobs on x86_64 and arm (`ci_build.yml`), in a conda environment with
  `dftbplus`, `xtb`, `pyscf==2.9.0`, `fennol`, `jax`, `ase` and `pqanalysis`.
- Test data lives in `<name>.zip` next to the test directory (extracted once into `.zip_cache/`);
  each test copies it into `tmpdir/`, runs `PQ`, and asserts that `stderr` is empty.
- `analysis.check_pq_output(prefix, folder, ref_folder="ref_data", force_atol=0.0)` compares with the reference:
  the **trajectory** (`.xyz`), **velocities** (`.vel`), **forces** (`.force`), **charges**
  (`.chrg`) with `isclose`, and the **energy file** (`.en`, all columns except the last two, which
  are the momentum and the loop time as far as I can tell) plus its column description.
  The reference files are named `ref_data*` inside each zip; a test variant uses its own
  `ref_data_<variant>` directory. Tests that use dftb+ pass `force_atol=DFTB_FORCE_ATOL`, an
  absolute tolerance for its noisy forces; all other tests compare forces with the relative one only.
- A failing test says what went wrong. If `PQ` exits with a non-zero status, `execute_pq` fails with
  the command, the exit status, the end of stdout (where PQ prints its error) and stderr. If a file
  differs from its reference, `check_pq_output` names the file and reports the worst value (frame,
  atom, column, both values), how many values are out of tolerance, and which frame headers (the
  box) differ. The messages are made in `diagnostics.py`, which has unit tests in
  `scripts/tests/test_integration_diagnostics.py`.
- `oh_mm_hessian` does not use reference data: it checks that the Hessian file is a finite
  6 x 6 matrix, the info file, and the error message of one invalid input (its Hessian is the zero
  matrix). `hessian_bond` checks the values against the analytic Hessian of a harmonic bond.
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
| `cgo` `test_cgo_nose_hoover_ramp` | `run-01_nose_hoover.in` | `mm-md` | same | Nose-Hoover chain (length 3, `coupling_frequency = 3000`) with a ramp from 900 to 998.15 K; deterministic, compared with `ref_data_nose_hoover`; fails on NaN (a chain driven far from the system temperature, e.g. a ramp start of 600 K, blows up) |
| `cgo` `test_cgo_stochastic_thermostat_follows_the_seed` (2 tests) | `run-01.in` with `velocity_rescaling` or `langevin` and `random_seed` | `mm-md` | same | no reference (stochastic): the same seed gives the same xyz/vel/force/energies bit for bit, another seed another run, and the thermostat changes the run compared with NVE |
| `cgo` `test_cgo_berendsen_manostat_isotropy` (2 tests) | `run-01_NPT_anisotropic.in`, `run-01_NPT_xy.in` | `mm-md` | same | the NPT run with a manostat relaxation time of 0.1 ps and `isotropy = anisotropic` or `xy` (semi-isotropic); compared with `ref_data_NPT_anisotropic` / `ref_data_NPT_xy` (the box is in every frame header); fails on NaN |
| `cgo` `test_cgo_anisotropic_manostat_keeps_a_fixed_axis` | the anisotropic NPT run with `fixed_axis = z` | `mm-md` | same | z stays at 27.44205774 in every frame, x moves by more than 1e-3, and without the fixed axis z moves (no reference) |
| `cgo` `test_cgo_anisotropic_manostat_with_the_cell_list` (2 tests) | the anisotropic NPT run with `cell-list = on`, with the default and with `cell-number = 16` | `mm-md` | same | three different, differently moving box lengths; must equal the brute-force reference `ref_data_NPT_anisotropic` |
| `cgo` `test_cgo_stochastic_manostat_follows_the_seed` | NPT run with `manostat = stochastic_rescaling`, `random_seed` and a temperature ramp (900 to 998.15 K) | `mm-md` | same | no reference (stochastic): the same seed gives the same xyz/vel/force/box/energies bit for bit, another seed another box, and the ramp changes the run |
| `cgo` `test_cgo_NPT_cellList` | `run-01_NPT.in` with `cell-list = on` | `mm-md` | same | the NPT run with the cell list against the brute-force NPT reference (`ref_data_NPT`); the box hardly moves (0.01 %) |
| `cgo` `test_cgo_NPT_cellList_follows_a_moving_box` | `run-01_NPT.in` with `cell-list = on; cell-number = 16` and a manostat relaxation time of 0.1 ps | `mm-md` | same | the box moves between 26.6 and 31.4 Angstrom in 10 steps and the neighbour cell count changes from 6 to 7; every frame (xyz, vel, force), the box and the energies must equal the same run without the cell list (no reference data) |
| `cgo` `test_cgo_restart_round_trip` (3 tests) | `run-01.in` (NVE), `run-01_NPT.in`, `run-01.in` with a Nose-Hoover chain | `mm-md` | same | a 10-step run against 5 steps plus 5 more from the `.rst` the first part wrote; the final `.rst` and the last frames of `.xyz`, `.vel` and `.force` must be **identical to the byte** (the restart file keeps every digit), and the step counter, the box and the chain levels are checked. Builds its three inputs from the existing ones, no reference data |
| `chig` `test_chig` | `run-01.in` | `mm-md` | chignolin in water, 7382 atoms, 10 steps of 0.5 fs | `force-field = on` (bonds, angles, dihedrals, impropers, LJ from the parameter file, topology with 191 linker `*` entries), intra-nonbonded file, SHAKE/RATTLE, Wolf |
| `dftbplus` | `run-01.in` | `qm-md` | 9 atoms, 100 Å box, 2 steps | external DFTB+ runner with the periodic-stress script and a custom `dftb_file` |
| `dftbplus` `test_dftbplus_triclinic_box` | `run-01_triclinic.in` | `qm-md` | 9 atoms, triclinic box 9 x 10 x 11 Å with angles 75, 80, 70°, 2 steps | NVE; compared with `ref_data_triclinic`, and the lattice written for DFTB+ (`geom.gen`) must have exactly these lengths and angles; the box is small so that the periodic images interact |
| `dftbplus` `test_dftbplus_triclinic_box_full_anisotropic_manostat` | `run-01_triclinic_npt.in` | `qm-md` | same, 6 steps | Berendsen thermostat + Berendsen manostat with `isotropy = full_anisotropic` (DFTB+ stress): all three angles and lengths change; compared with `ref_data_triclinic_npt` |
| `dftbplus` `test_dftbplus_triclinic_box_anisotropic_manostat_keeps_the_angles` | the NPT input with `isotropy = anisotropic` | `qm-md` | same | the angles stay at 75, 80, 70° in every frame and the lengths move (no reference) |
| `dftbplus` `test_dftbplus_full_anisotropic_manostat_from_a_box_with_right_angles` | the NPT input with a start box without angles (`Box 9 10 11`) and `isotropy = full_anisotropic` | `qm-md` | same | the angles must change from 90° (issue #1297: the box used to stay orthorhombic and only the molecules were sheared) |
| `cgo` `test_mm_md_refuses_a_full_anisotropic_manostat` | `run-01_NPT_anisotropic.in` with `isotropy = full_anisotropic` | `mm-md` | cgo | PQ exits with "isotropy = full_anisotropic needs a triclinic box" (issue #1297; used to explode the box silently); to be replaced when MM-MD supports triclinic boxes (issue #910) |
| `cgo` `test_mm_md_refuses_a_triclinic_box` | `run-01.in` with a triclinic `Box` line in the start file | `mm-md` | cgo | PQ exits with "Triclinic box is only supported for QM-MD and RP-QM-MD" (issue #910); has to be replaced by a real test when MM-MD supports triclinic boxes |
| `ase-dftbplus` | `run-01.in` | `qm-md` | 9 atoms, 2 steps | ASE DFTB+ with the `3ob` Slater-Koster set |
| `ase-xtb` | `run-01.in` | `qm-md` | 9 atoms, 2 steps | ASE xTB, GFN2 |
| `pyscf` | `run-01.in` | `qm-md` | 9 atoms, 2 steps | PySCF runner with the Hartree-Fock script |
| `fennol` | `run-01.in` | `qm-md` | 9 atoms, 2 steps | FeNNol ANI-1x model on CPU (`gpu_preprocessing = false`) |
| `mace` (2 inputs) | `run-01.in`, `run-02.in` | `qm-md` | 9 atoms, 2 steps | MACE-OFF and MACE-MP, small model. **Skipped in CI** (needs `torch`, `mace` and a CUDA device) |
| `qm-mm/*` (8 tests) | `run-01.in` in 8 directories | `qmmm-md` | one K ion + 2000 waters (6001 atoms), box 39 Å, 2 steps | DFTB+ as QM program, Berendsen thermostat; the 8 = smoothing `hotspot`/`exact` x `brute_force`/`cell_list` x (`force-field = bonded` with topology \| `force-field = off` with a water model) |
| `oh_mm_hessian` (4 tests) | `run-01.in`, `run-02-no-opt.in` | `mm-hessian` | 2 atoms | optimisation then Hessian, Hessian alone, with and without `max-learning-rate`, and the error for a minimum learning rate not below the maximum |
| `hessian_bond` `test_numerical_hessian_equals_the_analytic_one` (6 tests) | `run.in` with `hessian_builder` central, forward, five_point and a stretched or an equilibrium start file | `mm-hessian` | one tilted O-H bond (harmonic, k = 500, r0 = 1 Å), `force-field = bonded` | the Hessian equals the analytic `k (u u^T + (1 - r0/r)(1 - u u^T))` blocks within the error of each stencil (central 4e-5, forward 0.1, five point 6e-11 at a displacement of 0.001 Å; the stretched bond must also not be *more* accurate than its stencil allows); symmetric; shifting both atoms costs no energy (except forward) |
| `hessian_bond` `test_central_hessian_error_follows_the_displacement` | displacement 0.001 and 0.002 | `mm-hessian` | same | the central error grows as the displacement squared (ratio 3 to 5) |
| `hessian_bond` `test_hessian_info_file_describes_the_run` | five_point | `mm-hessian` | same | builder, displacement, definition (-dF_i/dx_j) and unit (kcal/mol/Å²) of the info file |
| `hessian_bond` `test_analytic_builder_is_refused_for_the_mm_evaluator` | `hessian_builder = analytic` | `mm-hessian` | same | "The selected evaluator does not support analytic Hessians" |

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
| `mm-hessian` | yes for a harmonic bond (`hessian_bond`: values against the analytic Hessian); `oh_mm_hessian` only checks the shape (its two atoms do not interact, the Hessian is the zero matrix). Not: angles, dihedrals, non-bonded forces, periodic images, more than 2 atoms |
| `qm-rpmd` (ring polymer) | no. Elsewhere: regression (`qm-rpmd-mpi-smoke`, MPI build, one smoke step), example |

### Integrator, thermostat, ramp, manostat, kinetics

| Feature | Status |
|---|---|
| Velocity Verlet (the only integrator) | yes |
| Thermostat Berendsen | yes (`cgo` NPT/ramp, all `qm-mm`) |
| Thermostat Nose-Hoover chain | yes (`cgo`, with a ramp, against a reference; also the restart test) |
| Thermostat velocity rescaling, Langevin | partly: reproducibility with `random_seed` and an effect on the run, no reference values and no check that the temperature is reached (see stochastic features). Elsewhere: unit, perf benchmarks |
| Temperature ramp (`start_temp`, `temp_ramp_steps`, `temp_ramp_frequency`) | yes, with Berendsen and with the Nose-Hoover chain (see coupling section for the others) |
| `end_temp` as synonym of `temp` | no. Elsewhere: unit |
| Manostat Berendsen, isotropic | yes (`cgo` NPT) |
| Manostat stochastic rescaling | partly: reproducibility with `random_seed`, with a temperature ramp, and an effect on the box; no reference values and no check that the pressure is reached. Elsewhere: unit |
| `isotropy` semi/anisotropic/full, `fixed_axis`, `compressibility` | partly, Berendsen only: `anisotropic` and `xy` against references, `fixed_axis = z`, and the anisotropic box with the cell list. `full_anisotropic` in QM-MD (also from a box with 90° angles, see the triclinic row below) and its refusal in MM-MD. No: `xz`/`yz`, the other fixed axes, `compressibility`, and the semi-isotropic/anisotropic stochastic manostat. Elsewhere: unit |
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
| Cell list | yes in NVE (`cgo`), with a Berendsen manostat that moves the box (`cgo`), with the water models and the hybrid code (`qm-mm`); not with a force field (see coupling) |
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
| Hessian builder `central` | yes (values against the analytic Hessian, error order checked) |
| Hessian builders forward, five-point | yes (same). Analytic: refused for the MM evaluator (tested). Elsewhere: unit |
| `optimize_before_hessian = false` | yes |

### Input, output and restart

| Feature | Status |
|---|---|
| Start file, moldescriptor, GUFF, topology, parameter, intra-nonbonded files | yes (by use) |
| `output_freq` 1 and 2 | yes. Larger values and 0: no |
| Output files compared: `.xyz .vel .force .chrg .en` | yes |
| Output files written but **not compared**: `.rst .box .stress .vir .mom .instant_en .info` (content), `.log`, `.timings`, `.ref` | see "What the checks cannot see" |
| `overwrite_output`, `include_output_metadata`, custom file names | partly: `overwrite_output = on` in the QM tests; the rest no |
| Restart from a file that PQ wrote | yes, exactly (byte for byte): MM state, box and Nose-Hoover chain state (NVE, Berendsen NPT, Nose-Hoover chain on `cgo`). No: a restart in the middle of a temperature ramp, with the Langevin thermostat or with the cell list, and QM or QM/MM restarts |
| Triclinic boxes | yes for QM-MD with DFTB+ (NVE, and a manostat with `full_anisotropic` that changes the angles, `anisotropic` that keeps them). A cubic start box with `full_anisotropic` becomes a box that can change its angles (QM-MD). MM-MD does not support triclinic boxes yet (issue #910); the refusal, also for `full_anisotropic`, is tested. No: ring polymer QM-MD, ASE and the other QM programs, the cell list and `fixed_axis` with a triclinic box |
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
need their *own* integration test, with or without a ramp. Nose-Hoover now has one (with a ramp);
Langevin and velocity rescaling only the seed test. The readers of the global value are
exercised by none.

**Neighbour search x manostat (covered for the Berendsen manostat).** A manostat changes the box
every step, so the cells have to be rebuilt. `test_cgo_NPT_cellList_follows_a_moving_box` runs a
box that moves by 15 % and crosses the point where the number of neighbouring cells changes, and
compares with brute force. With the default 7 cells that number never changes in this run and a
stale cell layout would still give the right result, so the cell number matters for this test. The
anisotropic Berendsen manostat (three different box lengths) is covered with the cell list; the
stochastic rescaling manostat is not.

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
stochastic manostat additionally reads the thermostat target (see ramp): it now runs with a ramp
(seed test), but there is no reference and no check of the pressure; the anisotropic variants are
run with the Berendsen thermostat only.

**Job type x output.** The trajectory writers are shared by MD and optimisation, but the optimiser
and Hessian engines write different files (`.opt`, `.hessian`). The Hessian of a harmonic bond is
checked against the analytic one; angles, dihedrals and non-bonded forces are not, and
`oh_mm_hessian` only checks shape and finiteness.

## What the checks cannot see

- **Length.** At most 10 steps. History-dependent state (Verlet half steps, Berendsen/NH chain
  history, ramp end, neighbour-list ageing, drift of constraints) is barely exercised, and
  anything that needs hundreds of steps (equilibration, conserved quantities, thermostat
  statistics) is not.
- **Unchecked outputs.** Restart, box, stress, virial, momentum and instant-energy files and the
  log/timings files are written but not compared, so an error only in one of them passes. The
  `.rst` file is only checked indirectly, by the restart round trip.
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
  a 10-step reference test. What is covered: Langevin and velocity rescaling are run with a fixed
  `random_seed` and must reproduce themselves exactly (and differ for another seed); the
  statistical test is still missing.
- **Long-run physics** (RDF, diffusion, energy conservation over thousands of steps) and
  performance. These belong in a nightly job and in the perf gate, not here.
- **Other platforms and build types** (Windows, macOS, Debug, static/LTO builds, Singularity
  containers, Python bindings): separate CI jobs build them but no pytest run uses them.

## Suggested next tests

In rough order of value for the cost; each is a **new variant** of an existing cheap test (the
`cgo` pattern below), so existing references stay untouched.

1. **Statistical test of the Langevin and velocity-rescaling thermostats** (average temperature
   within a tolerance over a longer run); the seed tests only prove reproducibility.
2. The remaining isotropies and fixed axes (`xz`, `yz`, the other `fixed_axis` values,
   `compressibility`), a triclinic box with the ring polymer or the ASE runner, and a statistical
   test of the stochastic rescaling manostat (average pressure).
3. **QM/MM that moves a molecule across a zone**: a start file with a molecule next to a boundary,
   or more steps, plus a non-zero `core_radius` and a `forced_*_list`.
4. **M-SHAKE** on a rigid-water system, and **cell list with the force field and SHAKE** on `chig`.
5. **Water models** in a pure MM run, and the remaining models with different intra/inter pairs.
6. **Comparison of the other outputs** (`.rst`, `.box`, `.stress`, `.vir`, `.mom`) in
   `check_pq_output`, which costs nothing but reference size.
7. The Hessian of angles, dihedrals and non-bonded forces (analytic or against forces), and the
   Hessian after `optimize_before_hessian = true`.
8. **`mm-opt` and `Adam`/decaying learning rates** with a real number of optimisation steps.

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
