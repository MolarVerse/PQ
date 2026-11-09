# ORCA RI-MP2 with PQ

Requires Python >= 3.9 and ORCA 6.1.1 (including its matching MPI runtime).
Load ORCA before starting PQ, for example `module load orca/6.1.1`.
`ORCA_EXE` can override the executable with a full path. ORCA's driver is
launched directly; do not launch the driver with `mpirun`.

`run-pure.in` and `water.rst` provide a complete two-step pure-QM example.
Run `PQ run-pure.in` from this directory after loading ORCA.

Add to a QM or QM/MM input:

```
qm_prog = orca;
qm_script = orca_rimp2.py;
orca_file = orca.template;
```

Copy `orca.template` from this directory to the calculation directory. Adjust
its basis, auxiliary basis, `%pal`, `%maxcore`, reference type and multiplicity.
For a custom/external script, or a static/container build, use
`qm_script_full_path = /absolute/path/to/orca_rimp2.py;` instead of `qm_script`.
The script and ORCA run in PQ's current working directory. Use separate working
directories for independent simulations.

The five required template placeholders are:

- `__METHOD__`: `HF`, `HF NoIter CalcGuessEnergy`, `RI-MP2 NoIter EnGrad`,
  or `RI-MP2 EnGrad`, plus the driver's `NoAutoStart` setting.
- `__SCF__`: converged SCF settings or zero iterations for frozen evaluations.
- `__GUESS__`: `MORead` and the explicit source GBW file, when applicable.
- `__POINTCHARGES__`: an external point-charge directive only in polarization.
- `__CHARGE__`: integer charge of the current QM region.

Use `* xyzfile __CHARGE__ 1 orca_coord.xyz` for a singlet, changing `1` and
`RHF` consistently for another reference. Do not put extra SCF, point-charge,
MO-input, optimization, gradient, or multi-job controls in the template; the
driver owns these. Basis/resource settings are shared by all stages.

## Calculation protocol

Without embedding charges the script runs one converged RI-MP2 gradient job.
It returns that job's total energy, gradients and relaxed MP2 Mulliken charges.

With embedding charges it runs:

1. `orca_polarized`: converged HF with the external charges.
2. `orca_frozen_hf`: HF `MORead NoIter CalcGuessEnergy` without charges, reading
   `orca_polarized.gbw`. This evaluates the internal HF energy without updating
   the polarized orbitals.
3. `orca_mp2`: RI-MP2 `MORead NoIter EnGrad` without charges, independently
   reading the same `orca_polarized.gbw`, requesting a relaxed MP2 density.

The energy sent to PQ is stage 2's HF energy plus stage 3's RI-MP2 correlation
energy. Gradients and relaxed MP2 atomic charges come from stage 3. The frozen
MP2 job's reported total energy is deliberately not used: without a separately
computed HF energy, some ORCA versions report only its correlation energy.

This follows PQ's Turbomole/QMCF polarization prescription. The no-field
reference is nonstationary; its analytic gradients must not be interpreted as
validated derivatives of the entire embedded-to-frozen energy protocol. This
is not a conventional electrostatic-embedding interface returning forces on
external charges. PQ continues to calculate its classical QM/MM interactions.
Compare this prescription with Turbomole and assess the intended MD behavior
before production use.

ORCA coordinates are in Angstrom; its point-charge file contains a count then
`q x y z` rows (charge in e, position in Angstrom). Only inactive SMOOTHING and
POINT_CHARGE molecules enter the file. PQ reads `qm_forces` as energy in
Hartree followed by Cartesian gradients in Hartree/Bohr, and `qm_charges` as
one numeric charge per QM atom. PQ handles the gradient-to-force sign and units.

The script validates process exit, normal termination, SCF convergence for
converged stages, output sizes, finite numbers, atom order and charge sum.
It rejects missing results and removes stale result files. Saved orbitals are
reused only after a successful complete step, with compatible template,
element sequence, charge, embedding mode and executable. PQ forces a new guess
on its first execution and QM-region changes; exact smoothing also resets the
guess for every configuration.

Inspect `orca_polarized.out`, `orca_frozen_hf.out`, and `orca_mp2.out` to diagnose
failures. Stage files are replaced on subsequent steps.

ORCA documentation:
- https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/initialguess.html#calculating-only-the-energy-of-an-input-density
- https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/coordinates.html#inclusion-of-point-charges
