# ORCA HF, DFT and RI-MP2 with PQ

Requires Python >= 3.9 and ORCA 6.1.1 (including its matching MPI runtime).
Load ORCA before starting PQ, for example `module load orca/6.1.1`.
`ORCA_EXE` can override the executable with a full path. ORCA's driver is
launched directly; do not launch the driver with `mpirun`.

`run-pure.in` and `water.rst` provide a complete two-step pure-QM example.
Run `PQ run-pure.in` from this directory after loading ORCA.

Choose a bundled driver and its matching template:

| Driver | Example template | Reference | Atomic charges |
| --- | --- | --- | --- |
| `orca_rimp2.py` | `orca.template` | HF / RI-MP2 | Relaxed MP2 Mulliken |
| `orca_hf.py` | `orca_hf.template` | HF | SCF Mulliken |
| `orca_dft.py` | `orca_dft.template` | Kohn-Sham DFT | SCF Mulliken |

`run-pure-hf.in` and `run-pure-dft.in` are complete two-step examples using the
same `water.rst`. For example, `PQ run-pure-dft.in`. To use the same drivers in
QM/MM, retain the hybrid settings and select the appropriate script/template.

The DFT example uses `PBE RKS`; change `PBE` in the template to select another
SCF functional, for example `B3LYP`. Match the reference and multiplicity:
`RHF`/`RKS` for a closed-shell singlet, or `UHF`/`UKS` with the appropriate
multiplicity for an unrestricted system. Double hybrids and post-SCF methods
are outside the HF/DFT drivers' protocol; use the dedicated RI-MP2 driver for
RI-MP2. Availability of ORCA gradients and frozen-density behavior must be
checked for other specialized functionals or approximations.
Avoid basis sets where ORCA removes linear-dependent functions (often diffuse
ones): `MORead NoIter` then gives wrong results (ORCA manual, section 2.20.5).

For the existing RI-MP2 example, add to a QM or QM/MM input:

```
qm_prog = orca;
qm_script = orca_rimp2.py;
orca_file = orca.template;
```

Copy `orca.template` from this directory to the calculation directory. Adjust
its basis, auxiliary basis, `%pal`, `%maxcore`, reference type and multiplicity.
For a custom/external script, or a static/container build, use
`qm_script_full_path = /absolute/path/to/orca_rimp2.py;` instead of `qm_script`.
When copying scripts outside the installation, copy `orca_common.py` beside
`orca_rimp2.py`, and both `orca_common.py` and `orca_scf.py` beside the HF/DFT
entry points. Installed/build-tree scripts include these helpers automatically.
The script and ORCA run in PQ's current working directory. Use separate working
directories for independent simulations.

The five required template placeholders are:

- `__METHOD__`: stage-specific method and run-type keywords, plus the driver's
  `NoAutoStart` setting. The HF/DFT drivers insert `HF`/`DFT`, respectively; the
  DFT functional (such as `PBE`) stays in the shared template.
- `__SCF__`: converged SCF settings or zero iterations for frozen evaluations.
- `__GUESS__`: `MORead` and the explicit source GBW file, when applicable.
- `__POINTCHARGES__`: an external point-charge directive only in polarization.
- `__CHARGE__`: integer charge of the current QM region.

Use `* xyzfile __CHARGE__ 1 orca_coord.xyz` for a singlet, changing `1` and
`RHF` consistently for another reference. Do not put extra SCF, point-charge,
MO-input, optimization, gradient, or multi-job controls in the template; the
driver owns these. Basis/resource settings are shared by all stages.

## RI-MP2 calculation protocol

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

## HF/DFT calculation protocol

Pure QM uses one converged `HF EnGrad` or `DFT EnGrad` calculation. Its energy,
Cartesian gradients and SCF Mulliken charges are sent to PQ.

With embedding charges the HF/DFT drivers use two jobs:

1. `orca_hf_polarized` / `orca_dft_polarized`: converge HF/KS orbitals in the
   external point-charge field. A compatible previous step is only an initial
   guess; the SCF is reconverged with the current geometry and field.
2. `orca_hf_frozen` / `orca_dft_frozen`: read that polarized GBW file and run
   `NoIter CalcGuessEnergy EnGrad` without point charges, with `MaxIter 0` and
   `ConvForced false`. The same basis, functional and reference settings apply.

The second job supplies all three results: no-field total energy, QM gradients
and SCF Mulliken charges. There is no separate correlation correction to add.
The output energy is cross-checked against the energy in the native `.engrad`
file. Polarized-stage energy is not returned to PQ. In the pure branch the job
names are `orca_hf` / `orca_dft`. Output files are replaced on subsequent steps.
HF/DFT restart metadata includes the driver mode so switching HF/DFT/RI-MP2
cannot silently reuse another driver's reference.

## Shared conventions and limitations

This follows PQ's Turbomole/QMCF polarization prescription. The no-field
reference is nonstationary; its analytic gradients must not be interpreted as
validated derivatives of the entire embedded-to-frozen energy protocol. This
is not a conventional electrostatic-embedding interface returning forces on
external charges. PQ continues to calculate its classical QM/MM interactions.
This limitation applies to HF/DFT as well as RI-MP2: analytic gradients at the
no-field nonstationary reference are not established derivatives of the entire
polarization-to-frozen-energy protocol. Compare the prescription with Turbomole
and assess the intended MD behavior before production use.

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
