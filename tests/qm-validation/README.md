# Turbomole validation (Perl vs Python drivers)

Needs Turbomole (`define`, `dscf`, `grad`, `ricc2` on `PATH`), Perl and Python 3.9+.

```
python3 tests/qm-validation/validate_turbomole.py src/QM/scripts
```

It runs the Perl driver on water, pure QM and with point charges, over four
steps, and records every Turbomole call. The Python driver then runs against a
replay of those calls, so both read the same Turbomole output. It checks that
the Python driver gives Turbomole the same inputs and writes byte-identical
`qm_forces` and `qm_charges`. Two separate Turbomole runs are not compared:
Turbomole's own results differ in the last digits from run to run.

Options: `--template-ricc2`, `--template-scf`, `--template-ridft` (RI-DFT is
only checked with its own template), `--work`. Attach the printed
`results.tar.gz` to the pull request.
