# Turbomole validation (Perl vs Python drivers)

Needs Turbomole (`dscf`, `grad`, `ricc2` on `PATH`), Perl and Python 3.9+.

```
python3 tests/qm-validation/validate_turbomole.py src/QM/scripts
```

It runs the Perl and the Python driver on water, pure QM and with point
charges, over four steps, and checks that `qm_forces` and `qm_charges` are
byte-identical. Options: `--template-ricc2`, `--template-scf`,
`--template-ridft` (RI-DFT is only checked with its own template), `--work`.
Attach the printed `results.tar.gz` to the pull request.
