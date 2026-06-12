# Langevin Friction/Sigma Reproducer

This reproducer checks that the Langevin random-force amplitude `_sigma`
depends on friction, while `setFriction()` updates only `_friction` and does not
recompute `_sigma`.

Run from the repository root:

```sh
python3 docs/reproducers/langevin-friction-sigma/verify_langevin_friction_sigma.py
```

The script exits with status 0 when it reproduces the current source-level
inconsistency and demonstrates, with the same formula shape, that changing
friction should change the noise amplitude.

Relevant source:

- `src/thermostat/langevinThermostat.cpp`
- `tests/src/thermostat/testThermostat.cpp`
- `include/output/references/referenceFiles/langevin.ref.bib`

References:

- M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
  Oxford University Press, 2017.
