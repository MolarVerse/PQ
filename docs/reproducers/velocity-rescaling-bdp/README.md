# Velocity-Rescaling Thermostat Reproducer

This reproducer checks two review findings for the `velocity_rescaling`
thermostat:

1. The implementation follows the documented single-Gaussian expression rather
   than the exact Bussi-Donadio-Parrinello canonical velocity-rescaling kinetic
   energy update.
2. The zero-temperature path has no guard, so `targetTemperature / temperature`
   is non-finite before velocity scaling.

Run from the repository root:

```sh
python3 docs/reproducers/velocity-rescaling-bdp/verify_velocity_rescaling.py
```

The script exits with status 0 when it reproduces the current implementation
properties above.

Relevant source:

- `src/thermostat/velocityRescalingThermostat.cpp`
- `src/thermostat/berendsenThermostat.cpp`
- `docs/sphinx/src/userGuide/inputFile.rst`
- `include/output/references/referenceFiles/velocity_rescaling.ref.bib`

References:

- G. Bussi, D. Donadio, and M. Parrinello, "Canonical sampling through velocity
  rescaling," *Journal of Chemical Physics* 126, 014101 (2007),
  DOI: `10.1063/1.2408420`.
