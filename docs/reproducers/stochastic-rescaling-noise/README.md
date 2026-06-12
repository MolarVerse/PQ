# Stochastic Cell-Rescaling Noise Reproducer

This reproducer checks the review finding that the semi-isotropic,
anisotropic, and full-anisotropic stochastic cell-rescaling paths reuse one
Gaussian random draw for multiple independent cell degrees of freedom.

Run from the repository root:

```sh
python3 docs/reproducers/stochastic-rescaling-noise/verify_stochastic_rescaling_noise.py
```

The script exits with status 0 when each affected `calculateMu` implementation
contains one `getNormalDistribution` call while applying the resulting random
value to multiple scaling components.

Relevant source:

- `src/manostat/stochasticRescalingManostat.cpp`
- `include/output/references/referenceFiles/stochastic_rescaling.ref.bib`

References:

- M. Bernetti and G. Bussi, "Pressure control using stochastic cell rescaling,"
  *Journal of Chemical Physics* 153, 114107 (2020), DOI: `10.1063/5.0020514`.
