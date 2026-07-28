# OH MM Hessian

This example shows the molecular mechanics Hessian workflow for a minimal OH system.

The optimized run computes a geometry optimization and then the Hessian in one input file:

```bash
PQ run-01.in
```

The optimized input uses deliberately loose one-step convergence settings so the example is fast and deterministic. Production inputs should use physically meaningful optimization thresholds.

The no-optimization run computes the Hessian directly at the input geometry:

```bash
PQ run-02-no-opt.in
```

The generated `*.hessian` matrix can be analyzed with PQAnalysis:

```bash
pqanalysis vibrations vibrations.in
```

The analysis input also writes visual mode files. `mode-*.xyz` files are sinusoidal XYZ animations for selected modes and can be opened with ASE:

```bash
ase gui mode-6.xyz
```

`modes.xyz` is an extended XYZ file with all selected mode vectors and metadata.

`run-01.in` writes `oh-opt.hessian` and `oh-opt.hessian.info`. `run-02-no-opt.in` writes `oh-no-opt.hessian` and `oh-no-opt.hessian.info`.
