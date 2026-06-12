# Triclinic Minimum-Image Reproducer

This reproducer demonstrates that fractional-coordinate rounding is not a
nearest-image algorithm for a general triclinic box.

Run from the repository root:

```sh
python3 docs/reproducers/triclinic-minimum-image/verify_triclinic_minimum_image.py
```

The script mirrors `TriclinicBox::calcShiftVector` and compares it with a
bounded brute-force lattice-vector search. It exits with status 0 when the
current implementation chooses a longer image than the brute-force nearest
periodic image.

Relevant source:

- `src/potential/potential.cpp`: pair forces use `box.calcShiftVector(dxyz)`.
- `src/box/triclinicBox.cpp`: `calcShiftVector` rounds fractional coordinates.
- `src/box/triclinicBox.cpp`: `applyPBC` has a 27-image fallback, but the
  force path does not use it.

References:

- M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
  Oxford University Press, 2017.
- D. Frenkel and B. Smit, *Understanding Molecular Simulation*, 2nd ed.,
  Academic Press, 2002.
