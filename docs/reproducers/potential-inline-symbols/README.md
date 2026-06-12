# Potential Equivalence Linkage Reproducer

This reproducer checks why `testBruteForceCellListEquivalence` currently
segfaults before validating brute-force and cell-list equivalence.

The source files define the virtual overrides as `inline` in `.cpp` files. In
the current shared-library build, the test binary keeps undefined references to
`PotentialBruteForce::calculateForces` and `PotentialCellList::calculateForces`,
while `libpotential.dylib` does not export those symbols.

Run from the repository root after building `build-stochastic-rescale-ci`:

```sh
python3 docs/reproducers/potential-inline-symbols/verify_potential_inline_symbols.py
```

The script exits with status 0 when it reproduces the missing-symbol condition.
If the test executable is present, it also runs the equivalence test and reports
the current non-zero result.

Relevant source:

- `src/potential/potentialBruteForce.cpp`
- `src/potential/potentialCellList.cpp`
- `tests/src/potential/testBruteForceCellListEquivalence.cpp`

References:

- C++ inline function linkage rule: an inline function definition must be
  reachable in every translation unit that odr-uses it. Defining an inline
  virtual override only in a `.cpp` file is not a reliable exported shared
  library entry point.
