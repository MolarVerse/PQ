# Atom-Type Counting Reproducer

This reproducer demonstrates that the current atom-type counting algorithm only
collapses adjacent duplicate atom types. A molecule with external atom-type
sequence `1,2,1` is counted as three unique atom types, although the correct
unique count is two.

Run from the repository root:

```sh
python3 docs/reproducers/atom-type-counting/verify_atom_type_counting.py
```

The script exits with status 0 when the current algorithmic model reproduces
the overcount for non-adjacent duplicates.

Relevant source:

- `src/simulationBox/molecule.cpp`
- `src/simulationBox/moleculeType.cpp`
- `src/input/guffDatReader.cpp`
- `tests/include/simulationBox/testMolecule.hpp`

References:

- C++ standard library semantics for `std::ranges::unique`: only consecutive
  equivalent elements are collapsed; callers must sort or otherwise deduplicate
  if non-adjacent duplicates should be removed.
