# Cell-List Neighbor Aliasing Reproducer

This reproducer demonstrates the precondition required by the current
half-neighbor cell-list algorithm:

```text
cells_per_side >= 2 * ceil(cutoff / cell_size) + 1
```

If this condition is not enforced, periodic neighbor offsets can alias to the
same physical cell. That can make the half-neighbor traversal double-count or
miss pair interactions depending on the layout.

Run from the repository root:

```sh
python3 docs/reproducers/cell-list-neighbor-alias/verify_cell_list_neighbor_alias.py
```

The script exits with status 0 when it finds aliased neighbor offsets for an
input configuration that the current parser accepts because `cell-number > 0`.

Relevant source:

- `src/simulationBox/celllist.cpp`
- `src/input/inputFileParser/cellListInputParser.cpp`
- `tests/src/potential/testBruteForceCellListEquivalence.cpp`

References:

- M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
  Oxford University Press, 2017.
- D. Frenkel and B. Smit, *Understanding Molecular Simulation*, 2nd ed.,
  Academic Press, 2002.
