#!/usr/bin/env python3
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CELL_LIST_SOURCE = ROOT / "src/simulationBox/celllist.cpp"
PARSER_SOURCE = ROOT / "src/input/inputFileParser/cellListInputParser.cpp"
EQUIVALENCE_TEST = ROOT / "tests/src/potential/testBruteForceCellListEquivalence.cpp"


def require(condition, message):
    if not condition:
        raise SystemExit(f"finding not reproduced: {message}")


def wrapped_index(cell_index, offset, cells_per_side):
    return tuple((cell_index[axis] + offset[axis]) % cells_per_side for axis in range(3))


def find_aliases(cells_per_side, n_neighbour):
    aliases = defaultdict(list)
    base_cell = (0, 0, 0)

    for i in range(-n_neighbour, n_neighbour + 1):
        for j in range(-n_neighbour, n_neighbour + 1):
            for k in range(-n_neighbour, n_neighbour + 1):
                offset = (i, j, k)
                if offset == (0, 0, 0):
                    continue
                aliases[wrapped_index(base_cell, offset, cells_per_side)].append(offset)

    return {cell: offsets for cell, offsets in aliases.items() if len(offsets) > 1}


def main():
    cell_list_source = CELL_LIST_SOURCE.read_text()
    parser_source = PARSER_SOURCE.read_text()
    equivalence_test = EQUIVALENCE_TEST.read_text()

    require("ceil(coulombCutoff / _cellSize)" in cell_list_source, "neighbor radius calculation changed")
    require("(totalCellNeighbours - 1) / 2" in cell_list_source, "half-neighbor stopping rule changed")
    require("cellNumber <= 0" in parser_source, "parser validation is no longer only positivity")
    require("kCellsPerSide >= 2 * nNeighbour + 1" in equivalence_test, "equivalence test no longer documents the precondition")

    box_edge = 10.0
    cells_per_side = 2
    cutoff = 4.0
    cell_size = box_edge / cells_per_side
    n_neighbour = math.ceil(cutoff / cell_size)
    required_cells = 2 * n_neighbour + 1

    aliases = find_aliases(cells_per_side, n_neighbour)

    print(f"cell_size: {cell_size}")
    print(f"n_neighbour: {n_neighbour}")
    print(f"cells_per_side: {cells_per_side}")
    print(f"required cells_per_side: {required_cells}")
    print(f"aliased neighbor cells: {aliases}")

    require(cells_per_side > 0, "example would not pass current positive-only parser check")
    require(cells_per_side < required_cells, "example unexpectedly satisfies the precondition")
    require(bool(aliases), "no periodic neighbor aliasing found")

    print("finding reproduced: accepted cell-list settings can alias neighbor offsets")


if __name__ == "__main__":
    main()
