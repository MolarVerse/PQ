#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MOLECULE_SOURCE = ROOT / "src/simulationBox/molecule.cpp"
MOLECULE_TYPE_SOURCE = ROOT / "src/simulationBox/moleculeType.cpp"
TEST_FIXTURE = ROOT / "tests/include/simulationBox/testMolecule.hpp"


def require(condition, message):
    if not condition:
        raise SystemExit(f"finding not reproduced: {message}")


def ranges_unique_duplicate_tail_size(values):
    """Model the size of the tail returned by std::ranges::unique."""
    if not values:
        return 0

    compacted = [values[0]]
    for value in values[1:]:
        if value != compacted[-1]:
            compacted.append(value)

    return len(values) - len(compacted)


def current_count(values):
    duplicate_tail_size = ranges_unique_duplicate_tail_size(values)
    return len(values) - duplicate_tail_size


def main():
    molecule_source = MOLECULE_SOURCE.read_text()
    molecule_type_source = MOLECULE_TYPE_SOURCE.read_text()
    fixture = TEST_FIXTURE.read_text()

    require("std::ranges::unique(extAtomTypes)" in molecule_source, "Molecule no longer uses ranges::unique on extAtomTypes")
    require("std::ranges::unique(_atomTypes)" in molecule_type_source, "MoleculeType no longer uses ranges::unique on _atomTypes")
    require("_atom1->setExternalAtomType(1)" in fixture, "test fixture changed")
    require("_atom2->setExternalAtomType(2)" in fixture, "test fixture changed")
    require("_atom3->setExternalAtomType(2)" in fixture, "adjacent-duplicate fixture changed")

    adjacent_duplicate = [1, 2, 2]
    non_adjacent_duplicate = [1, 2, 1]

    print(f"current count for {adjacent_duplicate}: {current_count(adjacent_duplicate)}")
    print(f"current count for {non_adjacent_duplicate}: {current_count(non_adjacent_duplicate)}")
    print(f"correct set count for {non_adjacent_duplicate}: {len(set(non_adjacent_duplicate))}")

    require(current_count(adjacent_duplicate) == 2, "adjacent duplicate no longer matches existing test expectation")
    require(current_count(non_adjacent_duplicate) == 3, "non-adjacent duplicate no longer overcounts")
    require(len(set(non_adjacent_duplicate)) == 2, "reference unique count changed unexpectedly")

    print("finding reproduced: non-adjacent duplicate atom types are overcounted")


if __name__ == "__main__":
    main()
