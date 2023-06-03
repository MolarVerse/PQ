#ifndef _CELL_H_

#define _CELL_H_

#include <vector>

#include "molecule.hpp"
#include "simulationBox.hpp"

/**
 * @class Cell
 *
 * @brief Cell is a class for cell
 *
 */
class Cell
{
private:
    std::vector<Molecule *> _molecules;
    std::vector<std::vector<size_t>> _atomInidices;
    std::vector<Cell *> _neighbourCells;
    Vec3D _lowerBoundary = {0, 0, 0};
    Vec3D _upperBoundary = {0, 0, 0};

    Vec3Dul _cellIndex = {0, 0, 0};

public:
    void addMolecule(Molecule &molecule) { _molecules.push_back(&molecule); }
    void addMolecule(Molecule *molecule) { _molecules.push_back(molecule); }
    [[nodiscard]] Molecule *getMolecule(const size_t index) const { return _molecules[index]; }
    std::vector<Molecule *> getMolecules() const { return _molecules; }

    void clearMolecules() { _molecules.clear(); }
    [[nodiscard]] size_t getNumberOfMolecules() const { return _molecules.size(); }

    void addNeighbourCell(Cell *cell) { _neighbourCells.push_back(cell); }
    [[nodiscard]] size_t getNeighbourCellSize() const { return _neighbourCells.size(); }
    [[nodiscard]] Cell *getNeighbourCell(const size_t index) const { return _neighbourCells[index]; }
    std::vector<Cell *> getNeighbourCells() const { return _neighbourCells; }

    void setLowerBoundary(const Vec3D &lowerBoundary) { _lowerBoundary = lowerBoundary; }
    [[nodiscard]] Vec3D getLowerBoundary() const { return _lowerBoundary; }

    void setUpperBoundary(const Vec3D &upperBoundary) { _upperBoundary = upperBoundary; }
    [[nodiscard]] Vec3D getUpperBoundary() const { return _upperBoundary; }

    void setCellIndex(const Vec3Dul &cellIndex) { _cellIndex = cellIndex; }
    [[nodiscard]] Vec3Dul getCellIndex() const { return _cellIndex; }

    void addAtomIndices(const std::vector<size_t> &atomIndices) { _atomInidices.push_back(atomIndices); }
    [[nodiscard]] const std::vector<size_t> &getAtomIndices(const size_t index) const { return _atomInidices[index]; }
    void clearAtomIndices() { _atomInidices.clear(); }
};

#endif // _CELL_H_