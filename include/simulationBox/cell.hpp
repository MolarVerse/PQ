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
    std::vector<std::vector<int>> _atomInidices;
    std::vector<Cell *> _neighbourCells;
    std::vector<double> _lowerBoundary = {0, 0, 0};
    std::vector<double> _upperBoundary = {0, 0, 0};

    std::vector<int> _cellIndex = {0, 0, 0};

public:
    void addMolecule(Molecule *molecule) { _molecules.push_back(molecule); }
    std::vector<Molecule *> getMolecules() const { return _molecules; }
    int getNumberOfMolecules() const { return int(_molecules.size()); }

    void addNeighbourCell(Cell *cell) { _neighbourCells.push_back(cell); }
    int getNeighbourCellSize() const { return int(_neighbourCells.size()); }
    Cell *getNeighbourCell(int index) const { return _neighbourCells[index]; }
    std::vector<Cell *> getNeighbourCells() const { return _neighbourCells; }

    void setLowerBoundary(const std::vector<double> &lowerBoundary) { _lowerBoundary = lowerBoundary; }
    std::vector<double> getLowerBoundary() const { return _lowerBoundary; }

    void setUpperBoundary(const std::vector<double> &upperBoundary) { _upperBoundary = upperBoundary; }
    std::vector<double> getUpperBoundary() const { return _upperBoundary; }

    void setCellIndex(const std::vector<int> &cellIndex) { _cellIndex = cellIndex; }
    std::vector<int> getCellIndex() const { return _cellIndex; }

    void addMolecule(Molecule &molecule) { _molecules.push_back(&molecule); }
    Molecule *getMolecule(int index) const { return _molecules[index]; }
    void clearMolecules() { _molecules.clear(); }

    void addAtomIndices(std::vector<int> atomIndices) { _atomInidices.push_back(atomIndices); }
    std::vector<int> getAtomIndices(int index) const { return _atomInidices[index]; }
    void clearAtomIndices() { _atomInidices.clear(); }
};

#endif // _CELL_H_