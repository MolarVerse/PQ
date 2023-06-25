#ifndef _CELL_H_

#define _CELL_H_

#include "molecule.hpp"
#include "simulationBox.hpp"

#include <vector>

namespace simulationBox
{
    class Cell;
}

/**
 * @class Cell
 *
 * @brief Cell is a class for cell
 *
 */
class simulationBox::Cell
{
  private:
    std::vector<Molecule *>          _molecules;
    std::vector<std::vector<size_t>> _atomInidices;
    std::vector<Cell *>              _neighbourCells;

    vector3d::Vec3D   _lowerBoundary = {0, 0, 0};
    vector3d::Vec3D   _upperBoundary = {0, 0, 0};
    vector3d::Vec3Dul _cellIndex     = {0, 0, 0};

  public:
    void clearMolecules() { _molecules.clear(); }
    void clearAtomIndices() { _atomInidices.clear(); }

    void addMolecule(Molecule &molecule) { _molecules.push_back(&molecule); }
    void addMolecule(Molecule *molecule) { _molecules.push_back(molecule); }
    void addNeighbourCell(Cell *cell) { _neighbourCells.push_back(cell); }
    void addAtomIndices(const std::vector<size_t> &atomIndices) { _atomInidices.push_back(atomIndices); }

    /***************************
     * standatd getter methods *
     ***************************/

    size_t            getNumberOfMolecules() const { return _molecules.size(); }
    size_t            getNumberOfNeighbourCells() const { return _neighbourCells.size(); }
    vector3d::Vec3D   getLowerBoundary() const { return _lowerBoundary; }
    vector3d::Vec3D   getUpperBoundary() const { return _upperBoundary; }
    vector3d::Vec3Dul getCellIndex() const { return _cellIndex; }

    Molecule               *getMolecule(const size_t index) const { return _molecules[index]; }
    std::vector<Molecule *> getMolecules() const { return _molecules; }

    Cell               *getNeighbourCell(const size_t index) const { return _neighbourCells[index]; }
    std::vector<Cell *> getNeighbourCells() const { return _neighbourCells; }

    const std::vector<size_t> &getAtomIndices(const size_t index) const { return _atomInidices[index]; }

    /***************************
     * standatd setter methods *
     ***************************/

    void setLowerBoundary(const vector3d::Vec3D &lowerBoundary) { _lowerBoundary = lowerBoundary; }
    void setUpperBoundary(const vector3d::Vec3D &upperBoundary) { _upperBoundary = upperBoundary; }
    void setCellIndex(const vector3d::Vec3Dul &cellIndex) { _cellIndex = cellIndex; }
};

#endif   // _CELL_H_