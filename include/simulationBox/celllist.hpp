#ifndef _CELL_LIST_H_

#define _CELL_LIST_H_

#include "cell.hpp"
#include "simulationBox.hpp"

#include <vector>

namespace simulationBox
{
    class CellList;
}

/**
 * @class CellList
 *
 * @brief CellList is a class for cell list
 *
 */
class simulationBox::CellList
{
  private:
    bool _activated = false;

    std::vector<Cell> _cells;

    vector3d::Vec3D   _cellSize;
    vector3d::Vec3Dul _nCells          = {7, 7, 7};
    vector3d::Vec3Dul _nNeighbourCells = {0, 0, 0};

  public:
    void setup(const SimulationBox &);
    void updateCellList(SimulationBox &);

    void determineCellSize(const SimulationBox &);
    void determineCellBoundaries(const SimulationBox &);

    void addNeighbouringCells(const SimulationBox &);
    void addCellPointers(Cell &);

    size_t            getCellIndex(const vector3d::Vec3Dul &cellIndices) const;
    vector3d::Vec3Dul getCellIndexOfMolecule(const SimulationBox &, const vector3d::Vec3D &);

    void activate() { _activated = true; }
    bool isActivated() const { return _activated; }

    /***************************
     * standatd getter methods *
     ***************************/

    vector3d::Vec3D   getCellSize() const { return _cellSize; }
    Cell             &getCell(const size_t index) { return _cells[index]; }
    std::vector<Cell> getCells() const { return _cells; }

    /***************************
     * standatd setter methods *
     ***************************/

    void setNumberOfCells(const size_t nCells) { _nCells = {nCells, nCells, nCells}; }
};

#endif   // _CELL_LIST_H_