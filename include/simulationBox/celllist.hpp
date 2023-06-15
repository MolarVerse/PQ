#ifndef _CELL_LIST_H_

#define _CELL_LIST_H_

#include <vector>

#include "cell.hpp"
#include "simulationBox.hpp"

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
    std::vector<Cell> _cells;

    bool _activated = false;

    Vec3Dul _nCells = {7, 7, 7};
    Vec3Dul _nNeighbourCells = {0, 0, 0};

    Vec3D _cellSize;

public:
    Cell &getCell(const size_t index) { return _cells[index]; }
    std::vector<Cell> getCells() const { return _cells; }

    void activate() { _activated = true; }
    [[nodiscard]] bool isActivated() const { return _activated; }

    void setNumberOfCells(const size_t nCells) { _nCells = {nCells, nCells, nCells}; }

    [[nodiscard]] Vec3D getCellSize() const { return _cellSize; }

    void setup(const simulationBox::SimulationBox &);
    void determineCellSize(const simulationBox::SimulationBox &);
    void determineCellBoundaries(const simulationBox::SimulationBox &);
    void addNeighbouringCells(const simulationBox::SimulationBox &);
    void addCellPointers(Cell &);
    void updateCellList(simulationBox::SimulationBox &);

    Vec3Dul getCellIndexOfMolecule(const simulationBox::SimulationBox &, const Vec3D &);
    [[nodiscard]] size_t getCellIndex(const Vec3Dul &cellIndices) const
    {
        return cellIndices[0] * _nCells[1] * _nCells[2] + cellIndices[1] * _nCells[2] + cellIndices[2];
    }
};

#endif // _CELL_LIST_H_