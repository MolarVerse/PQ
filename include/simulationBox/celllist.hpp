#ifndef _CELL_LIST_H_

#define _CELL_LIST_H_

#include <vector>

#include "cell.hpp"
#include "simulationBox.hpp"

/**
 * @class CellList
 *
 * @brief CellList is a class for cell list
 *
 */
class CellList
{
private:
    std::vector<Cell> _cells;

    bool _activated = false;

    size_t _nCellsX = 7;
    size_t _nCellsY = 7;
    size_t _nCellsZ = 7;

    size_t _nNeighbourCellsX;
    size_t _nNeighbourCellsY;
    size_t _nNeighbourCellsZ;

    std::vector<double> _cellSize;

public:
    Cell &getCell(const size_t index) { return _cells[index]; }
    std::vector<Cell> getCells() const { return _cells; }

    void activate() { _activated = true; }
    [[nodiscard]] bool isActivated() const { return _activated; }

    void setNumberOfCells(const size_t nCells)
    {
        _nCellsX = nCells;
        _nCellsY = nCells;
        _nCellsZ = nCells;
    }

    std::vector<double> getCellSize() const { return _cellSize; }

    void setup(const SimulationBox &);
    void determineCellSize(const SimulationBox &);
    void determineCellBoundaries(const SimulationBox &);
    void addNeighbouringCells(const SimulationBox &);
    void addCellPointers(Cell &);
    void updateCellList(SimulationBox &);
    std::vector<size_t> getCellIndexOfMolecule(const SimulationBox &, const std::vector<double> &);
    [[nodiscard]] size_t getCellIndex(const std::vector<size_t> &cellIndices) const
    {
        return cellIndices[0] * _nCellsY * _nCellsZ + cellIndices[1] * _nCellsZ + cellIndices[2];
    }
};

#endif // _CELL_LIST_H_