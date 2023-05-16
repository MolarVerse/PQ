#ifndef _CELL_LIST_H_

#define _CELL_LIST_H_

#include <vector>

#include "cell.hpp"

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
    int _nCellsX = 1;
    int _nCellsY = 1;
    int _nCellsZ = 1;

public:
    void setCells(Cell &cell, int index_x, int index_y, int index_z);
    Cell &getCell(int index) { return _cells[index]; }
    std::vector<Cell> getCells() const { return _cells; }
};

#endif // _CELL_LIST_H_