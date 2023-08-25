#ifndef _CELL_LIST_HPP_

#define _CELL_LIST_HPP_

#include "cell.hpp"       // for Cell
#include "defaults.hpp"   // for _NUMBER_OF_CELLS_DEFAULT_, _CELL_LIST_IS_ACT...
#include "vector3d.hpp"   // for Vec3Dul, Vec3D

#include <algorithm>   // for max
#include <cstddef>     // for size_t
#include <vector>      // for vector

namespace simulationBox
{
    class SimulationBox;   // forward declaration

    /**
     * @class CellList
     *
     * @brief CellList is a class for cell list
     *
     */
    class CellList
    {
      private:
        bool _activated = defaults::_CELL_LIST_IS_ACTIVE_DEFAULT_;

        std::vector<Cell> _cells;

        linearAlgebra::Vec3D   _cellSize;
        linearAlgebra::Vec3Dul _nNeighbourCells = {0, 0, 0};
        linearAlgebra::Vec3Dul _nCells          = {
            defaults::_NUMBER_OF_CELLS_DEFAULT_, defaults::_NUMBER_OF_CELLS_DEFAULT_, defaults::_NUMBER_OF_CELLS_DEFAULT_};

      public:
        void setup(const SimulationBox &);
        void updateCellList(SimulationBox &);

        void determineCellSize(const SimulationBox &);
        void determineCellBoundaries(const SimulationBox &);

        void addNeighbouringCells(const SimulationBox &);
        void addCellPointers(Cell &);

        size_t                 getCellIndex(const linearAlgebra::Vec3Dul &cellIndices) const;
        linearAlgebra::Vec3Dul getCellIndexOfMolecule(const SimulationBox &, const linearAlgebra::Vec3D &) const;

        void               activate() { _activated = true; }
        void               deactivate() { _activated = false; }
        [[nodiscard]] bool isActivated() const { return _activated; }

        void addCell(const Cell &cell) { _cells.push_back(cell); }
        void resizeCells(const size_t nCells) { _cells.resize(nCells); }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] linearAlgebra::Vec3Dul getNumberOfCells() const { return _nCells; }
        [[nodiscard]] linearAlgebra::Vec3Dul getNumberOfNeighbourCells() const { return _nNeighbourCells; }
        [[nodiscard]] linearAlgebra::Vec3D   getCellSize() const { return _cellSize; }
        [[nodiscard]] std::vector<Cell>      getCells() const { return _cells; }
        [[nodiscard]] Cell                  &getCell(const size_t index) { return _cells[index]; }

        /***************************
         * standard setter methods *
         ***************************/

        void setNumberOfCells(const size_t nCells) { _nCells = {nCells, nCells, nCells}; }
        void setNumberOfNeighbourCells(const size_t nNeighbourCells)
        {
            _nNeighbourCells = {nNeighbourCells, nNeighbourCells, nNeighbourCells};
        }
    };

}   // namespace simulationBox

#endif   // _CELL_LIST_HPP_