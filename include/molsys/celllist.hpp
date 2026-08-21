/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _CELL_LIST_HPP_

#define _CELL_LIST_HPP_

#include <cstddef>   // for size_t
#include <vector>    // for vector

#include "cell.hpp"       // for Cell
#include "defaults.hpp"   // for _NUMBER_OF_CELLS_DEFAULT_, _CELL_LIST_IS_ACT...
#include "vector3d.hpp"

namespace molsys
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
        bool _activated = defaults::CELL_LIST_IS_ACTIVE_DEFAULT;

        std::vector<Cell> _cells;

        linearAlgebra::Vec3D   _cellSize;
        linearAlgebra::Vec3Dul _nNeighbourCells{0, 0, 0};
        linearAlgebra::Vec3Dul _nCells{defaults::NUMBER_OF_CELLS_DEFAULT};

       public:
        [[nodiscard]] std::shared_ptr<CellList> clone() const;

        void setup(const SimulationBox &);
        void updateCellList(SimulationBox &);

        void determineCellSize(const linearAlgebra::Vec3D &box);
        void determineCellBoundaries(const linearAlgebra::Vec3D &box);
        void checkCoulombCutoff(const double coulombCutoff) const;

        void addNeighbouringCells(const double coulombCutoff);
        void addNeighbouringCellPointers(Cell &);
        void addMoleculesToCells(SimulationBox &simulationBox);
        void assignMoleculeHybridZoneIndices();
        void assignWaterMoleculeIndices(SimulationBox &);

        [[nodiscard]] size_t getCellIndex(
            const linearAlgebra::Vec3Dul &cellIndices
        ) const;
        [[nodiscard]] linearAlgebra::Vec3Dul getCellIndexOfAtom(
            const linearAlgebra::Vec3D &,
            const linearAlgebra::Vec3D &
        ) const;

        void resizeCells();
        void addCell(const Cell &cell);

        /*****************************
         * standard activate methods *
         *****************************/

        void               activate();
        void               deactivate();
        [[nodiscard]] bool isActive() const;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] linearAlgebra::Vec3Dul getNumberOfCells() const;
        [[nodiscard]] linearAlgebra::Vec3Dul getNumberOfNeighbourCells() const;
        [[nodiscard]] linearAlgebra::Vec3D   getCellSize() const;
        [[nodiscard]] const std::vector<Cell> &getCells() const;
        [[nodiscard]] Cell                    &getCell(const size_t index);

        /***************************
         * standard setter methods *
         ***************************/

        void setNumberOfCells(const size_t nCells);
        void setNumberOfNeighbourCells(const size_t nCells);
    };

}   // namespace molsys

#endif   // _CELL_LIST_HPP_
