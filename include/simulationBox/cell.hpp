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

#ifndef _CELL_HPP_

#define _CELL_HPP_

#include <cstddef>   // for size_t
#include <vector>    // for vector

#include "typeAliases.hpp"

namespace simulationBox
{
    /**
     * @class Cell
     *
     * @brief Cell is a class for a single cell in the cellList
     *
     */
    class Cell
    {
       private:
        std::vector<pq::Molecule *>      _molecules;
        std::vector<std::vector<size_t>> _atomIndices;
        std::vector<Cell *>              _neighbourCells;

        pq::Vec3D   _lowerBoundary = {0, 0, 0};
        pq::Vec3D   _upperBoundary = {0, 0, 0};
        pq::Vec3Dul _cellIndex     = {0, 0, 0};

       public:
        void clearMolecules();
        void clearAtomIndices();

        void addMolecule(pq::Molecule &molecule);
        void addMolecule(pq::Molecule *molecule);
        void addNeighbourCell(Cell *cell);
        void addAtomIndices(const std::vector<size_t> &atomIndices);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t             getNumberOfMolecules() const;
        [[nodiscard]] size_t             getNumberOfNeighbourCells() const;
        [[nodiscard]] const pq::Vec3D   &getLowerBoundary() const;
        [[nodiscard]] const pq::Vec3D   &getUpperBoundary() const;
        [[nodiscard]] const pq::Vec3Dul &getCellIndex() const;

        [[nodiscard]] pq::Molecule *getMolecule(const size_t index) const;
        [[nodiscard]] std::vector<Molecule *> getMolecules() const;

        [[nodiscard]] Cell *getNeighbourCell(const size_t index) const;
        [[nodiscard]] std::vector<Cell *> getNeighbourCells() const;

        [[nodiscard]] const std::vector<size_t> &getAtomIndices(
            const size_t index
        ) const;

        /***************************
         * standard setter methods *
         ***************************/

        void setLowerBoundary(const pq::Vec3D &lowerBoundary);
        void setUpperBoundary(const pq::Vec3D &upperBoundary);
        void setCellIndex(const pq::Vec3Dul &cellIndex);
    };

}   // namespace simulationBox

#endif   // _CELL_HPP_