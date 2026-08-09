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

#include "molecule.hpp"   // for Molecule
#include "simulationBox.hpp"

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
        std::vector<Molecule *>          _molecules;
        std::vector<std::vector<Atom *>> _atoms;
        std::vector<Cell *>              _neighbourCells;
        std::vector<size_t>              _coreMoleculeIndices;
        std::vector<size_t>              _smoothingMoleculeIndices;
        std::vector<size_t>              _nonSmoothingMoleculeIndices;
        std::vector<size_t>              _activeMoleculeIndices;
        std::vector<size_t>              _inactiveNonCoreMoleculeIndices;
        std::vector<size_t>              _waterMoleculeIndices;

        linearAlgebra::Vec3D   _lowerBoundary = {0, 0, 0};
        linearAlgebra::Vec3D   _upperBoundary = {0, 0, 0};
        linearAlgebra::Vec3Dul _cellIndex     = {0, 0, 0};

       public:
        void clearMolecules();
        void clearAtoms();

        void addMolecule(Molecule &molecule);
        void addMolecule(Molecule *molecule);
        void addNeighbourCell(Cell *cell);
        void addAtoms(const std::vector<Atom *> &atomPointers);
        void assignMoleculeHybridZoneIndices();
        void assignWaterMoleculeIndices(const simulationBox::SimulationBox &);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getNumberOfMolecules() const;
        [[nodiscard]] size_t getNumberOfNeighbourCells() const;
        [[nodiscard]] const linearAlgebra::Vec3D   &getLowerBoundary() const;
        [[nodiscard]] const linearAlgebra::Vec3D   &getUpperBoundary() const;
        [[nodiscard]] const linearAlgebra::Vec3Dul &getCellIndex() const;

        [[nodiscard]] Molecule *getMolecule(const size_t index) const
        {
            return _molecules[index];
        }
        [[nodiscard]] const std::vector<Molecule *> &getMolecules() const;
        [[nodiscard]] std::vector<Molecule *>       &getMolecules();

        [[nodiscard]] Cell *getNeighbourCell(const size_t index) const;
        [[nodiscard]] const std::vector<Cell *> &getNeighbourCells() const;

        [[nodiscard]] const std::vector<Atom *> &getAtoms(
            const size_t molIndex
        ) const
        {
            return _atoms[molIndex];
        }

        [[nodiscard]] const std::vector<size_t> &getCoreMoleculeIndices() const;
        [[nodiscard]] const std::vector<size_t> &getSmoothingMoleculeIndices(
        ) const;
        [[nodiscard]] const std::vector<size_t> &getNonSmoothingMoleculeIndices(
        ) const;
        [[nodiscard]] const std::vector<size_t> &getActiveMoleculeIndices(
        ) const;
        [[nodiscard]] const std::vector<size_t> &getInactiveNonCoreMoleculeIndices(
        ) const;
        [[nodiscard]] const std::vector<size_t> &getWaterMoleculeIndices(
        ) const;

        /***************************
         * standard setter methods *
         ***************************/

        void setLowerBoundary(const linearAlgebra::Vec3D &lowerBoundary);
        void setUpperBoundary(const linearAlgebra::Vec3D &upperBoundary);
        void setCellIndex(const linearAlgebra::Vec3Dul &cellIndex);
    };

}   // namespace simulationBox

#endif   // _CELL_HPP_
