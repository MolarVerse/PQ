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

#ifndef _BOND_HPP_

#define _BOND_HPP_

#include <cstddef>
#include <vector>

#include "connectivityElement.hpp"

namespace simulationBox
{
    class Molecule;   // forward declaration
}

namespace connectivity
{
    /**
     * @class Bond
     *
     * @brief Represents a bond between two atoms.
     *
     */
    class Bond : public ConnectivityElement
    {
       public:
        using ConnectivityElement::ConnectivityElement;
        Bond(
            simulationBox::Molecule *molecule1,
            simulationBox::Molecule *molecule2,
            size_t                   atomIndex1,
            size_t                   atomIndex2
        )
            : ConnectivityElement(
                  {molecule1, molecule2},
                  {atomIndex1, atomIndex2}
              ){};

        /***************************
         *                         *
         * standard getter methods *
         *                         *
         ***************************/

        [[nodiscard]] simulationBox::Molecule *getMolecule1() const
        {
            return _molecules[0];
        }
        [[nodiscard]] simulationBox::Molecule *getMolecule2() const
        {
            return _molecules[1];
        }
        [[nodiscard]] size_t getAtomIndex1() const { return _atomIndices[0]; }
        [[nodiscard]] size_t getAtomIndex2() const { return _atomIndices[1]; }
    };

}   // namespace connectivity

#endif   // _BOND_HPP_