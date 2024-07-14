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
#include "typeAliases.hpp"

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

        Bond(pq::Molecule *const, pq::Molecule *const, size_t, size_t);
        Bond(pq::Molecule *const, size_t, pq::Molecule *const, size_t);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] pq::Molecule *getMolecule1() const;
        [[nodiscard]] pq::Molecule *getMolecule2() const;
        [[nodiscard]] size_t        getAtomIndex1() const;
        [[nodiscard]] size_t        getAtomIndex2() const;
    };

}   // namespace connectivity

#endif   // _BOND_HPP_