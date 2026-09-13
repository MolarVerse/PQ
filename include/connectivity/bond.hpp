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

#include "connectivityElement.hpp"

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

        Bond(molsys::Molecule *, molsys::Molecule *, AtomIndex, AtomIndex);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] molsys::Molecule *getMolecule1() const;
        [[nodiscard]] molsys::Molecule *getMolecule2() const;
    };

}   // namespace connectivity

#endif   // _BOND_HPP_
