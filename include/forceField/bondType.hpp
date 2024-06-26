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

#ifndef _BOND_TYPE_HPP_

#define _BOND_TYPE_HPP_

#include <cstddef>

namespace forceField
{
    class BondType;   // forward declaration

    bool operator==(const BondType &, const BondType &);

    /**
     * @class BondType
     *
     * @brief represents a bond type
     *
     * @details this is a class representing a bond type defined in the
     * parameter file
     *
     */
    class BondType
    {
       private:
        size_t _id;

        double _equilBondLength;
        double _forceConstant;

       public:
        BondType(const size_t, const double, const double);

        friend bool operator==(const BondType &, const BondType &);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getId() const;
        [[nodiscard]] double getEquilibriumBondLength() const;
        [[nodiscard]] double getForceConstant() const;
    };

}   // namespace forceField

#endif   // _BOND_TYPE_HPP_