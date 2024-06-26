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

#ifndef _DIHEDRAL_TYPE_HPP_

#define _DIHEDRAL_TYPE_HPP_

#include <cstddef>

namespace forceField
{
    class DihedralType;   // forward declaration

    bool operator==(const DihedralType &, const DihedralType &);

    /**
     * @class DihedralType
     *
     * @brief represents a dihedral type
     *
     * @details this is a class representing a dihedral type defined in the
     * parameter file
     *
     */
    class DihedralType
    {
       private:
        size_t _id;

        double _forceConstant;
        double _periodicity;
        double _phaseShift;

       public:
        DihedralType(
            const size_t id,
            const double forceConstant,
            const double frequency,
            const double phaseShift
        );

        friend bool operator==(const DihedralType &, const DihedralType &);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getId() const;
        [[nodiscard]] double getForceConstant() const;
        [[nodiscard]] double getPeriodicity() const;
        [[nodiscard]] double getPhaseShift() const;
    };

}   // namespace forceField

#endif   // _DIHEDRAL_TYPE_HPP_