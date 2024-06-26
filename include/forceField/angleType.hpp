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

#ifndef _ANGLE_TYPE_HPP_

#define _ANGLE_TYPE_HPP_

#include <cstddef>

namespace forceField
{
    /**
     * @class AngleType
     *
     * @brief represents an angle type
     *
     * @details this is a class representing an angle type defined in the
     * parameter file
     *
     */
    class AngleType
    {
       private:
        size_t _id;

        double _equilibriumAngle;
        double _forceConstant;

       public:
        AngleType(const size_t, const double, const double);

        friend bool operator==(const AngleType &, const AngleType &);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getId() const;
        [[nodiscard]] double getEquilibriumAngle() const;
        [[nodiscard]] double getForceConstant() const;
    };

}   // namespace forceField

#endif   // _ANGLE_TYPE_HPP_