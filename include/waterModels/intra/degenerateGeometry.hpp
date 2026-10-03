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

#ifndef _DEGENERATE_GEOMETRY_HPP_

#define _DEGENERATE_GEOMETRY_HPP_

#include <cmath>
#include <format>
#include <string_view>

#include "exceptions.hpp"

namespace waterModel
{
    /**
     * @brief Throw if a length of an intramolecular water model is zero or not
     * finite.
     *
     * @details The intramolecular water forces divide by the O-H and H-H
     * distances and by the sine of the H-O-H angle. If one of them is zero
     * (two atoms on top of each other, both hydrogens on the same ray from
     * the oxygen) the forces and energies silently become NaN. An angle of
     * exactly pi is not degenerate: its sine does not vanish in floating
     * point and the cross product is exactly zero, so the result is finite.
     *
     * @param value the length to check (positive for a valid geometry)
     * @param what description of the quantity, used in the message
     *
     * @throws exc::WaterModelException if value is zero, negative, NaN or
     * infinite
     */
    inline void checkNonDegenerate(
        const double           value,
        const std::string_view what
    )
    {
        if (!(value > 0.0) || !std::isfinite(value))
        {
            throw exc::WaterModelException(
                std::format(
                    "Degenerate water geometry - the {} is zero or not "
                    "finite ({}), so the intramolecular water forces are "
                    "undefined. Check the starting structure or whether "
                    "the simulation has become unstable",
                    what,
                    value
                )
            );
        }
    }
}   // namespace waterModel

#endif   // _DEGENERATE_GEOMETRY_HPP_
