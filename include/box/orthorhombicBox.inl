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

#ifndef __ORTHORHOMBIC_BOX_INL__
#define __ORTHORHOMBIC_BOX_INL__

/**
 * @file orthorhombicBox.inl
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief This file contains the implementation of the free inline functions for
 * the orthorhombic box. The functions are used to image the orthorhombic box.
 *
 * @date 2024-12-09
 *
 * @see orthorhombicBox.hpp
 * @see box.hpp
 */

#include "orthorhombicBox.hpp"   // IWYU pragma: keep

namespace simulationBox
{

    /**
     * @brief image orthorhombic
     *
     * @param boxDimensions
     * @param x
     * @param y
     * @param z
     * @param tx
     * @param ty
     * @param tz
     */
    static inline void imageOrthoRhombic(
        const Real* const boxDimensions,
        Real&             x,
        Real&             y,
        Real&             z,
        Real&             tx,
        Real&             ty,
        Real&             tz
    )
    {
        const auto boxX = boxDimensions[0];
        const auto boxY = boxDimensions[1];
        const auto boxZ = boxDimensions[2];

        tx = -boxX * ::round(x / boxX);
        ty = -boxY * ::round(y / boxY);
        tz = -boxZ * ::round(z / boxZ);

        x += tx;
        y += ty;
        z += tz;
    }

}   // namespace simulationBox

#endif   // __ORTHORHOMBIC_BOX_INL__