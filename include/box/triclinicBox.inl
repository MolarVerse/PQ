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

#ifndef __TRICLINIC_BOX_INL__
#define __TRICLINIC_BOX_INL__

/**
 * @file triclinicBox.inl
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief This file contains the implementation of the free inline functions for
 * the triclinic box. The functions are used to image the triclinic box.
 *
 * @date 2024-12-09
 *
 * @see triclinicBox.hpp
 * @see box.hpp
 */

#include "typeAliases.hpp"

namespace simulationBox
{

    // TODO: this is not correct yet see - applyPBC of triclinicBox.cpp

    /**
     * @brief image triclinic
     *
     * @param boxDimensions
     * @param x
     * @param y
     * @param z
     * @param tx
     * @param ty
     * @param tz
     */
    static inline void imageTriclinic(
        const Real* const boxParams,
        Real&             x,
        Real&             y,
        Real&             z,
        Real&             tx,
        Real&             ty,
        Real&             tz
    )
    {
        const auto unitBoxX =
            ::round(boxParams[9] * x + boxParams[10] * y + boxParams[11] * z);
        const auto unitBoxY =
            ::round(boxParams[12] * x + boxParams[13] * y + boxParams[14] * z);
        const auto unitBoxZ =
            ::round(boxParams[15] * x + boxParams[16] * y + boxParams[17] * z);

        tx = boxParams[0] * unitBoxX + boxParams[1] * unitBoxY +
             boxParams[2] * unitBoxZ;
        ty = boxParams[3] * unitBoxX + boxParams[4] * unitBoxY +
             boxParams[5] * unitBoxZ;
        tz = boxParams[6] * unitBoxX + boxParams[7] * unitBoxY +
             boxParams[8] * unitBoxZ;

        x += tx;
        y += ty;
        z += tz;
    }

    /**
     * @brief image triclinic
     *
     * @param boxParams
     * @param x
     * @param y
     * @param z
     */
    static inline void imageTriclinic(
        const Real* const boxParams,
        Real&             x,
        Real&             y,
        Real&             z
    )
    {
        const auto unitBoxX =
            ::round(boxParams[9] * x + boxParams[10] * y + boxParams[11] * z);
        const auto unitBoxY =
            ::round(boxParams[12] * x + boxParams[13] * y + boxParams[14] * z);
        const auto unitBoxZ =
            ::round(boxParams[15] * x + boxParams[16] * y + boxParams[17] * z);

        x -= boxParams[0] * unitBoxX + boxParams[1] * unitBoxY +
             boxParams[2] * unitBoxZ;
        y -= boxParams[3] * unitBoxX + boxParams[4] * unitBoxY +
             boxParams[5] * unitBoxZ;
        z -= boxParams[6] * unitBoxX + boxParams[7] * unitBoxY +
             boxParams[8] * unitBoxZ;
    }

}   // namespace simulationBox

#endif   // __TRICLINIC_BOX_INL__