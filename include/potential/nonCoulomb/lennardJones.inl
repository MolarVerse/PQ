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

#ifndef __LENNARD_JONES_INL__
#define __LENNARD_JONES_INL__

#include "lennardJones.hpp"

namespace potential
{
    /**
     * @brief Calculate the Lennard-Jones potential
     *
     * @param r
     * @param r2
     * @param cutOff
     * @param params
     * @return Real
     */
    static inline Real calculateLennardJones(
        Real&             force,
        const Real        r,
        const Real        r2,
        const Real        cutOff,
        const Real* const params
    )
    {
        const Real c6        = params[0];
        const Real c12       = params[1];
        const Real energyCut = params[2];
        const Real forceCut  = params[3];
        const Real r2Inv     = 1.0 / r2;
        const Real r6        = r2Inv * r2Inv * r2Inv;
        const Real r12       = r6 * r6;
        const Real cr12      = c12 * r12;
        const Real cr6       = c6 * r6;
        const Real energy    = cr12 + cr6 - energyCut - forceCut * (cutOff - r);

        force += (12.0 * cr12 + 6.0 * cr6) / r - forceCut;

        return energy;
    }
}   // namespace potential

#endif   // __LENNARD_JONES_INL__