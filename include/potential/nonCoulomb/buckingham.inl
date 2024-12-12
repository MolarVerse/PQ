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

#ifndef __BUCKINGHAM_INL__
#define __BUCKINGHAM_INL__

#include "buckingham.hpp"

namespace potential
{
    /**
     * @brief Calculate the Buckingham potential
     *
     * @param r
     * @param r2
     * @param cutOff
     * @param params
     * @return Real
     */
    static inline Real calculateBuckingham(
        Real&             force,
        const Real        r,
        const Real        r2,
        const Real        cutOff,
        const Real* const params
    )
    {
        const Real dRho     = params[1];
        const Real forceCut = params[4];
        const Real r2Inv    = 1.0 / r2;
        const Real cr6      = params[2] * r2Inv * r2Inv * r2Inv;
        const Real expTerm  = params[0] * ::exp(dRho * r);
        const Real energy = expTerm + cr6 - params[3] - forceCut * (cutOff - r);

        force += -dRho * expTerm + 6.0 * cr6 / r - forceCut;

        return energy;
    }
}   // namespace potential

#endif   // __BUCKINGHAM_INL__