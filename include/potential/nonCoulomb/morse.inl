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

#ifndef __MORSE_INL__
#define __MORSE_INL__

#include "morse.hpp"

namespace potential
{
    /**
     * @brief Calculate the Morse potential
     *
     * @param r
     * @param cutOff
     * @param params
     * @return Real
     */
    static inline Real calculateMorse(
        Real&             force,
        const Real        r,
        const Real        cutOff,
        const Real* const params
    )
    {
        const Real dissociationEnergy = params[0];
        const Real wellWidth          = params[1];
        const Real term               = std::exp(wellWidth * (params[2] - r));
        const Real term2              = 1.0 - term;
        const Real term3              = dissociationEnergy * term2;
        const Real forceCut           = params[4];

        const Real energy = term3 * term2 - params[3] - forceCut * (cutOff - r);

        force += -2.0 * term3 * term * wellWidth - forceCut;

        return energy;
    }
}   // namespace potential

#endif   // __MORSE_INL__