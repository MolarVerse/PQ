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

#ifndef _COULOMB_SHIFTED_POTENTIAL_INL_
#define _COULOMB_SHIFTED_POTENTIAL_INL_

/**
 * @file coulombShiftedPotential.inl
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief This file contains the implementation of the free inline functions for
 * the shifted Coulomb potential. The functions are used to calculate the energy
 * and force of the shifted Coulomb potential.
 *
 * @date 2024-12-09
 *
 * @see coulombShiftedPotential.hpp
 *
 */

#include "constants.hpp"
#include "coulombShiftedPotential.hpp"

namespace potential
{
    /**
     * @brief calculate the energy and force of the shifted Coulomb potential
     *
     * @param force
     * @param r
     * @param chargeProduct
     * @param cutOff
     * @param params
     * @return Real
     */
    static inline Real calculateCoulombShiftedPotential(
        Real&             force,
        const Real        r,
        const Real        chargeProduct,
        const Real        cutOff,
        const Real* const params
    )
    {
        const auto prefactor = chargeProduct * constants::_COULOMB_PREFACTOR_;
        const auto dInv      = 1 / r;

        const auto forceCutOff = params[1];

        auto energy     = dInv - params[0] - forceCutOff * (cutOff - r);
        auto localForce = dInv * dInv - forceCutOff;

        energy     *= prefactor;
        localForce *= prefactor;

        force += localForce;

        return energy;
    }
}   // namespace potential

#endif   // _COULOMB_SHIFTED_POTENTIAL_INL_