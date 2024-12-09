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

#ifndef __COULOMB_WOLF_INL__
#define __COULOMB_WOLF_INL__

#include <cmath>   // for exp, sqrt, erfc

#include "constants.hpp"   // for _COULOMB_PREFACTOR_
#include "coulombWolf.hpp"

#ifndef M_PI
#define M_PI std::numbers::pi
#endif

namespace potential
{
    static inline Real calculateCoulombWolfPotential(
        Real&             force,
        const Real        r,
        const Real        chargeProduct,
        const Real        cutOff,
        const Real* const params
    )
    {
        const auto prefactor = chargeProduct * constants::_COULOMB_PREFACTOR_;

        const auto kappa = params[0];
        const auto wolf1 = params[1];
        const auto wolf2 = params[2];
        const auto wolf3 = params[3];

        const auto kappaR         = kappa * r;
        const auto erfcFactorInvR = ::erfc(kappaR) / r;
        const auto expFactor      = ::exp(-kappaR * kappaR);

        auto energy     = erfcFactorInvR - wolf1 + wolf3 * (r - cutOff);
        auto localForce = erfcFactorInvR / r + wolf2 * expFactor / r - wolf3;

        energy     *= prefactor;
        localForce *= prefactor;

        force += localForce;
        return energy;
    }

}   // namespace potential

#endif   // __COULOMB_WOLF_INL__
