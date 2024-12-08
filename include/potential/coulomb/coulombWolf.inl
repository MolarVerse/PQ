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
