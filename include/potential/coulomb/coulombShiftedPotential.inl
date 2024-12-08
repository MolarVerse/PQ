#ifndef _COULOMB_SHIFTED_POTENTIAL_INL_
#define _COULOMB_SHIFTED_POTENTIAL_INL_

#include "constants.hpp"
#include "coulombShiftedPotential.hpp"

namespace potential
{
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