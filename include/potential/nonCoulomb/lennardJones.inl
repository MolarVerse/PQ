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