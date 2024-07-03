#ifndef _POTENTIAL_TPP_

#define _POTENTIAL_TPP_

#include "potential.hpp"

namespace potential
{
    /**
     * @brief make shared pointer of the Coulomb potential
     *
     * @tparam T
     * @param p
     */
    template <typename T>
    void Potential::makeCoulombPotential(T p)
    {
        _coulombPotential = std::make_shared<T>(p);
    }

    /**
     * @brief make shared pointer of the non-Coulomb potential
     *
     * @tparam T
     * @param nonCoulombPotential
     */
    template <typename T>
    void Potential::makeNonCoulombPotential(T nonCoulombPotential)
    {
        _nonCoulombPotential = std::make_shared<T>(nonCoulombPotential);
    }

}   // namespace potential

#endif   // _POTENTIAL_TPP_