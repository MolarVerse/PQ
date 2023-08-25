#ifndef _COULOMB_SHIFTED_POTENTIAL_HPP_

#define _COULOMB_SHIFTED_POTENTIAL_HPP_

#include "coulombPotential.hpp"

#include <utility>   // for pair

namespace potential
{
    /**
     * @class CoulombShiftedPotential
     *
     * @brief
     * CoulombShiftedPotential inherits CoulombPotential
     * CoulombShiftedPotential is a class for the shifted Coulomb potential
     *
     */
    class CoulombShiftedPotential : public potential::CoulombPotential
    {
      public:
        using CoulombPotential::CoulombPotential;

        [[nodiscard]] std::pair<double, double> calculate(const double, const double) const override;
    };

}   // namespace potential

#endif   // _COULOMB_SHIFTED_POTENTIAL_HPP_