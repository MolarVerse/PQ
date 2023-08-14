#ifndef _COULOMB_SHIFTED_POTENTIAL_HPP_

#define _COULOMB_SHIFTED_POTENTIAL_HPP_

#include "coulombPotential.hpp"

namespace potential
{
    class CoulombShiftedPotential;
}   // namespace potential

/**
 * @class CoulombShiftedPotential
 *
 * @brief
 * CoulombShiftedPotential inherits CoulombPotential
 * CoulombShiftedPotential is a class for the shifted Coulomb potential
 *
 */
class potential::CoulombShiftedPotential : public potential::CoulombPotential
{
  public:
    using CoulombPotential::CoulombPotential;

    [[nodiscard]] std::pair<double, double> calculateEnergyAndForce(const double) const override;
};

#endif   // _COULOMB_SHIFTED_POTENTIAL_HPP_