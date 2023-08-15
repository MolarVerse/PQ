#include "coulombWolf.hpp"

#include <cmath>

using namespace potential;

/**
 * @brief Construct a new Coulomb Wolf:: Coulomb Wolf object
 *
 * @details this constructor calculates automatically the three need wolf parameters from kappa in order to gain speed
 *
 * @param coulombRadiusCutOff
 * @param kappa
 */
CoulombWolf::CoulombWolf(const double coulombRadiusCutOff, const double kappa) : CoulombPotential(coulombRadiusCutOff)
{
    _kappa          = kappa;
    _wolfParameter1 = ::erfc(_kappa * coulombRadiusCutOff) / coulombRadiusCutOff;
    _wolfParameter2 = 2.0 * _kappa / ::sqrt(M_PI);
    _wolfParameter3 = _wolfParameter1 / coulombRadiusCutOff +
                      _wolfParameter2 * ::exp(-_kappa * _kappa * coulombRadiusCutOff * coulombRadiusCutOff) / coulombRadiusCutOff;
}

/**
 * @brief calculate the energy and force of the Coulomb potential with Wolf summation as long range correction
 *
 * @param distance
 * @return std::pair<double, double>
 */
[[nodiscard]] std::pair<double, double> CoulombWolf::calculateEnergyAndForce(const double distance) const
{

    const auto kappaDistance = _kappa * distance;
    const auto erfcFactor    = ::erfc(kappaDistance);

    auto energy = erfcFactor / distance - _wolfParameter1 + _wolfParameter3 * (distance - _coulombRadiusCutOff);
    auto force =
        erfcFactor / (distance * distance) + _wolfParameter2 * ::exp(-kappaDistance * kappaDistance) / distance - _wolfParameter3;

    energy *= _coulombPreFactor;
    force  *= _coulombPreFactor;

    return {energy, force};
}