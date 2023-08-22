#include "coulombShiftedPotential.hpp"

#include "constants.hpp"

using namespace potential;

/**
 * @brief calculate the energy and force of the shifted Coulomb potential
 *
 * @param distance
 * @return std::pair<double, double>
 */
[[nodiscard]] std::pair<double, double> CoulombShiftedPotential::calculate(const double distance,
                                                                           const double chargeProduct) const
{
    const auto coulombPrefactor = chargeProduct * constants::_COULOMB_PREFACTOR_;

    auto energy = (1 / distance) - _coulombEnergyCutOff - _coulombForceCutOff * (_coulombRadiusCutOff - distance);
    auto force  = (1 / (distance * distance)) - _coulombForceCutOff;

    energy *= coulombPrefactor;
    force  *= coulombPrefactor;

    return {energy, force};
}