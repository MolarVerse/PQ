#include "coulombShiftedPotential.hpp"

using namespace potential;

/**
 * @brief calculate the energy and force of the shifted Coulomb potential
 *
 * @param distance
 * @return std::pair<double, double>
 */
[[nodiscard]] std::pair<double, double> CoulombShiftedPotential::calculateEnergyAndForce(const double distance) const
{
    auto energy = (1 / distance) - _coulombEnergyCutOff - _coulombForceCutOff * (_coulombRadiusCutOff - distance);
    auto force  = (1 / (distance * distance)) - _coulombForceCutOff;

    energy *= _coulombPreFactor;
    force  *= _coulombPreFactor;

    return {energy, force};
}