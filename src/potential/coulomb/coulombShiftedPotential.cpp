#include "coulombShiftedPotential.hpp"

using namespace potential;

[[nodiscard]] std::pair<double, double> CoulombShiftedPotential::calculateEnergyAndForce(const double distance) const
{
    auto energy = (1 / distance) - _coulombEnergyCutOff - _coulombForceCutOff * (_coulombRadiusCutOff - distance);
    auto force  = (1 / (distance * distance)) - _coulombForceCutOff;

    energy *= _coulombPreFactor;
    force  *= _coulombPreFactor;

    return {energy, force};
}