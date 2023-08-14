#include "coulombPotential.hpp"

using namespace potential;

CoulombPotential::CoulombPotential(const double coulombRadiusCutOff)
{
    _coulombRadiusCutOff = coulombRadiusCutOff;
    _coulombEnergyCutOff = 1 / _coulombRadiusCutOff;
    _coulombForceCutOff  = 1 / (_coulombRadiusCutOff * _coulombRadiusCutOff);
}

[[nodiscard]] std::pair<double, double> CoulombPotential::calculate(const std::vector<size_t> &indices,
                                                                    const double               distance,
                                                                    const double preFactor)   // note cannot be const
{
    _setCoulombPreFactor(indices, preFactor);
    return calculateEnergyAndForce(distance);
}