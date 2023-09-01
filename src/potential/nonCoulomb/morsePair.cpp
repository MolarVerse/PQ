#include "morsePair.hpp"

#include "mathUtilities.hpp"   // for compare

#include <cmath>   // for exp

using namespace potential;

/**
 * @brief operator overload for the comparison of two MorsePair objects
 *
 * @param other
 * @return true
 * @return false
 */
bool MorsePair::operator==(const MorsePair &other) const
{
    return NonCoulombPair::operator==(other) && utilities::compare(_dissociationEnergy, other._dissociationEnergy) &&
           utilities::compare(_wellWidth, other._wellWidth) &&
           utilities::compare(_equilibriumDistance, other._equilibriumDistance);
}

/**
 * @brief calculates the energy and force of a MorsePair
 *
 * @param distance
 * @return std::pair<double, double>
 */
std::pair<double, double> MorsePair::calculateEnergyAndForce(const double distance) const
{
    const auto expTerm = std::exp(-_wellWidth * (distance - _equilibriumDistance));

    const auto energy =
        _dissociationEnergy * (1.0 - expTerm) * (1.0 - expTerm) - _energyCutOff - _forceCutOff * (_radialCutOff - distance);
    const auto force = -2.0 * _dissociationEnergy * _wellWidth * expTerm * (1.0 - expTerm) - _forceCutOff;

    return {energy, force};
}