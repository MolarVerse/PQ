#include "lennardJonesPair.hpp"

#include <iostream>

using namespace potential;

/**
 * @brief operator overload for the comparison of two LennardJonesPair objects
 *
 * @param other
 * @return true
 * @return false
 */
bool LennardJonesPair::operator==(const LennardJonesPair &other) const
{
    return NonCoulombPair::operator==(other) && utilities::compare(_c6, other._c6) && utilities::compare(_c12, other._c12);
}

/**
 * @brief calculates the energy and force of a LennardJonesPair
 *
 * @param distance
 * @return std::pair<double, double>
 */
std::pair<double, double> LennardJonesPair::calculateEnergyAndForce(const double distance) const
{
    const auto distanceSquared = distance * distance;
    const auto distanceSixth   = distanceSquared * distanceSquared * distanceSquared;
    const auto distanceTwelfth = distanceSixth * distanceSixth;

    const auto energy = _c12 / distanceTwelfth + _c6 / distanceSixth - _energyCutOff - _forceCutOff * (_radialCutOff - distance);
    const auto force  = 12.0 * _c12 / (distanceTwelfth * distance) + 6.0 * _c6 / (distanceSixth * distance) - _forceCutOff;

    return {energy, force};
}