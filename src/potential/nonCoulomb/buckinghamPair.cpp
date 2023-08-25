#include "buckinghamPair.hpp"

#include "mathUtilities.hpp"   // for compare

#include <cmath>   // for exp

using namespace potential;

/**
 * @brief operator overload for the comparison of two BuckinghamPair objects
 *
 * @param other
 * @return true
 * @return false
 */
bool BuckinghamPair::operator==(const BuckinghamPair &other) const
{
    return NonCoulombPair::operator==(other) && utilities::compare(_a, other._a) && utilities::compare(_dRho, other._dRho) &&
           utilities::compare(_c6, other._c6);
}

/**
 * @brief calculates the energy and force of a BuckinghamPair
 *
 * @param distance
 * @return std::pair<double, double>
 */
std::pair<double, double> BuckinghamPair::calculateEnergyAndForce(const double distance) const
{
    const auto distanceSixth = distance * distance * distance * distance * distance * distance;
    const auto expTerm       = std::exp(_dRho * distance);
    const auto energy        = _a * expTerm + _c6 / distanceSixth - _energyCutOff - _forceCutOff * (_radialCutOff - distance);
    const auto force         = -_a * _dRho * expTerm + 6.0 * _c6 / (distanceSixth * distance) - _forceCutOff;

    return {energy, force};
}