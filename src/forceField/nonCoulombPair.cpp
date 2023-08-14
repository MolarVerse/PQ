#include "nonCoulombPair.hpp"

using namespace forceField;

/**
 * @brief operator overload for the comparison of two NonCoulombPair objects
 *
 * @details returns also true if the two types are switched
 *
 * @param other
 * @return true
 * @return false
 */
bool NonCoulombPair::operator==(const NonCoulombPair &other) const
{
    auto isEqual = _vanDerWaalsType1 == other._vanDerWaalsType1;
    isEqual      = isEqual && _vanDerWaalsType2 == other._vanDerWaalsType2;
    isEqual      = isEqual && utilities::compare(_radialCutOff, other._radialCutOff);

    auto isEqualSymmetric = _vanDerWaalsType1 == other._vanDerWaalsType2;
    isEqualSymmetric      = isEqualSymmetric && _vanDerWaalsType2 == other._vanDerWaalsType1;
    isEqualSymmetric      = isEqualSymmetric && utilities::compare(_radialCutOff, other._radialCutOff);

    return isEqual || isEqualSymmetric;
}

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

    const auto energy = _c12 / distanceTwelfth - _c6 / distanceSixth - _energyCutOff - _forceCutOff * (_radialCutOff - distance);
    const auto force  = 12.0 * _c12 / (distanceTwelfth * distance) - 6.0 * _c6 / (distanceSixth * distance) - _forceCutOff;

    return {energy, force};
}

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
    const auto energy        = _a * expTerm - _c6 / distanceSixth - _energyCutOff - _forceCutOff * (_radialCutOff - distance);
    const auto force         = -_a * _dRho * expTerm + 6.0 * _c6 / (distanceSixth * distance) - _forceCutOff;

    return {energy, force};
}

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
    const auto force = -2.0 * _dissociationEnergy * _wellWidth * expTerm * (1.0 - expTerm) / distance - _forceCutOff;

    return {energy, force};
}