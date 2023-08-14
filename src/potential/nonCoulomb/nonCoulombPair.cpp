#include "nonCoulombPair.hpp"

using namespace potential;

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

    const auto energy = _c12 / distanceTwelfth + _c6 / distanceSixth - _energyCutOff - _forceCutOff * (_radialCutOff - distance);
    const auto force  = 12.0 * _c12 / (distanceTwelfth * distance) + 6.0 * _c6 / (distanceSixth * distance) - _forceCutOff;

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

std::pair<double, double> GuffPair::calculateEnergyAndForce(const double distance) const
{
    const double c1 = _coefficients[0];
    const double n2 = _coefficients[1];
    const double c3 = _coefficients[2];
    const double n4 = _coefficients[3];

    const double distance_n2 = ::pow(distance, n2);
    const double distance_n4 = ::pow(distance, n4);

    auto energy = c1 / distance_n2 + c3 / distance_n4;
    auto force  = n2 * c1 / (distance_n2 * distance) + n4 * c3 / (distance_n4 * distance);

    const double c5 = _coefficients[4];
    const double n6 = _coefficients[5];
    const double c7 = _coefficients[6];
    const double n8 = _coefficients[7];

    const double distance_n6 = ::pow(distance, n6);
    const double distance_n8 = ::pow(distance, n8);

    energy += c5 / distance_n6 + c7 / distance_n8;
    force  += n6 * c5 / (distance_n6 * distance) + n8 * c7 / (distance_n8 * distance);

    const double c9     = _coefficients[8];
    const double cexp10 = _coefficients[9];
    const double rExp11 = _coefficients[10];

    double helper = ::exp(cexp10 * (distance - rExp11));

    energy += c9 / (1 + helper);
    force  += c9 * cexp10 * helper / ((1 + helper) * (1 + helper));

    const double c12    = _coefficients[11];
    const double cexp13 = _coefficients[12];
    const double rExp14 = _coefficients[13];

    helper = ::exp(cexp13 * (distance - rExp14));

    energy += c12 / (1 + helper);
    force  += c12 * cexp13 * helper / ((1 + helper) * (1 + helper));

    const double c15    = _coefficients[14];
    const double cexp16 = _coefficients[15];
    const double rExp17 = _coefficients[16];
    const double n18    = _coefficients[17];

    const double distance_n18 = ::pow(distance - rExp17, n18);
    helper                    = c15 * ::exp(cexp16 * distance_n18);

    energy += helper;
    force  += -cexp16 * n18 * distance_n18 / (distance - rExp17) * helper;

    const double c19    = _coefficients[18];
    const double cexp20 = _coefficients[19];
    const double rExp21 = _coefficients[20];
    const double n22    = _coefficients[21];

    const double distance_n22 = ::pow(distance - rExp21, n22);
    helper                    = c19 * ::exp(cexp20 * distance_n22);

    energy += helper;
    force  += -cexp20 * n22 * distance_n22 / (distance - rExp21) * helper;

    energy += -_energyCutOff - _forceCutOff * (_radialCutOff - distance);
    force  += -_forceCutOff;

    return {energy, force};
}