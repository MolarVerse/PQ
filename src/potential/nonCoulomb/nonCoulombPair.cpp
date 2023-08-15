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