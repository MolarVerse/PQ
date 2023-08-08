#include "nonCoulombicPair.hpp"

using namespace forceField;

/**
 * @brief operator overload for the comparison of two NonCoulombicPair objects
 *
 * @details returns also true if the two types are switched
 *
 * @param other
 * @return true
 * @return false
 */
bool NonCoulombicPair::operator==(const NonCoulombicPair &other) const
{
    auto isEqual = _vanDerWaalsType1 == other._vanDerWaalsType1;
    isEqual      = isEqual && _vanDerWaalsType2 == other._vanDerWaalsType2;
    isEqual      = isEqual && utilities::compare(_cutOff, other._cutOff);

    auto isEqualSymmetric = _vanDerWaalsType1 == other._vanDerWaalsType2;
    isEqualSymmetric      = isEqualSymmetric && _vanDerWaalsType2 == other._vanDerWaalsType1;
    isEqualSymmetric      = isEqualSymmetric && utilities::compare(_cutOff, other._cutOff);

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
    return NonCoulombicPair::operator==(other) && utilities::compare(_c6, other._c6) && utilities::compare(_c12, other._c12);
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
    return NonCoulombicPair::operator==(other) && utilities::compare(_a, other._a) && utilities::compare(_dRho, other._dRho) &&
           utilities::compare(_c6, other._c6);
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
    return NonCoulombicPair::operator==(other) && utilities::compare(_dissociationEnergy, other._dissociationEnergy) &&
           utilities::compare(_wellWidth, other._wellWidth) &&
           utilities::compare(_equilibriumDistance, other._equilibriumDistance);
}