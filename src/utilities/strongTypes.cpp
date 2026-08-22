#include "strongTypes.hpp"

#include "mathUtilities.hpp"

/**
 * @brief operator overload for the comparison of two LJParams objects
 *
 * @param other
 * @return true
 * @return false
 */
bool LJParams::operator==(const LJParams &other) const
{
    return utilities::compare(c6, other.c6) &&
           utilities::compare(c12, other.c12);
}

/**
 * @brief compare two MorseParams objects for equality
 *
 * @param other
 * @return true
 * @return false
 */
bool MorseParams::operator==(const MorseParams &other) const
{
    return utilities::compare(dissociationEnergy, other.dissociationEnergy) &&
           utilities::compare(wellWidth, other.wellWidth) &&
           utilities::compare(equilibriumDistance, other.equilibriumDistance);
}

/**
 * @brief compare two BuckinghamParams objects for equality
 *
 * @param other
 * @return true
 * @return false
 */
bool BuckinghamParams::operator==(const BuckinghamParams &other) const
{
    return utilities::compare(scaling, other.scaling) &&
           utilities::compare(dRho, other.dRho) &&
           utilities::compare(c6, other.c6);
}
