#include "bondType.hpp"

#include "mathUtilities.hpp"

using namespace forceField;

/**
 * @brief operator overload for the comparison of two BondType objects
 *
 * @param other
 * @return true
 * @return false
 */
bool BondType::operator==(const BondType &other) const
{
    auto isEqual = _id == other._id;
    isEqual      = isEqual && utilities::compare(_equilibriumBondLength, other._equilibriumBondLength);
    isEqual      = isEqual && utilities::compare(_forceConstant, other._forceConstant);

    return isEqual;
}