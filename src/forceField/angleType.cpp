#include "angleType.hpp"

#include "mathUtilities.hpp"

using namespace forceField;

/**
 * @brief operator overload for the comparison of two AngleType objects
 *
 * @param other
 * @return true
 * @return false
 */
bool AngleType::operator==(const AngleType &other) const
{
    auto isEqual = _id == other._id;
    isEqual      = isEqual && utilities::compare(_equilibriumAngle, other._equilibriumAngle);
    isEqual      = isEqual && utilities::compare(_forceConstant, other._forceConstant);

    return isEqual;
}