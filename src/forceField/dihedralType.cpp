#include "dihedralType.hpp"

#include "mathUtilities.hpp"

using namespace forceField;

/**
 * @brief operator overload for the comparison of two DihedralType objects
 *
 * @param other
 * @return true
 * @return false
 */
bool DihedralType::operator==(const DihedralType &other) const
{
    auto isEqual = _id == other._id;
    isEqual      = isEqual && utilities::compare(_forceConstant, other._forceConstant);
    isEqual      = isEqual && utilities::compare(_periodicity, other._periodicity);
    isEqual      = isEqual && utilities::compare(_phaseShift, other._phaseShift);

    return isEqual;
}