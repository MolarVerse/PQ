#include "enums/manostat.hpp"

/**
 * @brief checks if a specific axis is fixed in the given FixedAxis bitmask
 *
 * @param fixedAxis the FixedAxis bitmask
 * @param axisIndex the index of the axis to check (0 for X, 1 for Y, 2 for Z)
 * @return true if the axis is fixed, false otherwise
 */
bool isAxisFixed(FixedAxis fixedAxis, size_t axisIndex)
{
    const auto axisToCheck = static_cast<FixedAxis>(1U << axisIndex);
    return (fixedAxis & axisToCheck) == axisToCheck;
}
