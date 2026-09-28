/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

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
