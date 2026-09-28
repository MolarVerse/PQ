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

#include <stdexcept>
#include <utility>

/**
 * @brief checks if the given isotropy is semi-isotropic
 *
 * @param isotropy the isotropy to check
 * @return true if the isotropy is semi-isotropic, false otherwise
 */
bool isSemiIsotropic(Isotropy isotropy)
{
    return isotropy == Isotropy::SEMI_ISOTROPIC_XY ||
           isotropy == Isotropy::SEMI_ISOTROPIC_XZ ||
           isotropy == Isotropy::SEMI_ISOTROPIC_YZ;
}

/**
 * @brief gets the anisotropic axis for a given semi-isotropic isotropy
 *
 * @param isotropy the isotropy to check
 * @return the index of the anisotropic axis (0 for X, 1 for Y, 2 for Z)
 */
size_t get2DAnisotropicAxis(Isotropy isotropy)
{
    switch (isotropy)
    {
        case Isotropy::SEMI_ISOTROPIC_XY: return 2;   // Z is anisotropic
        case Isotropy::SEMI_ISOTROPIC_XZ: return 1;   // Y is anisotropic
        case Isotropy::SEMI_ISOTROPIC_YZ: return 0;   // X is anisotropic
        case Isotropy::ISOTROPIC:
        case Isotropy::ANISOTROPIC:
        case Isotropy::FULL_ANISOTROPIC:
            throw std::runtime_error("Not a semi-isotropic isotropy");
    }

    std::unreachable();
}

/**
 * @brief gets the isotropic axes for a given semi-isotropic isotropy
 *
 * @param isotropy the isotropy to check
 * @return an array containing the indices of the isotropic axes (0 for X, 1 for
 * Y, 2 for Z)
 */
std::array<size_t, 2> get2DIsotropicAxes(Isotropy isotropy)
{
    switch (isotropy)
    {
        case Isotropy::SEMI_ISOTROPIC_XY: return {0, 1};
        case Isotropy::SEMI_ISOTROPIC_XZ: return {0, 2};
        case Isotropy::SEMI_ISOTROPIC_YZ: return {1, 2};
        case Isotropy::ISOTROPIC:
        case Isotropy::ANISOTROPIC:
        case Isotropy::FULL_ANISOTROPIC:
            throw std::runtime_error("Not a semi-isotropic isotropy");
    }

    std::unreachable();
}

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
