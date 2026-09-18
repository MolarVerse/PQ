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

#include "mathUtilities.hpp"

#include "vector3d.hpp"

/**
 * @brief specializing of template function compare with tolerance
 *
 * @param lhs
 * @param rhs
 * @param tolerance
 * @return true
 * @return false
 */
bool utilities::compare(
    const linearAlgebra::Vector3D<double> &lhs,
    const linearAlgebra::Vector3D<double> &rhs,
    const double                          &tolerance
)
{
    auto isEq = true;
    isEq      = isEq && compare<double>(lhs[0], rhs[0], tolerance);
    isEq      = isEq && compare<double>(lhs[1], rhs[1], tolerance);
    isEq      = isEq && compare<double>(lhs[2], rhs[2], tolerance);

    return isEq;
}

/**
 * @brief specializing of template function compare for Vector3D<double>
 *
 * @param lhs
 * @param rhs
 * @return true
 * @return false
 */
bool utilities::compare(
    const linearAlgebra::Vector3D<double> &lhs,
    const linearAlgebra::Vector3D<double> &rhs
)
{
    auto isEq = true;
    isEq      = isEq && compare<double>(lhs[0], rhs[0]);
    isEq      = isEq && compare<double>(lhs[1], rhs[1]);
    isEq      = isEq && compare<double>(lhs[2], rhs[2]);

    return isEq;
}

/**
 * @brief Kronecker delta function
 *
 * @param lhs
 * @param rhs
 * @return size_t
 */
size_t utilities::kroneckerDelta(size_t lhs, size_t rhs)
{
    return lhs == rhs ? 1 : 0;
}
