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

/**
 * @brief specializing of template function compare with tolerance
 *
 * @param a
 * @param b
 * @param tolerance
 * @return true
 * @return false
 */
bool utilities::compare(const linearAlgebra::Vec3D &a, const linearAlgebra::Vec3D &b, const double &tolerance)
{
    return compare<double>(a[0], b[0], tolerance) && compare<double>(a[1], b[1], tolerance) &&
           compare<double>(a[2], b[2], tolerance);
}

/**
 * @brief specializing of template function compare
 *
 * @param a
 * @param b
 * @return true
 * @return false
 */
bool utilities::compare(const linearAlgebra::Vec3D &a, const linearAlgebra::Vec3D &b)
{
    return compare<double>(a[0], b[0]) && compare<double>(a[1], b[1]) && compare<double>(a[2], b[2]);
}