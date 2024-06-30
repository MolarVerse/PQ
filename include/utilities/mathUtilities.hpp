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

#ifndef _MATH_UTILITIES_HPP_

#define _MATH_UTILITIES_HPP_

#include <cmath>     // for fabs
#include <cstdlib>   // for abs
#include <limits>    // for numeric_limits

#include "vector3d.hpp"

namespace utilities
{
    /**
     * @brief compares two numbers with a tolerance
     *
     * @tparam T
     * @param a
     * @param b
     * @param tolerance
     * @return true
     * @return false
     */
    template <typename T>
    bool compare(const T &a, const T &b, const T &tolerance)
    {
        return std::abs(a - b) < tolerance;
    }

    bool compare(
        const linearAlgebra::Vec3D &a,
        const linearAlgebra::Vec3D &b,
        const double               &tolerance
    );

    /**
     * @brief compares two numbers via machine precision
     *
     * @tparam T
     * @param a
     * @param b
     * @return true
     * @return false
     */
    template <typename T>
    bool compare(const T &a, const T &b)
    {
        return std::fabs(a - b) < std::numeric_limits<T>::epsilon();
    }

    bool compare(const linearAlgebra::Vec3D &a, const linearAlgebra::Vec3D &b);

    /**
     * @brief calculates the sign of a number
     *
     * @tparam T
     * @param a
     * @return int
     */
    template <typename T>
    int sign(const T &a)
    {
        if (compare(a, T(0)))
            return 0;
        else if (a > T(0))
            return 1;
        else
            return -1;
    }

    size_t kroneckerDelta(const size_t i, const size_t j);

}   // namespace utilities

#endif   // _MATH_UTILITIES_HPP_