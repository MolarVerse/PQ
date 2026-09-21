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

namespace linalg
{
    template <typename T>
    class Vector3D;   // forward declaration

}   // namespace linalg

namespace utilities
{
    /**
     * @brief compares two numbers with a tolerance
     *
     * @tparam T
     * @param lhs
     * @param rhs
     * @param tolerance
     * @return true
     * @return false
     */
    template <typename T>
    [[nodiscard]]
    bool compare(const T &lhs, const T &rhs, const T &tolerance)
    {
        return std::abs(lhs - rhs) < tolerance;
    }

    [[nodiscard]]
    bool compare(
        const linalg::Vector3D<double> &lhs,
        const linalg::Vector3D<double> &rhs,
        const double                   &tolerance
    );

    /**
     * @brief compares two numbers via machine precision
     *
     * @tparam T
     * @param lhs
     * @param rhs
     * @return true
     * @return false
     */
    template <typename T>
    [[nodiscard]]
    bool compare(const T &lhs, const T &rhs)
    {
        return std::fabs(lhs - rhs) < std::numeric_limits<T>::epsilon();
    }

    [[nodiscard]]
    bool compare(
        const linalg::Vector3D<double> &lhs,
        const linalg::Vector3D<double> &rhs
    );

    /**
     * @brief check whether a number is exactly zero
     *
     * @details Uses exact equality (`a == T(0)`) rather than an epsilon
     * comparison: callers that guard against division-by-zero or `0 * Inf`
     * NaN propagation only need to catch literal zero. Use the explicit
     * `compare(a, T(0), tol)` overload when a tolerance is wanted.
     *
     * @tparam T
     * @param value
     * @return true if value == T(0), false otherwise
     */
    template <typename T>
    [[nodiscard]] bool isZero(const T &value)
    {
        return value == T(0);
    }

    /**
     * @brief calculates the sign of a number
     *
     * @tparam T
     * @param value
     * @return int
     */
    template <typename T>
    [[nodiscard]] int sign(const T &value)
    {
        if (compare(value, T(0)))
            return 0;

        if (value > T(0))
            return 1;

        return -1;
    }

    [[nodiscard]] size_t kroneckerDelta(size_t lhs, size_t rhs);

}   // namespace utilities

#endif   // _MATH_UTILITIES_HPP_
