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

#ifndef UTILITIES__MATH_UTILITIES_HPP
#define UTILITIES__MATH_UTILITIES_HPP

#include <concepts>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mstd/math.hpp>
#include <mstd/type_traits/quantity_traits.hpp>

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
        return mstd::fabs(lhs - rhs) < tolerance;
    }

    [[nodiscard]] bool compare(
        const linalg::Vector3D<double> &lhs,
        const linalg::Vector3D<double> &rhs,
        const double                   &tol
    );

    /**
     * @brief compares two quantities of the same dimension but possibly
     * different unit types, with a tolerance
     *
     * @details `lhs` is converted to the type of `rhs`, which defines the
     * unit of the comparison. The tolerance may be a quantity (converted to
     * the type of `rhs`) or a plain number, which is then interpreted in the
     * unit of `rhs`.
     */
    template <mstd::QuantityLike L, mstd::QuantityLike R, typename Tol>
    [[nodiscard]]
    bool compare(const L &lhs, const R &rhs, const Tol &tolerance)
    {
        const R tol = [&]
        {
            if constexpr (mstd::QuantityLike<Tol>)
                return R(tolerance);
            else
                return R(tolerance * R::reference);
        }();

        return mstd::fabs(R(lhs) - rhs) < tol;
    }

    /**
     * @brief compares two quantities of the same dimension but different
     * unit types via machine precision (in the unit of `rhs`)
     */
    template <mstd::QuantityLike L, mstd::QuantityLike R>
    requires(!std::same_as<L, R>)
    [[nodiscard]]
    bool compare(const L &lhs, const R &rhs)
    {
        return mstd::fabs(R(lhs) - rhs) < std::numeric_limits<R>::epsilon();
    }

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
        return mstd::fabs(lhs - rhs) < std::numeric_limits<T>::epsilon();
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
        if constexpr (requires { value.zero(); })
        {
            return value == value.zero();
        }
        else
        {
            return value == T(0);
        }
    }

    /**
     * @brief calculates the sign of a number or quantity
     *
     * @details Works for arithmetic types and for mp-units quantities (zero
     * is obtained via `T::zero()` where available, as `T(0)` is not valid for
     * dimensionful quantities).
     *
     * @tparam T
     * @param value
     * @return int
     */
    template <typename T>
    [[nodiscard]] int sign(const T &value)
    {
        const auto zero = [] {
            if constexpr (requires { T::zero(); })
                return T::zero();
            else
                return T(0);
        }();

        if (compare(value, zero))
            return 0;

        if (value > zero)
            return 1;

        return -1;
    }

    [[nodiscard]] size_t kroneckerDelta(size_t lhs, size_t rhs);

}   // namespace utilities

#endif   // UTILITIES__MATH_UTILITIES_HPP
