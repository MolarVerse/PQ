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

#ifndef LINEAR_ALGEBRA__VECTOR3D__VECTOR3D_DETAIL_HPP
#define LINEAR_ALGEBRA__VECTOR3D__VECTOR3D_DETAIL_HPP

#include <cstddef>
#include <mstd/type_traits.hpp>
#include <type_traits>
#include <utility>

#include "concepts/vector3dConcepts.hpp"
#include "innerType.hpp"

namespace linalg
{
    template <typename T>
    class Vector3D;

    /***********
     *         *
     * helpers *
     *         *
     ***********/

    namespace detail
    {
        /**
         * @brief true if `T` has to be indexed (`T[i]`) when combined
         * element-wise with `Other`
         *
         * @details An operand is indexed if it is a Vector3D that is at least
         * as deep as the other operand. The shallower operand (a scalar or a
         * less nested Vector3D) is broadcast to every component instead.
         *
         * @tparam T
         * @tparam Other
         */
        template <typename T, typename Other>
        inline constexpr bool isIndexed_v =
            pq::isVector3D_v<T> && pq::Vector3DDepthDifference_v<T, Other> >= 0;

        /**
         * @brief component `i` of `vec`, or `vec` itself if it is broadcast
         *
         * @tparam Other the operand `vec` is combined with
         * @tparam T
         * @param vec
         * @param index
         * @return `vec[i]` if `isIndexed_v<T, Other>`, otherwise `vec`
         */
        template <typename Other, typename T>
        constexpr decltype(auto) component(
            const T          &vec,
            const std::size_t index
        )
        {
            if constexpr (isIndexed_v<T, Other>)
                return (vec[index]);
            else
                return (vec);
        }

        /**
         * @brief element type of the result of `Op` applied to `L` and `R`
         *
         * @tparam Op
         * @tparam L
         * @tparam R
         */
        template <typename Op, typename L, typename R>
        using BinaryElement_t = std::remove_cvref_t<decltype(std::declval<Op>()(
            component<R>(std::declval<const L &>(), 0),
            component<L>(std::declval<const R &>(), 0)
        ))>;

        /**
         * @brief result type of `Op` applied component-wise to `L` and `R`
         *
         * @tparam Op
         * @tparam L
         * @tparam R
         */
        template <typename Op, typename L, typename R>
        using BinaryResult_t = linalg::Vector3D<BinaryElement_t<Op, L, R>>;

        /**
         * @brief apply `Op` component-wise to two operands
         *
         * @details Covers all combinations of Vector3D (any depth) and scalar
         * operands, e.g. `Vector3D<int> + Vector3D<double>`,
         * `Vector3D<Vector3D<int>> * Vector3D<double>` or `double - Vector3D`.
         * At least one operand has to be a Vector3D.
         *
         * @tparam Op binary function object, e.g. `std::plus<>`
         * @tparam L
         * @tparam R
         * @param lhs
         * @param rhs
         * @return BinaryResult_t<Op, L, R>
         */
        template <typename Op, typename L, typename R>
        constexpr auto binaryOp(const L &lhs, const R &rhs)
            -> BinaryResult_t<Op, L, R>
        {
            constexpr Op func{};

            return BinaryResult_t<Op, L, R>(
                func(component<R>(lhs, 0), component<L>(rhs, 0)),
                func(component<R>(lhs, 1), component<L>(rhs, 1)),
                func(component<R>(lhs, 2), component<L>(rhs, 2))
            );
        }

        /**
         * @brief result type of `Op` applied component-wise to `U`
         *
         * @tparam Op
         * @tparam U
         */
        template <typename Op, typename U>
        using UnaryResult_t = linalg::Vector3D<std::remove_cvref_t<
            decltype(std::declval<Op>()(std::declval<const U &>()[0]))>>;

        /**
         * @brief apply `Op` component-wise to a Vector3D
         *
         * @tparam Op unary function object, e.g. `std::negate<>`
         * @tparam U
         * @param vec
         * @return UnaryResult_t<Op, U>
         */
        template <typename Op, typename U>
        constexpr auto unaryOp(const U &vec) -> UnaryResult_t<Op, U>
        {
            constexpr Op func{};

            return UnaryResult_t<Op, U>(
                func(vec[0]),
                func(vec[1]),
                func(vec[2])
            );
        }

        /**
         * @brief Vector3D of arithmetic or quantity-like values
         */
        template <typename T>
        concept NumericVector3D =
            pq::ArithmeticVector3D<T> || pq::QuantityLikeVector3D<T>;

        /**
         * @brief both operands are Vector3D with a depth difference of at
         * most one, e.g. `Vector3D<int> + Vector3D<Vector3D<double>>`
         */
        template <typename U, typename V>
        concept VectorVectorOperands =
            NumericVector3D<U> && NumericVector3D<V> &&
            (pq::Vector3DDepthDifference_v<U, V> >= -1) &&
            (pq::Vector3DDepthDifference_v<U, V> <= 1);

        /**
         * @brief a Vector3D combined with an arithmetic scalar (either order)
         */
        template <typename U, typename V>
        concept VectorScalarOperands =
            (NumericVector3D<U> && pq::Arithmetic<V>) ||
            (pq::Arithmetic<U> && NumericVector3D<V>);

        /**
         * @brief operands of the binary operators `+`, `-` and `/`
         */
        template <typename U, typename V>
        concept BinaryOperands =
            VectorVectorOperands<U, V> || VectorScalarOperands<U, V>;

        /**
         * @brief a Vector3D (lhs) multiplied by a quantity-like scalar (rhs)
         */
        template <typename U, typename V>
        concept QuantityScalarOperands =
            NumericVector3D<U> && !pq::isVector3D_v<V> &&
            mstd::NonArithmeticScalar<pq::InnerType_t<U>, V>;

        /**
         * @brief operands of the binary operator `*`
         */
        template <typename U, typename V>
        concept MultiplyOperands =
            BinaryOperands<U, V> || QuantityScalarOperands<U, V>;

    }   // namespace detail

}   // namespace linalg

#endif   // LINEAR_ALGEBRA__VECTOR3D__VECTOR3D_DETAIL_HPP
