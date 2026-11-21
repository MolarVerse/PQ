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

#ifndef LINEAR_ALGEBRA__STATIC_MATRIX__STATIC_MATRIX3X3_HPP
#define LINEAR_ALGEBRA__STATIC_MATRIX__STATIC_MATRIX3X3_HPP

#include <concepts>
#include <mstd/type_traits/quantity_traits.hpp>
#include <ostream>

#include "staticMatrix3x3Class.hpp"

namespace linalg
{

    template <typename T>
    std::ostream &operator<<(
        std::ostream             &ostream,
        const StaticMatrix3x3<T> &mat
    );

    /****************************
     * operator+ and operator+= *
     ****************************/

    template <typename T>
    StaticMatrix3x3<T> operator+(
        const StaticMatrix3x3<T> &lhs,
        const StaticMatrix3x3<T> &rhs
    );

    template <typename T>
    StaticMatrix3x3<T> operator+(const StaticMatrix3x3<T> &lhs, const T &rhs);

    template <typename T>
    void operator+=(StaticMatrix3x3<T> &lhs, const StaticMatrix3x3<T> &rhs);

    // mixed element types, e.g. quantity<kcal/mol> + quantity<energy/...>
    template <typename T, typename U>
    requires(!std::same_as<T, U>) &&
            requires(const T &lhs, const U &rhs) { lhs + rhs; }
    constexpr auto operator+(
        const StaticMatrix3x3<T> &lhs,
        const StaticMatrix3x3<U> &rhs
    ) -> StaticMatrix3x3<std::remove_cvref_t<decltype(lhs[0][0] + rhs[0][0])>>;

    // the result keeps the element type of lhs
    template <typename T, typename U>
    requires(!std::same_as<T, U>) &&
            requires(T &lhs, const U &rhs) { lhs += rhs; }
    constexpr void operator+=(
        StaticMatrix3x3<T>       &lhs,
        const StaticMatrix3x3<U> &rhs
    );

    /****************************
     * operator- and operator-= *
     ****************************/

    template <typename T>
    StaticMatrix3x3<T> operator-(
        const StaticMatrix3x3<T> &lhs,
        const StaticMatrix3x3<T> &rhs
    );

    // mixed element types, e.g. quantity<bar> - quantity<Pa>
    template <typename T, typename U>
    requires(!std::same_as<T, U>) &&
            requires(const T &lhs, const U &rhs) { lhs - rhs; }
    constexpr auto operator-(
        const StaticMatrix3x3<T> &lhs,
        const StaticMatrix3x3<U> &rhs
    ) -> StaticMatrix3x3<std::remove_cvref_t<decltype(lhs[0][0] - rhs[0][0])>>;

    template <typename T>
    void operator-=(StaticMatrix3x3<T> &lhs, const StaticMatrix3x3<T> &rhs);

    // the result keeps the element type of lhs
    template <typename T, typename U>
    requires(!std::same_as<T, U>) &&
            requires(T &lhs, const U &rhs) { lhs -= rhs; }
    constexpr void operator-=(
        StaticMatrix3x3<T>       &lhs,
        const StaticMatrix3x3<U> &rhs
    );

    /****************************
     * operator* and operator*= *
     ****************************/

    template <typename T, typename U>
    requires requires(const T &lhsElem, const U &rhsElem) {
        lhsElem * rhsElem;
        lhsElem + lhsElem;
    }
    constexpr auto operator*(
        const StaticMatrix3x3<T> &lhs,
        const StaticMatrix3x3<U> &rhs
    ) -> StaticMatrix3x3<std::remove_cvref_t<decltype(lhs[0][0] * rhs[0][0])>>;

    template <typename T>
    StaticMatrix3x3<T> operator*(
        const StaticMatrix3x3<T> &mat,
        const T                  &scalar
    );

    template <typename T>
    StaticMatrix3x3<T> operator*(
        const T                  &scalar,
        const StaticMatrix3x3<T> &mat
    );

    template <typename T>
    Vector3D<T> operator*(
        const StaticMatrix3x3<T> &mat,
        const Vector3D<T>        &vec
    );

    // mixed element types, e.g. StaticMatrix3x3<quantity<1/Å>> * Vector3D<Å>
    template <typename T, typename U>
    requires(!std::same_as<T, U>) &&
            requires(const T &lhsElem, const U &rhsElem) {
                lhsElem * rhsElem;
                lhsElem * rhsElem + lhsElem * rhsElem;
            }
    constexpr auto operator*(
        const StaticMatrix3x3<T> &mat,
        const Vector3D<U>        &vec
    ) -> Vector3D<std::remove_cvref_t<decltype(mat[0][0] * vec[0])>>;

    // unit/quantity scalar, e.g. tensor * mstd::units::Pa
    template <typename T, typename V>
    requires(!std::same_as<T, V>) && mstd::NonArithmeticScalar<T, V>
    constexpr auto operator*(const StaticMatrix3x3<T> &mat, const V &scalar)
        -> StaticMatrix3x3<std::remove_cvref_t<decltype(mat[0][0] * scalar)>>;

    template <typename T, typename V>
    requires(!std::same_as<T, V>) && mstd::NonArithmeticScalar<T, V>
    constexpr auto operator*(const V &scalar, const StaticMatrix3x3<T> &mat)
        -> StaticMatrix3x3<std::remove_cvref_t<decltype(mat[0][0] * scalar)>>;

    // plain number times a matrix of quantities, e.g. 2.0 * virial
    template <typename T, typename V>
    requires(!std::same_as<T, V>) && std::is_arithmetic_v<V> &&
            requires(const T &elem, const V &val) { elem * val; }
    constexpr auto operator*(const StaticMatrix3x3<T> &mat, const V &scalar)
        -> StaticMatrix3x3<std::remove_cvref_t<decltype(mat[0][0] * scalar)>>;

    template <typename T, typename V>
    requires(!std::same_as<T, V>) && std::is_arithmetic_v<V> &&
            requires(const T &elem, const V &val) { elem * val; }
    constexpr auto operator*(const V &scalar, const StaticMatrix3x3<T> &mat)
        -> StaticMatrix3x3<std::remove_cvref_t<decltype(mat[0][0] * scalar)>>;

    template <typename T, typename U>
    requires requires(T &elem, const U &val) { elem *= val; }
    void operator*=(StaticMatrix3x3<T> &lhs, const U &scalar);

    /****************************
     * operator/ and operator/= *
     ****************************/

    template <typename T, typename U>
    requires std::convertible_to<U, T>
    StaticMatrix3x3<T> operator/(
        const StaticMatrix3x3<T> &mat,
        const U                  &scalar
    );

    template <typename T, typename U>
    requires std::convertible_to<U, T>
    StaticMatrix3x3<T> operator/(
        const U                  &scalar,
        const StaticMatrix3x3<T> &mat
    );

    // element type not constructible from the scalar, e.g.
    // StaticMatrix3x3<quantity<one>> / size_t
    template <typename T, typename U>
    requires(!std::convertible_to<U, T>) &&
            requires(const T &elem, const U &scalar) { elem / scalar; }
    constexpr auto operator/(const StaticMatrix3x3<T> &mat, const U &scalar)
        -> StaticMatrix3x3<std::remove_cvref_t<decltype(mat[0][0] / scalar)>>;

    template <typename T, typename U>
    requires requires(T &elem, const U &val) { elem /= val; }
    void operator/=(StaticMatrix3x3<T> &lhs, const U &scalar);

    /****************************
     * general matrix functions *
     ****************************/

    template <typename T>
    auto isFinite(const StaticMatrix3x3<T> &mat) -> bool;

    template <typename T>
    StaticMatrix3x3<T> transpose(const StaticMatrix3x3<T> &mat);

    // for quantities the determinant carries the product unit, e.g.
    // length^3 for a matrix of lengths
    template <typename T>
    auto det(const StaticMatrix3x3<T> &mat)
        -> std::remove_cvref_t<decltype(mat[0][0] * mat[0][0] * mat[0][0])>;

    // lhs and rhs may differ (e.g. momentum x velocity); the element type is
    // the product type
    template <typename T, typename U>
    auto tensorProduct(const Vector3D<T> &lhs, const Vector3D<U> &rhs)
        -> StaticMatrix3x3<std::remove_cvref_t<decltype(lhs[0] * rhs[0])>>;

    // lhs is a quantity holding a Vector3D (e.g. `vec * mass`); the result is
    // the plain tensor product of its numerical values
    template <typename Q, typename U>
    requires(
        mstd::is_quantity_like_v<Q> &&
        std::same_as<typename Q::rep, Vector3D<double>>
    )
    auto tensorProduct(const Q &lhs, const Vector3D<U> &rhs)
        -> StaticMatrix3x3<double>;

    template <typename T>
    auto cofactorMatrix(
        const StaticMatrix3x3<T> &mat
    ) -> StaticMatrix3x3<std::remove_cvref_t<decltype(mat[0][0] * mat[0][0])>>;

    // for quantities the inverse carries the inverse unit, e.g. 1/length for
    // a matrix of lengths
    template <typename T>
    auto inverse(const StaticMatrix3x3<T> &mat)
        -> StaticMatrix3x3<std::remove_cvref_t<
            decltype(mat[0][0] * mat[0][0] / (mat[0][0] * mat[0][0] * mat[0][0]))>>;

    template <typename T>
    Vector3D<T> diagonal(const StaticMatrix3x3<T> &mat);

    template <typename T>
    StaticMatrix3x3<T> diagonalMatrix(const Vector3D<T> &vec);

    template <typename T>
    StaticMatrix3x3<T> diagonalMatrix(const T &scalar);

    template <typename T>
    T trace(const StaticMatrix3x3<T> &mat);

    template <typename T>
    [[nodiscard]] StaticMatrix3x3<T> kroneckerDeltaMatrix();

    template <typename T>
    [[nodiscard]] StaticMatrix3x3<T> exp(const StaticMatrix3x3<T> &mat);

    template <typename T>
    [[nodiscard]] StaticMatrix3x3<T> expPade(const StaticMatrix3x3<T> &mat);

}   // namespace linalg

#ifndef LINEAR_ALGEBRA__STATIC_MATRIX__STATIC_MATRIX3X3_TPP
#include "staticMatrix3x3.tpp"   // IWYU pragma: export
#endif

#endif   // LINEAR_ALGEBRA__STATIC_MATRIX__STATIC_MATRIX3X3_HPP
