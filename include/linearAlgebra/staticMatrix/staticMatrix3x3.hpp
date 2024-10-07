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

#ifndef _STATIC_MATRIX_3X3_HPP_

#define _STATIC_MATRIX_3X3_HPP_

#include <cstddef>   // for size_t
#include <ostream>   // for operator<<, ostream

#include "staticMatrix3x3Class.hpp"   // for StaticMatrix3x3
#include "vector3d.hpp"               // for Vector3D

namespace linearAlgebra
{

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const StaticMatrix3x3<T> &mat);

    /****************************
     * operator+ and operator+= *
     ****************************/

    template <typename T>
    StaticMatrix3x3<T> operator+(const StaticMatrix3x3<T> &, const StaticMatrix3x3<T> &);

    template <typename T>
    StaticMatrix3x3<T> operator+(const StaticMatrix3x3<T> &lhs, const T &rhs);

    template <typename T>
    void operator+=(StaticMatrix3x3<T> &lhs, const StaticMatrix3x3<T> &rhs);

    /****************************
     * operator- and operator-= *
     ****************************/

    template <typename T>
    StaticMatrix3x3<T> operator-(const StaticMatrix3x3<T> &, const StaticMatrix3x3<T> &);

    template <typename T>
    void operator-=(StaticMatrix3x3<T> &lhs, const StaticMatrix3x3<T> &rhs);

    /****************************
     * operator* and operator*= *
     ****************************/

    template <typename T>
    StaticMatrix3x3<T> operator*(const StaticMatrix3x3<T> &, const StaticMatrix3x3<T> &);

    template <typename T>
    StaticMatrix3x3<T> operator*(const StaticMatrix3x3<T> &mat, const T t);

    template <typename T>
    StaticMatrix3x3<T> operator*(const T t, const StaticMatrix3x3<T> &mat);

    template <typename T>
    Vector3D<T> operator*(const StaticMatrix3x3<T> &, const Vector3D<T> &);

    template <typename T>
    void operator*=(StaticMatrix3x3<T> &lhs, const T t);

    /****************************
     * operator/ and operator/= *
     ****************************/

    template <typename T>
    StaticMatrix3x3<T> operator/(const StaticMatrix3x3<T> &mat, const T t);

    template <typename T>
    StaticMatrix3x3<T> operator/(const T t, const StaticMatrix3x3<T> &mat);

    template <typename T>
    void operator/=(StaticMatrix3x3<T> &lhs, const T t);

    /****************************
     * general matrix functions *
     ****************************/

    template <typename T>
    StaticMatrix3x3<T> transpose(const StaticMatrix3x3<T> &mat);

    template <typename T>
    T det(const StaticMatrix3x3<T> &mat);

    template <typename T>
    StaticMatrix3x3<T> tensorProduct(const Vector3D<T> &, const Vector3D<T> &);

    template <typename T>
    StaticMatrix3x3<T> cofactorMatrix(const StaticMatrix3x3<T> &mat);

    template <typename T>
    StaticMatrix3x3<T> inverse(const StaticMatrix3x3<T> &mat);

    template <typename T>
    Vector3D<T> diagonal(const StaticMatrix3x3<T> &mat);

    template <typename T>
    StaticMatrix3x3<T> diagonalMatrix(const Vector3D<T> &vec);

    template <typename T>
    StaticMatrix3x3<T> diagonalMatrix(const T t);

    template <typename T>
    T trace(const StaticMatrix3x3<T> &mat);

    template <typename T>
    [[nodiscard]] StaticMatrix3x3<T> kroneckerDeltaMatrix();

    template <typename T>
    [[nodiscard]] StaticMatrix3x3<T> exp(const StaticMatrix3x3<T> &mat);

}   // namespace linearAlgebra

#include "staticMatrix3x3.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _STATIC_MATRIX_3X3_HPP_