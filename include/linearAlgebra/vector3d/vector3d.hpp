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

#ifndef _VECTOR3d_HPP_

#define _VECTOR3d_HPP_

#include "concepts/vector3dConcepts.hpp"
#include "vector3dClass.hpp"

namespace linearAlgebra
{
    /************************
     * comparison operators *
     ************************/

    template <class U>
    requires std::equality_comparable<U>
    bool operator==(const Vector3D<U> &lhs, const Vector3D<U> &rhs);

    /*********************
     * binary + operator *
     *********************/

    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 0)
    auto operator+(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] + rhs[0])>;

    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 1)
    auto operator+(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] + rhs)>;

    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == -1)
    auto operator+(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs + rhs[0])>;

    template <pq::ArithmeticVector3D U, pq::Arithmetic V>
    auto operator+(const U &vec, const V &scalar)
        -> Vector3D<decltype(vec[0] + scalar)>;

    template <pq::Arithmetic U, pq::ArithmeticVector3D V>
    auto operator+(const U &scalar, const V &vec)
        -> Vector3D<decltype(vec[0] + scalar)>;

    /*********************
     * binary - operator *
     *********************/

    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 0)
    auto operator-(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] - rhs[0])>;

    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 1)
    auto operator-(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] - rhs)>;

    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == -1)
    auto operator-(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs - rhs[0])>;

    template <pq::ArithmeticVector3D U, pq::Arithmetic V>
    auto operator-(const U &vec, const V &scalar)
        -> Vector3D<decltype(vec[0] - scalar)>;

    template <pq::Arithmetic U, pq::ArithmeticVector3D V>
    auto operator-(const U &scalar, const V &vec)
        -> Vector3D<decltype(vec[0] - scalar)>;

    /*********************
     * binary * operator *
     *********************/

    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 0)
    auto operator*(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] * rhs[0])>;

    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 1)
    auto operator*(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] * rhs)>;

    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == -1)
    auto operator*(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs * rhs[0])>;

    template <pq::ArithmeticVector3D U, pq::Arithmetic V>
    auto operator*(const U &vec, const V &scalar)
        -> Vector3D<decltype(vec[0] * scalar)>;

    template <pq::Arithmetic U, pq::ArithmeticVector3D V>
    auto operator*(const U &scalar, const V &vec)
        -> Vector3D<decltype(vec[0] * scalar)>;

    /*********************
     * binary / operator *
     *********************/

    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 0)
    auto operator/(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] / rhs[0])>;

    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 1)
    auto operator/(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] / rhs)>;

    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == -1)
    auto operator/(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs / rhs[0])>;

    template <pq::ArithmeticVector3D U, pq::Arithmetic V>
    auto operator/(const U &vec, const V &scalar)
        -> Vector3D<decltype(vec[0] / scalar)>;

    template <pq::Arithmetic U, pq::ArithmeticVector3D V>
    auto operator/(const U &scalar, const V &vec)
        -> Vector3D<decltype(scalar / vec[0])>;

}   // namespace linearAlgebra

#include "vector3d.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _VECTOR3d_HPP_