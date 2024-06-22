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

#include <cmath>

#include "concepts/vector3dConcepts.hpp"
#include "vector3dClass.hpp"

namespace linearAlgebra
{
    /************************
     * comparison operators *
     ************************/

    // operator==
    template <class U>
    requires std::equality_comparable<U>
    bool operator==(const Vector3D<U> &lhs, const Vector3D<U> &rhs);

    template <class U>
    requires std::equality_comparable<U>
    bool operator!=(const Vector3D<U> &lhs, const Vector3D<U> &rhs);

    // operator<
    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator<(const Vector3D<U> &lhs, const Vector3D<V> &rhs);

    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator<(const Vector3D<U> &lhs, const V &rhs);

    // operator<=
    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator>=(const Vector3D<U> &lhs, const Vector3D<V> &rhs);

    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator>=(const Vector3D<U> &lhs, const V &rhs);

    // operator>
    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator>(const Vector3D<U> &lhs, const Vector3D<V> &rhs);

    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator>(const Vector3D<U> &lhs, const V &rhs);

    // operator<=
    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator<=(const Vector3D<U> &lhs, const Vector3D<V> &rhs);

    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator<=(const Vector3D<U> &lhs, const V &rhs);

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

    /*****************
     * fabs function *
     *****************/

    template <pq::ArithmeticVector3D U>
    auto fabs(const U &vec) -> Vector3D<decltype(std::fabs(vec[0]))>;

    /**********************
     * rounding functions *
     **********************/

    template <pq::ArithmeticVector3D U>
    auto round(const U &vec) -> Vector3D<decltype(std::rint(vec[0]))>;

    template <pq::ArithmeticVector3D U>
    auto floor(const U &vec) -> Vector3D<decltype(std::floor(vec[0]))>;

    template <pq::ArithmeticVector3D U>
    auto ceil(const U &vec) -> Vector3D<decltype(std::ceil(vec[0]))>;

    template <pq::ArithmeticVector3D U>
    auto rint(const U &vec) -> Vector3D<decltype(std::rint(vec[0]))>;

    /********************
     * min/max function *
     ********************/

    template <pq::ArithmeticVector3D U>
    auto min(const U &lhs) -> pq::InnerType_t<U>;

    template <pq::ArithmeticVector3D U>
    auto max(const U &lhs) -> pq::InnerType_t<U>;

    /******************
     * norm functions *
     ******************/

    template <pq::ArithmeticVector3D U>
    auto norm(const U &vec) -> decltype(std::sqrt(vec[0] * vec[0]));

    template <pq::ArithmeticVector3D U>
    auto normSquared(const U &vec) -> decltype(vec[0] * vec[0]);

    template <pq::ArithmeticVector3D U>
    auto norms(std::vector<U> v) -> std::vector<decltype(norm(v[0]))>;

    /****************
     * sum function *
     ****************/

    template <pq::ArithmeticVector3D U>
    auto sum(const U &vec) -> decltype(vec[0] + vec[0]);

    /*****************
     * prod function *
     *****************/

    template <pq::ArithmeticVector3D U>
    auto prod(const U &vec) -> decltype(vec[0] * vec[0]);

    /*****************
     * mean function *
     *****************/

    template <pq::ArithmeticVector3D U>
    auto mean(const U &vec) -> decltype(sum(vec) / 3);

    /***************
     * dot product *
     ***************/

    template <pq::ArithmeticVector3D U>
    auto dot(const U &lhs, const U &rhs) -> decltype(lhs[0] * rhs[0]);

    /*****************
     * cross product *
     *****************/

    template <pq::ArithmeticVector3D U>
    auto cross(const U &lhs, const U &rhs)
        -> Vector3D<decltype(lhs[0] * rhs[0])>;

    /***************************
     * angle related functions *
     ***************************/

    template <pq::ArithmeticVector3D U>
    auto cos(const U &vec) -> Vector3D<decltype(std::cos(vec[0]))>;

    template <pq::ArithmeticVector3D U>
    auto cos(const U &lhs, const U &rhs) -> decltype(dot(lhs, rhs));

    template <pq::ArithmeticVector3D U>
    auto angle(const U &v1, const U &v2) -> decltype(std::acos(cos(v1, v2)));

}   // namespace linearAlgebra

#include "vector3d.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _VECTOR3d_HPP_