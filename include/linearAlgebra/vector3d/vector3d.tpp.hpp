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

#ifndef _VECTOR3D_TPP_

#define _VECTOR3D_TPP_

#include "concepts/vector3dConcepts.hpp"
#include "vector3d.hpp"

namespace linearAlgebra
{
    /************************
     *                      *
     * comparison operators *
     *                      *
     ************************/

    /**************
     * operator== *
     **************/

    /**
     * @brief operator ==
     *
     * @param const Vector3D<U>&
     * @param const Vector3D<U>&
     * @return bool
     */
    template <class U>
    requires std::equality_comparable<U>
    bool operator==(const Vector3D<U> &lhs, const Vector3D<U> &rhs)
    {
        return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2];
    }

    /**
     * @brief operator !=
     *
     * @param const Vector3D<U>&
     * @param const Vector3D<U>&
     * @return bool
     */
    template <class U>
    requires std::equality_comparable<U>
    bool operator!=(const Vector3D<U> &lhs, const Vector3D<U> &rhs)
    {
        return !(lhs == rhs);
    }

    /**************
     * operator< *
     **************/

    /**
     * @brief operator <
     *
     * @param const Vector3D<U>&
     * @param const Vector3D<V>&
     * @return bool
     */
    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator<(const Vector3D<U> &lhs, const Vector3D<V> &rhs)
    {
        return lhs[0] < rhs[0] && lhs[1] < rhs[1] && lhs[2] < rhs[2];
    }

    /**
     * @brief operator <
     *
     * @param const Vector3D<U>&
     * @param const V
     * @return bool
     */
    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator<(const Vector3D<U> &lhs, const V &rhs)
    {
        return lhs[0] < rhs && lhs[1] < rhs && lhs[2] < rhs;
    }

    /**************
     * operator<= *
     **************/

    /**
     * @brief operator <=
     *
     * @param const Vector3D<U>&
     * @param const Vector3D<V>&
     * @return bool
     */
    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator<=(const Vector3D<U> &lhs, const Vector3D<V> &rhs)
    {
        return !(lhs > rhs);
    }

    /**
     * @brief operator <=
     *
     * @param const Vector3D<U>&
     * @param const V
     * @return bool
     */
    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator<=(const Vector3D<U> &lhs, const V &rhs)
    {
        return !(lhs > rhs);
    }

    /*************
     * operator> *
     *************/

    /**
     * @brief operator >
     *
     * @param const Vector3D<U>&
     * @param const Vector3D<V>&
     * @return bool
     */
    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator>(const Vector3D<U> &lhs, const Vector3D<V> &rhs)
    {
        return lhs[0] > rhs[0] && lhs[1] > rhs[1] && lhs[2] > rhs[2];
    }

    /**
     * @brief operator >
     *
     * @param const Vector3D<U>&
     * @param const V
     * @return bool
     */
    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator>(const Vector3D<U> &lhs, const V &rhs)
    {
        return lhs[0] > rhs && lhs[1] > rhs && lhs[2] > rhs;
    }

    /**************
     * operator>= *
     **************/

    /**
     * @brief operator >=
     *
     * @param const Vector3D<U>&
     * @param const Vector3D<V>&
     * @return bool
     */
    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator>=(const Vector3D<U> &lhs, const Vector3D<V> &rhs)
    {
        return !(lhs < rhs);
    }

    /**
     * @brief operator >=
     *
     * @param const Vector3D<U>&
     * @param const V
     * @return bool
     */
    template <class U, class V>
    requires std::three_way_comparable<U> && std::three_way_comparable<V>
    bool operator>=(const Vector3D<U> &lhs, const V &rhs)
    {
        return !(lhs < rhs);
    }

    /*********************
     *                   *
     * binary + operator *
     *                   *
     *********************/

    /**
     * @brief + operator special case for two Vector3d objects
     *
     * @example Vector3D<int> + Vector3D<double>
     *
     * @param const Vector3D<U>&
     * @param const Vector3D<V>&
     * @return Vector3D<decltype(lhs.x + rhs.x)>
     */
    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 0)
    auto operator+(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] + rhs[0])>
    {
        using ResultType = decltype(lhs[0] + rhs[0]);

        return Vector3D<ResultType>(
            lhs[0] + rhs[0],
            lhs[1] + rhs[1],
            lhs[2] + rhs[2]
        );
    }

    /**
     * @brief + operator special case for nested Vector3d objects
     *
     * @example Vector3D<Vector3D<int>> + Vector3D<double>
     *
     * @param const Vector3D<U>&
     * @param const Vector3D<V>&
     * @return Vector3D<decltype(lhs.x + rhs.x)>
     */
    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 1)
    auto operator+(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] + rhs)>
    {
        using ResultType = decltype(lhs[0] + rhs);

        return Vector3D<ResultType>(lhs[0] + rhs, lhs[1] + rhs, lhs[2] + rhs);
    }

    /**
     * @brief + operator special case for nested Vector3d objects
     *
     * @example Vector3D<int> + Vector3D<Vector3D<double>>
     *
     * @param const Vector3D<U>&
     * @param const Vector3D<V>&
     * @return Vector3D<decltype(lhs.x + rhs.x)>
     */
    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == -1)
    auto operator+(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs + rhs[0])>
    {
        using ResultType = decltype(lhs + rhs[0]);

        return Vector3D<ResultType>(lhs + rhs[0], lhs + rhs[1], lhs + rhs[2]);
    }

    /**
     * @brief + operator for a Vector3d object and a scalar
     *
     * @example Vector3D<int> + double
     *
     * @param const Vector3D<U>&
     * @param const V
     * @return Vector3D<decltype(vec[0] + scalar)>
     */
    template <pq::ArithmeticVector3D U, pq::Arithmetic V>
    auto operator+(const U &vec, const V &scalar)
        -> Vector3D<decltype(vec[0] + scalar)>
    {
        using ResultType = decltype(vec[0] + scalar);

        return Vector3D<ResultType>(
            vec[0] + scalar,
            vec[1] + scalar,
            vec[2] + scalar
        );
    }

    /**
     * @brief + operator for a Vector3d object and a scalar
     *
     * @example double + Vector3D<int>
     *
     * @param const V
     * @param const Vector3D<U>&
     * @return Vector3D<decltype(vec[0] + scalar)>
     */
    template <pq::Arithmetic U, pq::ArithmeticVector3D V>
    auto operator+(const U &scalar, const V &vec)
        -> Vector3D<decltype(vec[0] + scalar)>
    {
        using ResultType = decltype(vec[0] + scalar);

        return Vector3D<ResultType>(
            vec[0] + scalar,
            vec[1] + scalar,
            vec[2] + scalar
        );
    }

    /*********************
     *                   *
     * binary - operator *
     *                   *
     *********************/

    /**
     * @brief - operator special case for two Vector3d objects
     *
     * @example Vector3D<int> - Vector3D<double>
     *
     * @param const Vector3D<U>&
     * @param const Vector3D<V>&
     * @return Vector3D<decltype(lhs.x - rhs.x)>
     */
    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 0)
    auto operator-(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] - rhs[0])>
    {
        using ResultType = decltype(lhs[0] - rhs[0]);

        return Vector3D<ResultType>(
            lhs[0] - rhs[0],
            lhs[1] - rhs[1],
            lhs[2] - rhs[2]
        );
    }

    /**
     * @brief - operator special case for nested Vector3d objects
     *
     * @example Vector3D<Vector3D<int>> - Vector3D<double>
     *
     * @param const Vector3D<U>&
     * @param const Vector3D<V>&
     * @return Vector3D<decltype(lhs.x - rhs.x)>
     */
    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 1)
    auto operator-(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] - rhs)>
    {
        using ResultType = decltype(lhs[0] - rhs);

        return Vector3D<ResultType>(lhs[0] - rhs, lhs[1] - rhs, lhs[2] - rhs);
    }

    /**
     * @brief - operator special case for nested Vector3d objects
     *
     * @example Vector3D<int> - Vector3D<Vector3D<double>>
     *
     * @param const Vector3D<U>&
     * @param const Vector3D<V>&
     * @return Vector3D<decltype(lhs.x - rhs.x)>
     */
    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == -1)
    auto operator-(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs - rhs[0])>
    {
        using ResultType = decltype(lhs - rhs[0]);

        return Vector3D<ResultType>(lhs - rhs[0], lhs - rhs[1], lhs - rhs[2]);
    }

    /**
     * @brief - operator for a Vector3d object and a scalar
     *
     * @example Vector3D<int> - double
     *
     * @param const Vector3D<U>&
     * @param const V
     * @return Vector3D<decltype(vec[0] - scalar)>
     */
    template <pq::ArithmeticVector3D U, pq::Arithmetic V>
    auto operator-(const U &vec, const V &scalar)
        -> Vector3D<decltype(vec[0] - scalar)>
    {
        using ResultType = decltype(vec[0] - scalar);

        return Vector3D<ResultType>(
            vec[0] - scalar,
            vec[1] - scalar,
            vec[2] - scalar
        );
    }

    /**
     * @brief - operator for a Vector3d object and a scalar
     *
     * @example double - Vector3D<int>
     *
     * @param const V
     * @param const Vector3D<U>&
     * @return Vector3D<decltype(vec[0] - scalar)>
     */
    template <pq::Arithmetic U, pq::ArithmeticVector3D V>
    auto operator-(const U &scalar, const V &vec)
        -> Vector3D<decltype(vec[0] - scalar)>
    {
        using ResultType = decltype(vec[0] - scalar);

        return Vector3D<ResultType>(
            vec[0] - scalar,
            vec[1] - scalar,
            vec[2] - scalar
        );
    }

    /*********************
     *                   *
     * binary * operator *
     *                   *
     *********************/

    /**
     * @brief * operator for two Vector3d objects
     *
     * @example Vector3D<int> * Vector3D<double>
     *
     * @param const U
     * @param const V
     * @return Vector3D<decltype(lhs.x * rhs.x)>
     */
    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 0)
    auto operator*(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] * rhs[0])>
    {
        using ResultType = decltype(lhs[0] * rhs[0]);

        return Vector3D<ResultType>(
            lhs[0] * rhs[0],
            lhs[1] * rhs[1],
            lhs[2] * rhs[2]
        );
    }

    /**
     * @brief * operator for a Vector3d object and a scalar
     *
     * @example Vector3D<int> * double
     *
     * @param const U
     * @param const V
     * @return Vector3D<decltype(lhs.x * rhs)>
     */
    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 1)
    auto operator*(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] * rhs)>
    {
        using ResultType = decltype(lhs[0] * rhs);

        return Vector3D<ResultType>(lhs[0] * rhs, lhs[1] * rhs, lhs[2] * rhs);
    }

    /**
     * @brief * operator for a Vector3d object and a scalar
     *
     * @example Vector3D<int> * double
     *
     * @param const U
     * @param const V
     * @return Vector3D<decltype(lhs * rhs.x)>
     */
    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == -1)
    auto operator*(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs * rhs[0])>
    {
        using ResultType = decltype(lhs * rhs[0]);

        return Vector3D<ResultType>(lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]);
    }

    /**
     * @brief Operator * for a Vector3d object and a scalar
     *
     * @example Vector3D<int> * double
     *
     * @tparam U
     * @tparam V
     * @param vec
     * @param scalar
     * @return Vector3D<decltype(vec[0] * scalar)>
     */
    template <pq::ArithmeticVector3D U, pq::Arithmetic V>
    auto operator*(const U &vec, const V &scalar)
        -> Vector3D<decltype(vec[0] * scalar)>
    {
        using ResultType = decltype(vec[0] * scalar);

        return Vector3D<ResultType>(
            vec[0] * scalar,
            vec[1] * scalar,
            vec[2] * scalar
        );
    }

    /**
     * @brief Operator * for a scalar and a Vector3d object
     *
     * @example double * Vector3D<int>
     *
     * @tparam U
     * @tparam V
     * @param scalar
     * @param vec
     * @return Vector3D<decltype(vec[0] * scalar)>
     */
    template <pq::Arithmetic U, pq::ArithmeticVector3D V>
    auto operator*(const U &scalar, const V &vec)
        -> Vector3D<decltype(vec[0] * scalar)>
    {
        using ResultType = decltype(vec[0] * scalar);

        return Vector3D<ResultType>(
            vec[0] * scalar,
            vec[1] * scalar,
            vec[2] * scalar
        );
    }

    /*********************
     *                   *
     * binary / operator *
     *                   *
     *********************/

    /**
     * @brief / operator for two Vector3d objects
     *
     * @example Vector3D<int> / Vector3D<double>
     *
     * @tparam U
     * @tparam V
     * @param lhs
     * @param rhs
     * @return Vector3D<decltype(lhs.x / rhs.x)>
     */
    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 0)
    auto operator/(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] / rhs[0])>
    {
        using ResultType = decltype(lhs[0] / rhs[0]);

        return Vector3D<ResultType>(
            lhs[0] / rhs[0],
            lhs[1] / rhs[1],
            lhs[2] / rhs[2]
        );
    }

    /**
     * @brief / operator for a Vector3d object and a scalar
     *
     * @example Vector3D<int> / double
     *
     * @tparam U
     * @tparam V
     * @param lhs
     * @param rhs
     * @return Vector3D<decltype(lhs.x / rhs)>
     */
    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == 1)
    auto operator/(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] / rhs)>
    {
        using ResultType = decltype(lhs[0] / rhs);

        return Vector3D<ResultType>(lhs[0] / rhs, lhs[1] / rhs, lhs[2] / rhs);
    }

    /**
     * @brief / operator for a Vector3d object and a scalar
     *
     * @example Vector3D<int> / double
     *
     * @tparam U
     * @tparam V
     * @param lhs
     * @param rhs
     * @return Vector3D<decltype(lhs / rhs.x)>
     */
    template <pq::ArithmeticVector3D U, pq::ArithmeticVector3D V>
    requires(pq::Vector3DDepthDifference_v<U, V> == -1)
    auto operator/(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs / rhs[0])>
    {
        using ResultType = decltype(lhs / rhs[0]);

        return Vector3D<ResultType>(lhs / rhs[0], lhs / rhs[1], lhs / rhs[2]);
    }

    /**
     * @brief / operator for a Vector3d object and a scalar
     *
     * @example Vector3D<int> / double
     *
     * @tparam U
     * @tparam V
     * @param vec
     * @param scalar
     * @return Vector3D<decltype(vec[0] / scalar)>
     */
    template <pq::ArithmeticVector3D U, pq::Arithmetic V>
    auto operator/(const U &vec, const V &scalar)
        -> Vector3D<decltype(vec[0] / scalar)>
    {
        using ResultType = decltype(vec[0] / scalar);

        return Vector3D<ResultType>(
            vec[0] / scalar,
            vec[1] / scalar,
            vec[2] / scalar
        );
    }

    /**
     * @brief / operator for a scalar and a Vector3d object
     *
     * @example double / Vector3D<int>
     *
     * @tparam U
     * @tparam V
     * @param scalar
     * @param vec
     * @return Vector3D<decltype(vec[0] / scalar)>
     */
    template <pq::Arithmetic U, pq::ArithmeticVector3D V>
    auto operator/(const U &scalar, const V &vec)
        -> Vector3D<decltype(scalar / vec[0])>
    {
        using ResultType = decltype(scalar / vec[0]);

        return Vector3D<ResultType>(
            scalar / vec[0],
            scalar / vec[1],
            scalar / vec[2]
        );
    }

    /*****************
     *               *
     * fabs function *
     *               *
     *****************/

    /**
     * @brief fabs function for a Vector3d object
     *
     * @example fabs(Vector3D<int>)
     *
     * @tparam U
     * @param vec
     * @return Vector3D<decltype(std::fabs(vec[0]))>
     */
    template <pq::ArithmeticVector3D U>
    auto fabs(const U &vec) -> Vector3D<decltype(std::fabs(vec[0]))>
    {
        using ResultType = decltype(std::fabs(vec[0]));

        return Vector3D<ResultType>(
            std::fabs(vec[0]),
            std::fabs(vec[1]),
            std::fabs(vec[2])
        );
    }

    /**********************
     *                    *
     * rounding functions *
     *                    *
     **********************/

    /**
     * @brief round function for a Vector3d object
     *
     * @example round(Vector3D<double>)
     *
     * @tparam U
     * @param vec
     * @return Vector3D<decltype(std::round(vec[0]))>
     */
    template <pq::ArithmeticVector3D U>
    auto round(const U &vec) -> Vector3D<decltype(std::rint(vec[0]))>
    {
        using ResultType = decltype(std::rint(vec[0]));

        return Vector3D<ResultType>(
            std::rint(vec[0]),
            std::rint(vec[1]),
            std::rint(vec[2])
        );
    }

    /**
     * @brief floor function for a Vector3d object
     *
     * @example floor(Vector3D<double>)
     *
     * @tparam U
     * @param vec
     * @return Vector3D<decltype(std::floor(vec[0]))>
     */
    template <pq::ArithmeticVector3D U>
    auto floor(const U &vec) -> Vector3D<decltype(std::floor(vec[0]))>
    {
        using ResultType = decltype(std::floor(vec[0]));

        return Vector3D<ResultType>(
            std::floor(vec[0]),
            std::floor(vec[1]),
            std::floor(vec[2])
        );
    }

    /**
     * @brief ceil function for a Vector3d object
     *
     * @example ceil(Vector3D<double>)
     *
     * @tparam U
     * @param vec
     * @return Vector3D<decltype(std::ceil(vec[0]))>
     */
    template <pq::ArithmeticVector3D U>
    auto ceil(const U &vec) -> Vector3D<decltype(std::ceil(vec[0]))>
    {
        using ResultType = decltype(std::ceil(vec[0]));

        return Vector3D<ResultType>(
            std::ceil(vec[0]),
            std::ceil(vec[1]),
            std::ceil(vec[2])
        );
    }

    /**
     * @brief rint function for a Vector3d object
     *
     * @example rint(Vector3D<double>)
     *
     * @tparam U
     * @param vec
     * @return Vector3D<decltype(std::rint(vec[0]))>
     */
    template <pq::ArithmeticVector3D U>
    auto rint(const U &vec) -> Vector3D<decltype(std::rint(vec[0]))>
    {
        using ResultType = decltype(std::rint(vec[0]));

        return Vector3D<ResultType>(
            std::rint(vec[0]),
            std::rint(vec[1]),
            std::rint(vec[2])
        );
    }

    /*********************
     *                   *
     * min/max functions *
     *                   *
     *********************/

    /**
     * @brief minimum function for a Vector3d object
     *
     * @example minimum(Vector3D<int>)
     *
     * @tparam U
     * @param vec
     * @return InnerType_t<U>
     */
    template <pq::ArithmeticVector3D U>
    auto minimum(const U &vec) -> pq::InnerType_t<U>
    {
        return std::min(vec[0], std::min(vec[1], vec[2]));
    }

    /**
     * @brief maximum function for a Vector3d object
     *
     * @example maximum(Vector3D<int>)
     *
     * @tparam U
     * @param vec
     * @return InnerType_t<U>
     */
    template <pq::ArithmeticVector3D U>
    auto maximum(const U &vec) -> pq::InnerType_t<U>
    {
        return std::max(vec[0], std::max(vec[1], vec[2]));
    }

    /******************
     *                *
     * norm functions *
     *                *
     ******************/

    /**
     * @brief norm function for a Vector3d object
     *
     * @example norm(Vector3D<double>)
     *
     * @tparam U
     * @param vec
     * @return decltype(std::sqrt(vec[0] * vec[0]))
     */
    template <pq::ArithmeticVector3D U>
    auto norm(const U &vec) -> decltype(std::sqrt(vec[0] * vec[0]))
    {
        return std::sqrt(normSquared(vec));
    }

    /**
     * @brief normSquared function for a Vector3d object
     *
     * @example normSquared(Vector3D<double>)
     *
     * @tparam U
     * @param vec
     * @return decltype(vec[0] * vec[0])
     */
    template <pq::ArithmeticVector3D U>
    auto normSquared(const U &vec) -> decltype(vec[0] * vec[0])
    {
        return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    }

    /****************
     *              *
     * sum function *
     *              *
     ****************/

    template <pq::ArithmeticVector3D U>
    auto sum(const U &vec) -> decltype(vec[0] + vec[0])
    {
        return vec[0] + vec[1] + vec[2];
    }

    /*****************
     *               *
     * prod function *
     *               *
     *****************/

    template <pq::ArithmeticVector3D U>
    auto prod(const U &vec) -> decltype(vec[0] * vec[0])
    {
        return vec[0] * vec[1] * vec[2];
    }

}   // namespace linearAlgebra

#endif   // _VECTOR3D_TPP_