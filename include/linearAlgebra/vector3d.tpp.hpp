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

#include "concepts.hpp"
#include "vector3d.hpp"

namespace linearAlgebra
{
    /************************
     *                      *
     * comparison operators *
     *                      *
     ************************/

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
    template <pq::Vector3DConcept U, pq::Vector3DConcept V>
    requires(pq::Vector3DDepth_v<U> == pq::Vector3DDepth_v<V>)
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
    template <pq::Vector3DConcept U, pq::Vector3DConcept V>
    requires(pq::Vector3DDepth_v<U> - 1 == pq::Vector3DDepth_v<V>)
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
    template <pq::Vector3DConcept U, pq::Vector3DConcept V>
    requires(pq::Vector3DDepth_v<U> == pq::Vector3DDepth_v<V> - 1)
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
    template <pq::Vector3DConcept U, pq::Arithmetic V>
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
    template <pq::Arithmetic U, pq::Vector3DConcept V>
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
    template <pq::Vector3DConcept U, pq::Vector3DConcept V>
    requires(pq::Vector3DDepth_v<U> == pq::Vector3DDepth_v<V>)
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
    template <pq::Vector3DConcept U, pq::Vector3DConcept V>
    requires(pq::Vector3DDepth_v<U> - 1 == pq::Vector3DDepth_v<V>)
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
    template <pq::Vector3DConcept U, pq::Vector3DConcept V>
    requires(pq::Vector3DDepth_v<U> == pq::Vector3DDepth_v<V> - 1)
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
    template <pq::Vector3DConcept U, pq::Arithmetic V>
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
    template <pq::Arithmetic U, pq::Vector3DConcept V>
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

    // template <class T, class U>
    // requires pq::Addable<Vector3D<T>, Vector3D<U>>
    // Vector3D<Vector3D<std::common_type_t<T, U>>> operator+(
    //     const Vector3D<Vector3D<T>> &lhs,
    //     const Vector3D<Vector3D<U>> &rhs
    // )
    // {
    //     return Vector3D<Vector3D<std::common_type_t<T, U>>>(
    //         lhs.x + rhs.x,
    //         lhs.y + rhs.y,
    //         lhs.z + rhs.z
    //     );
    // }

    // /**
    //  * @brief + operator for a Vector3d object and a scalar
    //  *
    //  * @param const T
    //  * @return Vector3D
    //  */
    // template <class T, class U>
    // requires pq::Addable<T, U>
    // Vector3D<std::common_type_t<T, U>> operator+(
    //     const Vector3D<T> &lhs,
    //     const U            rhs
    // )
    // {
    //     using CommonType = std::common_type_t<T, U>;
    //     return Vector3D<CommonType>(
    //         static_cast<CommonType>(lhs[0]) + rhs,
    //         static_cast<CommonType>(lhs[1]) + rhs,
    //         static_cast<CommonType>(lhs[2]) + rhs
    //     );
    // }

    // template <class T, class U>
    // requires pq::Addable<T, U>
    // Vector3D<Vector3D<std::common_type_t<T, U>>> operator+(
    //     const Vector3D<Vector3D<T>> &lhs,
    //     const Vector3D<U>           &rhs
    // )
    // {
    //     return Vector3D<Vector3D<std::common_type_t<T, U>>>(
    //         lhs[0] + rhs,
    //         lhs[1] + rhs,
    //         lhs[2] + rhs
    //     );
    // }

}   // namespace linearAlgebra

#endif   // _VECTOR3D_TPP_
