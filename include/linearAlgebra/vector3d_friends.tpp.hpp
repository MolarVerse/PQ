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

#ifndef _VEC3D_FRIENDS_TPP_

#define _VEC3D_FRIENDS_TPP_

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

    /**
     * @brief + operator for two Vector3d objects
     *
     * @param const Vector3D<U>&
     * @param const Vector3D<V>&
     * @return Vector3D<std::common_type_t<U, V>>
     */
    template <class U, class V>
    requires pq::Addable<U> && pq::Addable<V>
    auto operator+(const Vector3D<U> &lhs, const Vector3D<V> &rhs)
    {
        return Vector3D<std::common_type_t<U, V>>(
            lhs[0] + rhs[0],
            lhs[1] + rhs[1],
            lhs[2] + rhs[2]
        );
    }

    // /**
    //  * @brief + operator for two Vector3d of Vector3d objects
    //  *
    //  * @param const Vector3D<Vector3D<U>>&
    //  * @param const Vector3D<Vector3D<V>>&
    //  * @return Vector3D<Vector3D<std::common_type_t<U, V>>>
    //  */
    // template <class U, class V>
    // requires pq::Addable<U> && pq::Addable<V>
    // Vector3D<Vector3D<std::common_type_t<U, V>>> operator+(
    //     const Vector3D<Vector3D<U>> &lhs,
    //     const Vector3D<Vector3D<V>> &rhs
    // )
    // {
    //     return Vector3D<Vector3D<std::common_type_t<U, V>>>(
    //         lhs[0] + rhs[0],
    //         lhs[1] + rhs[1],
    //         lhs[2] + rhs[2]
    //     );
    // }

    /**
     * @brief + operator for a Vector3d object and a scalar
     *
     * @param const Vector3D<U>&
     * @param const V
     * @return Vector3D<std::common_type_t<U, V>>
     */
    template <class U, class V>
    requires pq::Addable<V> && pq::Vector3DType<U>
    auto operator+(const Vector3D<U> &vec, const V scalar)
    {
        return Vector3D<std::common_type_t<U, V>>(
            vec[0] + scalar,
            vec[1] + scalar,
            vec[2] + scalar
        );
    }

    /**
     * @brief + operator for a Vector3d object and a scalar
     *
     * @param const V
     * @param const Vector3D<U>&
     * @return Vector3D<std::common_type_t<U, V>>
     */
    template <class U, class V>
    requires pq::Addable<U> && pq::Vector3DType<V>
    auto operator+(const U scalar, const Vector3D<V> &vec)
    {
        return Vector3D<std::common_type_t<U, V>>(
            vec[0] + scalar,
            vec[1] + scalar,
            vec[2] + scalar
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

#endif   // _VEC3D_FRIENDS_TPP_
