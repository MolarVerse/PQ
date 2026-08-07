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

#ifndef _VECTOR3D_DEPTH_HPP_

#define _VECTOR3D_DEPTH_HPP_

#include <concepts>

#include "concepts.hpp"

namespace linearAlgebra
{
    template <class T>
    class Vector3D;
}

namespace pq
{
    /**
     * @brief type trait to determine the depth of a Vector3D
     *
     * @tparam T
     */
    template <class T>
    struct Vector3DDepth
    {
        static constexpr int value = 0;
    };

    /**
     * @brief type trait to determine the depth of a Vector3D
     *
     * @details specialization for Vector3D
     *
     * @tparam T
     */
    template <class T>
    struct Vector3DDepth<linearAlgebra::Vector3D<T>>
    {
        static constexpr int value = 1 + Vector3DDepth<T>::value;
    };

    /**
     * @brief constexpr variable to check the depth of a Vector3D
     *
     * @tparam T
     */
    template <class T>
    constexpr int Vector3DDepth_v = Vector3DDepth<T>::value;

    /**
     * @brief constexpr variable to check the depth difference of two Vector3Ds
     *
     * @tparam T
     * @tparam U
     */
    template <class T, class U>
    constexpr int Vector3DDepthDifference_v =
        Vector3DDepth_v<T> - Vector3DDepth_v<U>;

}   // namespace pq

#endif   // _VECTOR3D_DEPTH_HPP_