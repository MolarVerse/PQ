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

#ifndef _VECTOR3D_CONCEPTS_HPP_

#define _VECTOR3D_CONCEPTS_HPP_

#include <concepts>

#include "concepts.hpp"
#include "vector3dDepth.hpp"

namespace linearAlgebra
{
    template <class T>
    class Vector3D;
}

namespace pq
{
    /**
     * @brief Concept for Vector3D types
     *
     * @tparam T
     */
    template <class T>
    concept Vector3DConcept =
        std::same_as<T, linearAlgebra::Vector3D<typename T::value_type>>;

    /**
     * @brief Concept for Vector3D types with arithmetic value_type
     *
     * @tparam T
     */
    template <class T>
    concept ArithmeticVector3D =
        Vector3DConcept<T> && pq::is_InnerType_arithmetic_v<T>;

    /**
     * @brief Concept for Vector3D types with depth 1
     *
     * @tparam T
     */
    template <class T>
    concept OneDimArithmeticVector3D =
        ArithmeticVector3D<T> && Vector3DDepth_v<T> == 1;

}   // namespace pq

#endif   // _VECTOR3D_CONCEPTS_HPP_