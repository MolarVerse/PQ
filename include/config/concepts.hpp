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

#ifndef _CONCEPTS_HPP_

#define _CONCEPTS_HPP_

#include <concepts>

namespace linearAlgebra
{
    template <class T>
    class Vector3D;
}

namespace pq
{
    template <class T>
    concept Integral = std::integral<T>;

    template <class T>
    concept FloatingPoint = std::floating_point<T>;

    template <class T>
    concept Arithmetic = Integral<T> || FloatingPoint<T>;

    template <class T>
    concept Vector3DConcept =
        std::same_as<T, linearAlgebra::Vector3D<typename T::value_type>>;

    template <class T>
    concept ArithmeticVector3D = Vector3DConcept<T> || Arithmetic<T>;

    template <typename T>
    struct Vector3DDepth
    {
        static constexpr int value = 0;
    };

    template <typename T>
    struct Vector3DDepth<linearAlgebra::Vector3D<T>>
    {
        static constexpr int value = 1 + Vector3DDepth<T>::value;
    };

    template <typename T>
    constexpr int Vector3DDepth_v = Vector3DDepth<T>::value;

}   // namespace pq

#endif   // _CONCEPTS_HPP_