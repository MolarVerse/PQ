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
    concept Addable = requires(T a, T b) {
        { a + b } -> std::convertible_to<T>;
        { a - b } -> std::convertible_to<T>;
    };

    template <class T>
    concept Multipliable = requires(T a, T b) {
        { a* b } -> std::convertible_to<T>;
    };

    template <class T>
    concept Dividable = Multipliable<T> && requires(T a, T b) {
        { a / b } -> std::convertible_to<T>;
    };

    template <class T>
    struct is_vector3d : std::false_type
    {
    };

    template <class T>
    requires std::is_same_v<T, linearAlgebra::Vector3D<typename T::value_type>>
    struct is_vector3d<T> : std::true_type
    {
    };

    template <class T>
    concept Vector3DType = is_vector3d<T>::value;

}   // namespace pq

#endif   // _CONCEPTS_HPP_