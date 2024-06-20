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
    concept Number = std::integral<T> || std::floating_point<T>;

    template <typename T>
    concept Vector3DValueType = requires(T a) {
        { a.x } -> std::same_as<typename T::value_type>;
        { a.y } -> std::same_as<typename T::value_type>;
        { a.z } -> std::same_as<typename T::value_type>;
        {
            linearAlgebra::Vector3D<typename T::value_type>{a.x, a.y, a.z}
        } -> std::same_as<linearAlgebra::Vector3D<typename T::value_type>>;
    };

    template <typename T>
    concept NumberVector3D = Vector3DValueType<T>;

    template <typename T, typename U>
    concept NumberVector3DPair = NumberVector3D<T> && NumberVector3D<U>;

    template <typename T, typename U>
    concept _Addable = requires(T a, U b) {
        { a + b } -> std::same_as<decltype(a + b)>;
    };

    template <typename T, typename U>
    concept Addable =
        NumberVector3DPair<T, U> && (_Addable<T, U> || Vector3DValueType<T>);

}   // namespace pq

#endif   // _CONCEPTS_HPP_