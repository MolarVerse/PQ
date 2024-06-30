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

#ifndef _BASE_CONCEPTS_HPP_

#define _BASE_CONCEPTS_HPP_

#include <concepts>

namespace pq
{
    /**
     * @brief Concept for integral types
     *
     * @tparam T
     */
    template <class T>
    concept Integral = std::integral<T>;

    /**
     * @brief Concept for floating point types
     *
     * @tparam T
     */
    template <class T>
    concept FloatingPoint = std::floating_point<T>;

    /**
     * @brief Concept for arithmetic types
     *
     * @tparam T
     */
    template <class T>
    concept Arithmetic = Integral<T> || FloatingPoint<T>;

    /**
     * @brief constexpr variable to check if a type is arithmetic
     *
     * @tparam T
     */
    template <class T>
    constexpr bool is_arithmetic_v = Arithmetic<T>;

}   // namespace pq

#endif   // _BASE_CONCEPTS_HPP_