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

#ifndef _INNER_TYPE_HPP_

#define _INNER_TYPE_HPP_

#include <concepts>

#include "base.hpp"

namespace pq
{
    /**
     * @brief type trait to determine the inner type of a template class
     *
     * @tparam T
     */
    template <typename T>
    struct InnerType
    {
        using type = T;
    };

    /**
     * @brief type trait to determine the inner type of a template class
     *
     * @details specialization for template classes
     *
     * @tparam TemplateClass
     * @tparam T
     */
    template <template <typename> class TemplateClass, typename T>
    struct InnerType<TemplateClass<T>>
    {
        using type = typename InnerType<T>::type;
    };

    /**
     * @brief The inner type of a template class
     *
     * @tparam T
     */
    template <typename T>
    using InnerType_t = typename InnerType<T>::type;

    // Check if the inner type of a template class is arithmetic
    template <typename T>
    constexpr bool is_InnerType_arithmetic_v = is_arithmetic_v<InnerType_t<T>>;

}   // namespace pq

#endif   // _INNER_TYPE_HPP_